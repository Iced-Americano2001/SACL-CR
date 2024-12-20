import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
import numpy as np
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
    
    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = False,
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        self.momentum = 0.999  # 初始化动量
        self.K = 8192
        self.register_buffer("queue_code", torch.randn(self.K, 768))
        self.register_buffer("queue_nl", torch.randn(self.K, 768))
        self.queue_code = nn.functional.normalize(self.queue_code, dim=0)
        self.queue_nl = nn.functional.normalize(self.queue_nl, dim=0)
        self.register_buffer(
            "queue_ptr_code", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_nl", torch.zeros(1, dtype=torch.long))
    
        self.encoder_momentum = AutoModel.from_pretrained(model_name)
        
        for param in self.encoder_momentum.parameters():
            param.requires_grad = False

        for param, param_m in zip(self.model.parameters(), self.encoder_momentum.parameters()):
            param_m.data.copy_(param.data)

        if not normlized:
            self.temperature = 1.0
            logger.info(
                "reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError(
                    "Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def _copy_params(self, model_source, model_target):
        for param_s, param_t in zip(model_source.parameters(), model_target.parameters()):
            param_t.data.copy_(param_s.data)

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update for the momentum encoder.
        """
        for param, param_m in zip(self.model.parameters(), self.encoder_momentum.parameters()):
            param_m.data = param_m.data * self.momentum + \
                param.data * (1. - self.momentum)

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def encode_with_momentum(self, features):
        if features is None:
            return None
        psg_out = self.encoder_momentum(**features, return_dict=True)
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    @torch.no_grad()
    def _dequeue_and_enqueue_nl(self, nl_vec):
        """
        NL Queue dequeue and enqueue
        """
        nl_size = nl_vec.shape[0]
        ptr = int(self.queue_ptr_nl)
        if nl_size == 16:
            self.queue_nl[ptr:ptr + nl_size, :] = nl_vec
            ptr = (ptr + nl_size) % self.K  
            self.queue_ptr_nl[0] = ptr
        else:
            print('no push nl')

    @torch.no_grad()
    def _dequeue_and_enqueue_code(self, code_vec):
        """
        PL Queue dequeue and enqueue
        """
        code_size = code_vec.shape[0]
        ptr = int(self.queue_ptr_code)
        if code_size == 16:
            self.queue_code[ptr:ptr + code_size, :] = code_vec
            ptr = (ptr + code_size) % self.K  
            self.queue_ptr_code[0] = ptr
        else:
            print('no push code')

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)
        group_size = p_reps.size(0) // q_reps.size(0)
        with torch.no_grad():
            self._momentum_update()
            q_reps_momentum = self.encode_with_momentum(query)
            p_reps_momentum = self.encode_with_momentum(passage)

        if self.negatives_cross_device and self.use_inbatch_neg:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)     
            
        scores_now1 = self.compute_similarity(q_reps[:, None, :,], p_reps_momentum.view(
            q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature  
        scores_neg1 = self.compute_similarity(
            q_reps, self.queue_code.clone().detach()) / self.temperature 
        scores1 = torch.cat([scores_now1, scores_neg1], dim=1) 
        scores1 = scores1.view(q_reps.size(0), -1) 
        target_1 = torch.zeros(scores1.size(
            0), device=scores1.device, dtype=torch.long)
        loss_1 = self.compute_loss(scores1, target_1)

        scores_now2 = self.compute_similarity(q_reps_momentum[:, None, :,], p_reps.view(
            q_reps_momentum.size(0), group_size, -1)).squeeze(1) / self.temperature  
        scores_neg2 = self.compute_similarity(self.queue_nl.clone().detach(), p_reps).reshape(16, 65536) / self.temperature
        scores2 = torch.cat([scores_now2, scores_neg2], dim=1) 
        scores2 = scores2.view(q_reps.size(0), -1) 
        target_2 = torch.zeros(scores2.size(
            0), device=scores2.device, dtype=torch.long)
        loss_2 = self.compute_loss(scores2, target_2)
        loss = loss_1 + loss_2
        p_reps_reshaped = p_reps_momentum.view(16, 8, -1)

        p_reps_negative_samples = p_reps_reshaped[:, -1, :]  
        self._dequeue_and_enqueue_code(p_reps_negative_samples)
        self._dequeue_and_enqueue_nl(q_reps_momentum)
                
                
        return EncoderOutput(
            loss=loss,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

# -*- coding: utf-8 -*-
# @Author : Stanley
# @Email : gzlishouxian@gmail.com
# @File : model.py
# @Software: VSCode
from abc import ABC
from torch import nn
from configure import configure, mode
from transformers import BertModel
import torch


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels):
        super(IntentClassifier, self).__init__()
        dropout_rate = configure['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, device):
        super(SlotClassifier, self).__init__()
        dropout_rate = configure['dropout_rate']
        self.num_slot_labels = num_slot_labels
        self.sigmoid = nn.Sigmoid()
        self.device = device
        if mode == 'convert_onnx':
            self.device = 'cpu'

        if configure['slot_model'] == 'gp':
            self.RoPE = True
            self.inner_dim = 64
            self.linear_1 = nn.Linear(input_dim, self.inner_dim * 2)
            self.linear_2 = nn.Linear(input_dim, num_slot_labels * 2)
        else:
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(input_dim, 2 * num_slot_labels)

    def sinusoidal_position_embedding(self, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, output_dim)).to(self.device)
        return embeddings

    @staticmethod
    def sequence_masking(x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, x, attention_mask):
        if configure['slot_model'] == 'gp':
            seq_len = x.size()[1]
            outputs = self.linear_1(x)  # [2, 43, 128]
            # 取出q和k
            qw, kw = outputs[..., ::2], outputs[..., 1::2]  # [2, 43, 64] 从0,1开始间隔为2
            if self.RoPE:
                pos = self.sinusoidal_position_embedding(seq_len, self.inner_dim)
                # 是将奇数列信息抽取出来也就是cosm拿出来并复制
                cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
                # 是将偶数列信息抽取出来也就是sinm拿出来并复制
                sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
                # 奇数列加上负号 得到第二个q的矩阵
                qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
                qw2 = torch.reshape(qw2, qw.shape)
                # 最后融入位置信息
                qw = qw * cos_pos + qw2 * sin_pos
                # 奇数列加上负号 得到第二个q的矩阵
                kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
                kw2 = torch.reshape(kw2, kw.shape)
                # 最后融入位置信息
                kw = kw * cos_pos + kw2 * sin_pos
            # 最后计算初logits结果
            logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
            dense_out = self.linear_2(x)
            dense_out = torch.einsum('bnh->bhn', dense_out) / 2
            # logits[:, None] 增加一个维度
            logits = logits[:, None] + dense_out[:, ::2, None] + dense_out[:, 1::2, :, None]
            logits = self.add_mask_tril(logits, mask=attention_mask)
            probs = torch.sigmoid(logits)
        else:
            x = self.dropout(x)
            x = self.linear(x)
            batch_size = x.size(0)
            logits = x.view(batch_size, -1, self.num_slot_labels, 2)
            probs = torch.sigmoid(logits)
        return logits, probs


class NLUModel(nn.Module, ABC):
    def __init__(self, entity_labels, intent_labels, device):
        super(NLUModel, self).__init__()
        self.model = BertModel.from_pretrained(configure['ptm'])
        hidden_size = self.model.config.hidden_size
        self.entity_num_labels = entity_labels
        self.intent_classifier = IntentClassifier(hidden_size, intent_labels)
        self.slot_classifier = SlotClassifier(hidden_size, entity_labels, device)

    def forward(self, input_ids):
        attention_mask = torch.where(input_ids > 0, 1, 0)
        model_output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = model_output.last_hidden_state
        pooled_output = model_output.pooler_output
        slot_logits, slot_probs = self.slot_classifier(last_hidden_state, attention_mask)
        intent_logits = self.intent_classifier(pooled_output)
        intent_prob = torch.softmax(intent_logits, dim=1)
        return slot_logits, slot_probs, intent_logits, intent_prob

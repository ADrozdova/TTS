import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, in_feats, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.in_feats = in_feats
        self.dropout = nn.Dropout(dropout)

        self.q = nn.Linear(in_feats, n_heads * in_feats)
        self.k = nn.Linear(in_feats, n_heads * in_feats)
        self.v = nn.Linear(in_feats, n_heads * in_feats)

        self.fc = nn.Linear(n_heads * in_feats, in_feats)

    def forward(self, q, k, v, mask=None):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        sz = self.in_feats // self.n_heads
        q = torch.stack(torch.split(q, sz, dim=-1), dim=1)
        k = torch.stack(torch.split(k, sz, dim=-1), dim=1)
        v = torch.stack(torch.split(v, sz, dim=-1), dim=1)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (sz ** 0.5)

        if mask is not None:
            attention_scores.masked_fill_(mask.unsqueeze(1), -np.inf)

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        output = torch.matmul(attention_scores, v)
        output = torch.cat(torch.split(output, 1, dim=1), dim=-1).squeeze(1)

        output = self.fc(output)

        return output, attention_scores.sum(dim=1)

"""Pure-PyTorch residual relational graph transformer for Ascend NPU."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RelationalGraphLayer(nn.Module):
    def __init__(self, hidden_dim, heads, relation_count, dropout):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_k = nn.Embedding(relation_count, hidden_dim)
        self.relation_v = nn.Embedding(relation_count, hidden_dim)
        self.relation_bias = nn.Embedding(relation_count, heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_src, edge_dst, edge_type):
        node_count = hidden.shape[0]
        query = self.q(hidden).view(node_count, self.heads, self.head_dim)
        key = self.k(hidden).view(node_count, self.heads, self.head_dim)
        value = self.v(hidden).view(node_count, self.heads, self.head_dim)
        relation_k = self.relation_k(edge_type).view(-1, self.heads, self.head_dim)
        relation_v = self.relation_v(edge_type).view(-1, self.heads, self.head_dim)
        logits = (
            query[edge_dst] * (key[edge_src] + relation_k)
        ).sum(-1) / math.sqrt(self.head_dim)
        logits = logits + self.relation_bias(edge_type)
        weights = torch.sigmoid(logits)
        denominator = torch.zeros(
            node_count, self.heads, dtype=hidden.dtype, device=hidden.device
        )
        denominator.index_add_(0, edge_dst, weights)
        weights = weights / denominator[edge_dst].clamp_min(1e-6)
        messages = weights.unsqueeze(-1) * (value[edge_src] + relation_v)
        aggregate = torch.zeros(
            node_count,
            self.heads,
            self.head_dim,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        aggregate.index_add_(0, edge_dst, messages)
        aggregate = aggregate.reshape(node_count, -1)
        hidden = self.norm1(hidden + self.dropout(self.output(aggregate)))
        hidden = self.norm2(hidden + self.dropout(self.ff(hidden)))
        return hidden


class ResidualRGT(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, heads=4, layers=3, relation_count=5, dropout=0.15):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                RelationalGraphLayer(hidden_dim, heads, relation_count, dropout)
                for _ in range(layers)
            ]
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 8),
        )

    def pool(self, hidden, ptr):
        means, maxima = [], []
        for index in range(len(ptr) - 1):
            start = int(ptr[index].item())
            stop = int(ptr[index + 1].item())
            segment = hidden[start:stop]
            means.append(segment.mean(0))
            maxima.append(segment.max(0).values)
        return self.context(torch.cat([torch.stack(means), torch.stack(maxima)], dim=-1))

    def forward(self, batch):
        hidden = self.input(batch["x"])
        for layer in self.layers:
            hidden = layer(hidden, batch["edge_src"], batch["edge_dst"], batch["edge_type"])
        context = self.pool(hidden, batch["ptr"])
        counts = batch["ptr"][1:] - batch["ptr"][:-1]
        node_context = torch.repeat_interleave(context, counts, dim=0)
        base_probability = batch["v11"].clamp(1e-5, 1.0 - 1e-5)
        base_logit = torch.logit(base_probability)
        uncertainty = 1.0 - torch.abs(base_probability - 0.5) * 2.0
        gate = torch.sigmoid(
            self.gate(torch.cat([hidden, node_context, uncertainty.unsqueeze(-1)], dim=-1)).squeeze(-1)
        )
        residual = self.residual(torch.cat([hidden, node_context], dim=-1)).squeeze(-1)
        final_logit = base_logit + gate * residual
        return {
            "logits": final_logit,
            "gate": gate,
            "residual": residual,
            "count_logits": self.count_head(context),
        }


def pairwise_loss(logits, labels, v11, ptr, hard_negatives):
    losses = []
    for index in range(len(ptr) - 1):
        start = int(ptr[index].item())
        stop = int(ptr[index + 1].item())
        local_labels = labels[start:stop]
        positives = torch.nonzero(local_labels > 0.5).flatten()
        negatives = torch.nonzero(local_labels <= 0.5).flatten()
        if not len(positives) or not len(negatives):
            continue
        keep = min(int(hard_negatives), int(len(negatives)))
        hard = negatives[torch.topk(v11[start:stop][negatives], keep).indices]
        differences = logits[start:stop][positives].unsqueeze(1) - logits[start:stop][hard].unsqueeze(0)
        losses.append(F.softplus(-differences).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


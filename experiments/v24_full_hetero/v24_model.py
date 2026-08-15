"""Pure-PyTorch HGT plus causal-path encoder with a V11 residual gate."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TypeLinear(nn.Module):
    def __init__(self, type_count: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(type_count, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(type_count, output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, values, type_ids):
        weights = self.weight[type_ids]
        return torch.bmm(values.unsqueeze(1), weights).squeeze(1) + self.bias[type_ids]


class HeteroGraphLayer(nn.Module):
    def __init__(self, hidden_dim, heads, node_types, relations, dropout):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = TypeLinear(node_types, hidden_dim, hidden_dim)
        self.k = TypeLinear(node_types, hidden_dim, hidden_dim)
        self.v = TypeLinear(node_types, hidden_dim, hidden_dim)
        self.relation_k = nn.Embedding(relations, hidden_dim)
        self.relation_v = nn.Embedding(relations, hidden_dim)
        self.relation_bias = nn.Embedding(relations, heads)
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

    def forward(self, hidden, node_type, edge_src, edge_dst, edge_type):
        count = hidden.shape[0]
        query = self.q(hidden, node_type).view(count, self.heads, self.head_dim)
        key = self.k(hidden, node_type).view(count, self.heads, self.head_dim)
        value = self.v(hidden, node_type).view(count, self.heads, self.head_dim)
        relation_k = self.relation_k(edge_type).view(-1, self.heads, self.head_dim)
        relation_v = self.relation_v(edge_type).view(-1, self.heads, self.head_dim)
        logits = (
            query[edge_dst] * (key[edge_src] + relation_k)
        ).sum(-1) / math.sqrt(self.head_dim)
        logits = logits + self.relation_bias(edge_type)
        weights = torch.sigmoid(logits)
        denominator = torch.zeros(count, self.heads, dtype=hidden.dtype, device=hidden.device)
        denominator.index_add_(0, edge_dst, weights)
        weights = weights / denominator[edge_dst].clamp_min(1e-6)
        messages = weights.unsqueeze(-1) * (value[edge_src] + relation_v)
        aggregate = torch.zeros(
            count, self.heads, self.head_dim, dtype=hidden.dtype, device=hidden.device
        )
        aggregate.index_add_(0, edge_dst, messages)
        aggregate = aggregate.reshape(count, -1)
        hidden = self.norm1(hidden + self.dropout(self.output(aggregate)))
        return self.norm2(hidden + self.dropout(self.ff(hidden)))


class CausalPathEncoder(nn.Module):
    def __init__(self, hidden_dim, heads, layers, node_types, relations, max_length, dropout):
        super().__init__()
        self.max_length = max_length
        self.node_embedding = nn.Embedding(node_types + 1, hidden_dim, padding_idx=node_types)
        self.edge_embedding = nn.Embedding(relations + 2, hidden_dim, padding_idx=relations + 1)
        self.position = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_types, edge_types, lengths):
        node_pad = self.node_embedding.num_embeddings - 1
        edge_start = self.edge_embedding.num_embeddings - 2
        edge_pad = self.edge_embedding.num_embeddings - 1
        safe_nodes = torch.where(node_types >= 0, node_types, torch.full_like(node_types, node_pad))
        relation_tokens = torch.full_like(node_types, edge_pad)
        relation_tokens[:, 0] = edge_start
        relation_tokens[:, 1:] = torch.where(
            edge_types >= 0, edge_types, torch.full_like(edge_types, edge_pad)
        )
        positions = torch.arange(self.max_length, device=node_types.device).unsqueeze(0)
        hidden = (
            self.node_embedding(safe_nodes)
            + self.edge_embedding(relation_tokens)
            + self.position(positions)
        )
        padding = positions >= lengths.unsqueeze(1)
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        return self.norm(hidden[:, 0])


class V24HeteroPathRanker(nn.Module):
    def __init__(
        self,
        alarm_feature_dim=252,
        node_numeric_dim=18,
        hidden_dim=192,
        heads=6,
        graph_layers=4,
        path_layers=2,
        node_types=15,
        relations=7,
        max_path_nodes=12,
        dropout=0.15,
    ):
        super().__init__()
        self.node_type_embedding = nn.Embedding(node_types, hidden_dim)
        self.node_numeric = nn.Linear(node_numeric_dim, hidden_dim)
        self.alarm_input = nn.Sequential(
            nn.Linear(alarm_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                HeteroGraphLayer(hidden_dim, heads, node_types, relations, dropout)
                for _ in range(graph_layers)
            ]
        )
        self.path_encoder = CausalPathEncoder(
            hidden_dim, heads, path_layers, node_types, relations, max_path_nodes, dropout
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        fusion_dim = hidden_dim * 3
        self.residual = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.count_head = nn.Linear(hidden_dim, 8)

    def pool(self, hidden, ptr):
        means, maxima = [], []
        for index in range(len(ptr) - 1):
            start, stop = int(ptr[index].item()), int(ptr[index + 1].item())
            segment = hidden[start:stop]
            means.append(segment.mean(0))
            maxima.append(segment.max(0).values)
        return self.context(torch.cat([torch.stack(means), torch.stack(maxima)], dim=-1))

    def forward(self, batch):
        hidden = self.node_type_embedding(batch["node_type"]) + self.node_numeric(batch["node_numeric"])
        alarm_update = self.alarm_input(batch["alarm_x"])
        hidden = hidden.index_add(0, batch["alarm_node_index"], alarm_update)
        hidden = self.input_norm(hidden)
        for layer in self.graph_layers:
            hidden = layer(
                hidden,
                batch["node_type"],
                batch["edge_src"],
                batch["edge_dst"],
                batch["edge_type"],
            )
        context = self.pool(hidden, batch["node_ptr"])
        alarm_counts = batch["alarm_ptr"][1:] - batch["alarm_ptr"][:-1]
        alarm_context = torch.repeat_interleave(context, alarm_counts, dim=0)
        graph_alarm = hidden[batch["alarm_node_index"]]
        path_alarm = self.path_encoder(
            batch["path_node_type"], batch["path_edge_type"], batch["path_length"]
        )
        fused = torch.cat([graph_alarm, path_alarm, alarm_context], dim=-1)
        base_probability = batch["v11"].clamp(1e-5, 1.0 - 1e-5)
        # torch.logit falls back to CPU on Ascend 910B; this equivalent stays on NPU.
        base_logit = torch.log(base_probability) - torch.log(1.0 - base_probability)
        uncertainty = 1.0 - torch.abs(base_probability - 0.5) * 2.0
        gate = torch.sigmoid(
            self.gate(torch.cat([fused, uncertainty.unsqueeze(-1)], dim=-1)).squeeze(-1)
        )
        residual = self.residual(fused).squeeze(-1)
        logits = base_logit + gate * residual
        return {
            "logits": logits,
            "base_logits": base_logit,
            "gate": gate,
            "residual": residual,
            "count_logits": self.count_head(context),
        }


def pairwise_loss(logits, labels, v11, ptr, hard_negatives):
    losses = []
    for index in range(len(ptr) - 1):
        start, stop = int(ptr[index].item()), int(ptr[index + 1].item())
        local = labels[start:stop]
        positives = torch.nonzero(local > 0.5).flatten()
        negatives = torch.nonzero(local <= 0.5).flatten()
        if not len(positives) or not len(negatives):
            continue
        keep = min(int(hard_negatives), int(len(negatives)))
        hard = negatives[torch.topk(v11[start:stop][negatives], keep).indices]
        differences = logits[start:stop][positives].unsqueeze(1) - logits[start:stop][hard].unsqueeze(0)
        losses.append(F.softplus(-differences).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def multitask_loss(output, batch, weights, positive_weight, hard_negatives):
    bce = F.binary_cross_entropy_with_logits(
        output["logits"], batch["labels"], pos_weight=positive_weight
    )
    pair = pairwise_loss(
        output["logits"], batch["labels"], batch["v11"], batch["alarm_ptr"], hard_negatives
    )
    correction_target = (batch["base_mask"] != batch["labels"]).float()
    gate = F.binary_cross_entropy(output["gate"], correction_target)
    confident = (torch.abs(batch["v11"] - 0.5) >= 0.45) & (correction_target < 0.5)
    preserve = (
        F.smooth_l1_loss(output["logits"][confident], output["base_logits"][confident])
        if torch.any(confident)
        else output["logits"].sum() * 0.0
    )
    total = (
        weights["bce"] * bce
        + weights["pairwise"] * pair
        + weights["gate"] * gate
        + weights["preserve"] * preserve
    )
    return total, {
        "bce": float(bce.detach().cpu()),
        "pairwise": float(pair.detach().cpu()),
        "gate": float(gate.detach().cpu()),
        "preserve": float(preserve.detach().cpu()),
    }

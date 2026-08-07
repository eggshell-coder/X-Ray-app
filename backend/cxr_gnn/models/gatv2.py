"""
models/gatv2.py — GATv2-based graph classifier.

Architecture (must stay byte-identical to training so best_gatv2.pt's
state_dict loads without key mismatches):
    BN(input) → dropout → GATv2Conv1 (K heads, concat) → BN → ELU
              → DropEdge → GATv2Conv2 (1 head) → BN → ELU
              → [mean_pool || max_pool] → Linear → ELU → Dropout → Linear
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge

from cxr_gnn.config import Config, NUM_CLASSES


class GATv2Classifier(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: Optional[int],
        cfg: Config,
        n_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        H = cfg.hidden
        K = cfg.heads
        self._dropout   = cfg.dropout
        self._in_drop   = cfg.in_dropout
        self._dropedge  = cfg.dropedge
        self._edge_dim  = edge_feat_dim

        self.in_bn = nn.BatchNorm1d(node_feat_dim)

        self.gat1 = GATv2Conv(
            node_feat_dim, H, heads=K,
            edge_dim=edge_feat_dim,
            dropout=cfg.dropout,
            concat=True,
        )
        self.bn1 = nn.BatchNorm1d(H * K)

        self.gat2 = GATv2Conv(
            H * K, H, heads=1,
            edge_dim=edge_feat_dim,
            dropout=cfg.dropout,
            concat=True,
        )
        self.bn2 = nn.BatchNorm1d(H)

        self.head = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, n_classes),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
        return_attention: bool = False,
    ):
        x = self.in_bn(x)
        x = F.dropout(x, p=self._in_drop, training=self.training)

        ei, ea = edge_index, edge_attr
        if self.training and self._dropedge > 0 and ea is not None:
            ei, mask = dropout_edge(edge_index, p=self._dropedge)
            ea = ea[mask]
        elif self.training and self._dropedge > 0:
            ei, _ = dropout_edge(edge_index, p=self._dropedge)

        x = F.elu(self.bn1(self.gat1(x, ei, ea)))

        if return_attention:
            x, (attn_edge_index, attn_weights) = self.gat2(
                x, edge_index, edge_attr,
                return_attention_weights=True,
            )
        else:
            x = self.gat2(x, ei, ea)
            attn_edge_index = attn_weights = None

        x = F.elu(self.bn2(x))

        h = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1,
        )

        logits = self.head(h)

        if return_attention:
            return logits, (attn_edge_index, attn_weights)
        return logits

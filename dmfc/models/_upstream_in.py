"""Vendored copy of the upstream higgsfield/interaction_network_pytorch IN classes.

Source: ../Interaction Network.ipynb (cells 10, 12, 14) — Battaglia et al. 2016
n-body interaction network reference implementation, MIT-licensed by the upstream
fork at https://github.com/higgsfield/interaction_network_pytorch.

Modifications relative to the notebook (per CONSTITUTION fork-discipline rules):
  * Removed `Variable(...)` wrappers (deprecated since torch 0.4).
  * `InteractionNetwork.forward` returns `(predicted, effect_receivers)` instead of
    just `predicted`, so wrappers can read the per-object relational state.
  * No mathematical changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationalModel(nn.Module):
    """Relation-centric MLP. Maps (sender, receiver, relation_attrs) -> effect."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_relations, input_size] -> [batch, n_relations, output_size]."""
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        return x.view(batch_size, n_relations, self.output_size)


class ObjectModel(nn.Module):
    """Object-centric MLP. Upstream uses this for the next-velocity head."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_objects, input_size] -> [batch * n_objects, 2]."""
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)


class InteractionNetwork(nn.Module):
    """Battaglia 2016 IN. One-step relational message passing + per-object readout."""

    def __init__(
        self,
        n_objects: int,
        object_dim: int,
        n_relations: int,
        relation_dim: int,
        effect_dim: int,
    ) -> None:
        super().__init__()
        self.relational_model = RelationalModel(2 * object_dim + relation_dim, effect_dim, 150)
        self.object_model = ObjectModel(object_dim + effect_dim, 100)

    def forward(
        self,
        objects: torch.Tensor,
        sender_relations: torch.Tensor,
        receiver_relations: torch.Tensor,
        relation_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted, effect_receivers).

        Shapes:
            objects:            [B, N_obj, D_obj]
            sender_relations:   [B, N_obj, N_rel]
            receiver_relations: [B, N_obj, N_rel]
            relation_info:      [B, N_rel, D_rel]
            predicted:          [B * N_obj, 2]
            effect_receivers:   [B, N_obj, D_eff]
        """
        senders = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], dim=2))
        effect_receivers = receiver_relations.bmm(effects)
        predicted = self.object_model(torch.cat([objects, effect_receivers], dim=2))
        return predicted, effect_receivers

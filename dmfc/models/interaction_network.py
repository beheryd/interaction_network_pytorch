"""Recurrent Mental Pong interaction network.

Mirrors the vendored Battaglia 2016 IN (`dmfc.models._upstream_in`) with:
  * a fixed 2-object / 2-directed-relation graph (ball, paddle),
  * recurrence via concatenation of the previous step's `effect_receivers`
    into the next step's per-object input (Battaglia residual style),
  * a 7-dim Rajalingham output head reading from concatenated per-object
    `effect_receivers` (output indices: [0,1] vis-sim, [2,3] vis, [4,5] sim,
    [6] final intercept y).

Per-object `effect_receivers` is the published "hidden state" used by the
downstream DMFC analysis pipeline (Milestone 4).

NOTE on the relational MLP: the upstream higgsfield notebook ends its
`RelationalModel` with a `nn.ReLU()` after the final `nn.Linear`. For the
upstream solar-system task with non-negative-ish targets that happened not to
matter; for Mental Pong the relational effect must encode signed offsets, and
an early pilot showed 100% of pre-final-ReLU activations negative -> dead
network. We therefore drop the final ReLU here. A second pilot exposed a
related issue: feeding unbounded `prev_effect_receivers` back into the next-
step object features is a recurrent loop with no fixed point, and at lr=1e-3
training diverged after ~10K steps. The fix is gradient clipping in the
training loop (`dmfc.training.train.GRAD_CLIP_NORM`) — standard for recurrent
networks. Tanh on the output was tried and rejected: it saturates at init
(pre-activations have magnitude ~30+) and the network can't learn out of
saturation.

The vendored `_upstream_in.RelationalModel` is left untouched per CONSTITUTION
fork discipline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_OBJECTS: int = 2  # ball, paddle
N_RELATIONS: int = 2  # ball->paddle, paddle->ball (directed)
RELATION_DIM: int = 1  # zeros (no per-relation attributes)
OBS_DIM: int = 7  # (x, y, dx, dy, visible, is_ball, is_paddle)
OUTPUT_DIM: int = 7  # Rajalingham 7-d output (vis-sim, vis, sim, final)
RELATIONAL_HIDDEN: int = 150  # follows higgsfield notebook


class _RelationalMLP(nn.Module):
    """4-layer MLP matching upstream RelationalModel minus the final ReLU.

    See the module-level docstring for why the final ReLU is dropped and why
    we rely on gradient clipping (rather than an output nonlinearity) to keep
    the recurrent state bounded.
    """

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        return x.view(batch_size, n_relations, self.output_size)


def _build_relation_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    """Sender/receiver one-hots for the 2-object directed graph.

    Object 0 = ball, object 1 = paddle. Relation 0 = ball -> paddle, relation 1
    = paddle -> ball. Both matrices are shape (N_OBJECTS, N_RELATIONS).
    """
    sender = torch.zeros(N_OBJECTS, N_RELATIONS)
    receiver = torch.zeros(N_OBJECTS, N_RELATIONS)
    # ball -> paddle
    sender[0, 0] = 1.0
    receiver[1, 0] = 1.0
    # paddle -> ball
    sender[1, 1] = 1.0
    receiver[0, 1] = 1.0
    return sender, receiver


def observation_to_object_features(
    ball_state: torch.Tensor, paddle_state: torch.Tensor
) -> torch.Tensor:
    """Pack env-observation tensors into the per-object feature tensor.

    Args:
        ball_state:   [..., 5] = (x, y, dx, dy, visible_flag).
        paddle_state: [..., 3] = (paddle_x, paddle_y, paddle_vy).

    Returns:
        objects: [..., 2, OBS_DIM] with ball at index 0, paddle at index 1.
                 Columns: (x, y, dx, dy, visible, is_ball, is_paddle).

    Paddle's "dx" and "visible" are filled to 0.0 and 1.0 respectively so the
    feature columns match the ball's. Identity flags break the otherwise-shared
    representation.
    """
    leading = ball_state.shape[:-1]
    zero = torch.zeros(*leading, 1, device=ball_state.device, dtype=ball_state.dtype)
    one = torch.ones(*leading, 1, device=ball_state.device, dtype=ball_state.dtype)
    ball_feats = torch.cat(
        [
            ball_state,  # (x, y, dx, dy, visible)
            one,  # is_ball
            zero,  # is_paddle
        ],
        dim=-1,
    )
    px = paddle_state[..., 0:1]
    py = paddle_state[..., 1:2]
    pvy = paddle_state[..., 2:3]
    paddle_feats = torch.cat(
        [
            px,  # x
            py,  # y
            zero,  # dx
            pvy,  # dy
            one,  # visible
            zero,  # is_ball
            one,  # is_paddle
        ],
        dim=-1,
    )
    return torch.stack([ball_feats, paddle_feats], dim=-2)


class MentalPongIN(nn.Module):
    """Recurrent IN over the Mental Pong 2-object graph."""

    def __init__(
        self,
        effect_dim: int = 10,
        relational_hidden: int = RELATIONAL_HIDDEN,
    ) -> None:
        super().__init__()
        self.effect_dim = effect_dim
        # The relational MLP sees concatenated (sender, receiver, relation_info).
        # Object features at step t are (OBS_DIM + effect_dim) — observed plus
        # the previous step's effect_receivers for that object.
        object_input_dim = OBS_DIM + effect_dim
        self.relational_model = _RelationalMLP(
            input_size=2 * object_input_dim + RELATION_DIM,
            output_size=effect_dim,
            hidden_size=relational_hidden,
        )
        self.output_head = nn.Linear(N_OBJECTS * effect_dim, OUTPUT_DIM)
        sender, receiver = _build_relation_matrices()
        # registered as buffers so they move with .to(device) and aren't trained
        self.register_buffer("sender_relations_template", sender)
        self.register_buffer("receiver_relations_template", receiver)

    def step(self, objects_with_recurrence: torch.Tensor) -> torch.Tensor:
        """One IN message-passing step.

        Args:
            objects_with_recurrence: [B, N_OBJECTS, OBS_DIM + effect_dim]
                — observed features concatenated with previous effect_receivers.

        Returns:
            effect_receivers: [B, N_OBJECTS, effect_dim]
        """
        b = objects_with_recurrence.size(0)
        sender = self.sender_relations_template.unsqueeze(0).expand(b, -1, -1)
        receiver = self.receiver_relations_template.unsqueeze(0).expand(b, -1, -1)
        relation_info = torch.zeros(
            b,
            N_RELATIONS,
            RELATION_DIM,
            device=objects_with_recurrence.device,
            dtype=objects_with_recurrence.dtype,
        )
        senders = sender.permute(0, 2, 1).bmm(objects_with_recurrence)
        receivers = receiver.permute(0, 2, 1).bmm(objects_with_recurrence)
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], dim=2))
        effect_receivers = receiver.bmm(effects)
        return effect_receivers

    def forward(self, object_features_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the IN over a fixed-length sequence.

        Args:
            object_features_seq: [B, T, N_OBJECTS, OBS_DIM] — packed env
                observations per timestep.

        Returns:
            outputs: [B, T, OUTPUT_DIM] — 7-d Rajalingham output per step.
            effect_receivers_seq: [B, T, N_OBJECTS, effect_dim] — per-object
                relational state per step (the DMFC-comparison hidden state).
        """
        b, t, n_obj, d_obs = object_features_seq.shape
        if n_obj != N_OBJECTS or d_obs != OBS_DIM:
            raise ValueError(
                f"expected object_features_seq[..., {N_OBJECTS}, {OBS_DIM}], "
                f"got [..., {n_obj}, {d_obs}]"
            )
        device = object_features_seq.device
        dtype = object_features_seq.dtype
        prev_effect = torch.zeros(b, N_OBJECTS, self.effect_dim, device=device, dtype=dtype)

        outputs = []
        effects_seq = []
        for step_idx in range(t):
            obj_in = torch.cat([object_features_seq[:, step_idx], prev_effect], dim=-1)
            effect_receivers = self.step(obj_in)
            out_t = self.output_head(effect_receivers.reshape(b, -1))
            outputs.append(out_t)
            effects_seq.append(effect_receivers)
            prev_effect = effect_receivers
        outputs_t = torch.stack(outputs, dim=1)
        effects_t = torch.stack(effects_seq, dim=1)
        return outputs_t, effects_t

"""Typed YAML config loader for IN training.

A run config has three sections: `model`, `training`, and a top-level `seed`.
The whole structure is type-checked at load time so a bad config fails before
the training loop spins up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from dmfc.training.losses import LOSS_VARIANTS


@dataclass(frozen=True)
class ModelConfig:
    effect_dim: int
    relational_hidden: int = 150


@dataclass(frozen=True)
class TrainingConfig:
    loss_variant: str
    optimizer: str
    lr: float
    batch_size: int
    max_steps: int
    log_every: int = 100
    checkpoint_every: int = 5000
    conditions_per_batch_source: str = "random"  # "random" or "canonical_79"
    grad_clip_norm: float = 1.0
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.loss_variant not in LOSS_VARIANTS:
            raise ValueError(
                f"loss_variant must be one of {LOSS_VARIANTS}; got {self.loss_variant!r}"
            )
        if self.optimizer not in ("adam",):
            raise ValueError(f"unsupported optimizer {self.optimizer!r}")
        if self.conditions_per_batch_source not in ("random", "canonical_79"):
            raise ValueError(
                f"unsupported conditions_per_batch_source {self.conditions_per_batch_source!r}"
            )
        if self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive; got {self.grad_clip_norm}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative; got {self.weight_decay}")


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    training: TrainingConfig
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | str) -> RunConfig:
    """Load a YAML run config and return the typed dataclass."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "model" not in raw or "training" not in raw:
        raise ValueError(f"invalid config at {path}: missing 'model' or 'training' section")
    return RunConfig(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        seed=int(raw.get("seed", 0)),
    )


def dump_config(cfg: RunConfig, path: Path | str) -> None:
    """Persist the resolved config (post-CLI overrides) into the run directory."""
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


__all__ = [
    "ModelConfig",
    "RunConfig",
    "TrainingConfig",
    "dump_config",
    "load_config",
]

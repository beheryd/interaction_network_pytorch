"""IN training loop with run-artifact serialization.

Usage:
    python -m dmfc.training.train --config configs/in_vis_and_occ.yaml --seed 0

Each run writes a self-describing artifact directory:

    runs/in_<variant>_h<effect_dim>_s<seed>_<ts>_<git>/
        config.yaml          # resolved config (post-CLI overrides)
        git.txt              # commit hash + dirty bit
        seed.txt             # the seed actually used
        log.txt              # stdout/stderr tee
        curves.jsonl         # one JSON line per logged step
        checkpoint.pt        # final model state_dict + optimizer state
        hidden_states.npz    # per-step effect_receivers on the canonical 79
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from dmfc.envs.conditions import ConditionSpec, load_conditions
from dmfc.envs.mental_pong import _mask_ball, _resample_to_bins, integrate_trajectory
from dmfc.envs.random_conditions import sample_batch
from dmfc.models.interaction_network import (
    OBS_DIM,
    MentalPongIN,
    observation_to_object_features,
)
from dmfc.training.config import RunConfig, dump_config, load_config
from dmfc.training.losses import OUTPUT_DIM, compute_loss

DT_MS: int = 50
INTEGRATOR_DT_MS: int = 41
# Default gradient norm clip. The recurrent IN feeds prev_effect_receivers
# back into the next-step object features; without clipping the second pilot
# trained well to step ~10K then diverged when the recurrent state grew
# unbounded. 1.0 is the standard choice for RNN training. Per-run overrides
# live in TrainingConfig.grad_clip_norm.
GRAD_CLIP_NORM: float = 1.0


# ---------------------------------------------------------------------------
# Reproducibility


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Run directory


def _git_state() -> tuple[str, bool]:
    """Return (short hash, dirty flag). Falls back to ('nogit', False) outside a repo."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
            != ""
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit", False


def make_run_dir(cfg: RunConfig, runs_root: Path | str = "runs") -> Path:
    """Create runs/in_<variant>_h<effect_dim>_s<seed>_<ts>_<hash>/ and return it."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha, dirty = _git_state()
    suffix = sha + ("-dirty" if dirty else "")
    name = f"in_{cfg.training.loss_variant}_h{cfg.model.effect_dim}_" f"s{cfg.seed}_{ts}_{suffix}"
    run_dir = Path(runs_root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "git.txt").write_text(f"{sha}\ndirty={dirty}\n")
    (run_dir / "seed.txt").write_text(f"{cfg.seed}\n")
    dump_config(cfg, run_dir / "config.yaml")
    return run_dir


def setup_logging(log_path: Path) -> logging.Logger:
    """File + stdout logging. The file becomes log.txt in the run dir."""
    logger = logging.getLogger("dmfc.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Batch construction


def _condition_to_arrays(
    spec: ConditionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the env's integrate-resample-mask pipeline on one condition.

    Returns:
        ball_in:    (n_bins, 5) — masked ball state (env input).
        ball_true:  (n_bins, 4) — unmasked (x, y, dx, dy) for supervision targets.
        times_ms:   (n_bins,)   — times in ms.
        y_f_oracle: float       — constant intercept y target.
    """
    traj_steps = integrate_trajectory(spec)
    times_ms, traj_bins = _resample_to_bins(traj_steps, DT_MS, INTEGRATOR_DT_MS)
    ball_in = _mask_ball(traj_bins)
    return ball_in, traj_bins, times_ms, spec.y_f_oracle


def build_batch(
    specs: list[ConditionSpec], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a list of `ConditionSpec` into padded torch tensors.

    Paddle is held at y=0 throughout training (no controller in M3). Ball-true
    values are broadcast into target indices [0:2], [2:4], [4:6]; index [6]
    holds the y_f_oracle (constant per condition across the trial).

    Returns:
        object_features: [B, T_max, 2, OBS_DIM] — env input packed for the IN.
        targets:         [B, T_max, OUTPUT_DIM] — supervision targets.
        visible_mask:    [B, T_max] — 1 visible, 0 occluded.
        valid_mask:      [B, T_max] — 1 in-trial, 0 padded.
    """
    arrays = [_condition_to_arrays(s) for s in specs]
    n_bins = [a[0].shape[0] for a in arrays]
    t_max = max(n_bins)
    b = len(specs)
    object_features = np.zeros((b, t_max, 2, OBS_DIM), dtype=np.float32)
    targets = np.zeros((b, t_max, OUTPUT_DIM), dtype=np.float32)
    visible_mask = np.zeros((b, t_max), dtype=np.float32)
    valid_mask = np.zeros((b, t_max), dtype=np.float32)
    for i, (ball_in, ball_true, _, y_f) in enumerate(arrays):
        n = ball_in.shape[0]
        ball_t = torch.from_numpy(ball_in.astype(np.float32))
        # Paddle held at y=0, vy=0, paddle_x=10.
        paddle_t = torch.zeros(n, 3, dtype=torch.float32)
        paddle_t[:, 0] = 10.0
        feats = observation_to_object_features(ball_t, paddle_t).numpy()
        object_features[i, :n] = feats
        targets[i, :n, 0:2] = ball_true[:, 0:2]
        targets[i, :n, 2:4] = ball_true[:, 0:2]
        targets[i, :n, 4:6] = ball_true[:, 0:2]
        targets[i, :n, 6] = y_f
        visible_mask[i, :n] = ball_in[:, 4]
        valid_mask[i, :n] = 1.0
    return (
        torch.from_numpy(object_features).to(device),
        torch.from_numpy(targets).to(device),
        torch.from_numpy(visible_mask).to(device),
        torch.from_numpy(valid_mask).to(device),
    )


# ---------------------------------------------------------------------------
# Training loop


def train(cfg: RunConfig, run_dir: Path, logger: logging.Logger) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    model = MentalPongIN(
        effect_dim=cfg.model.effect_dim,
        relational_hidden=cfg.model.relational_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_params}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    rng = np.random.default_rng(cfg.seed)

    if cfg.training.conditions_per_batch_source == "canonical_79":
        canonical = load_conditions()

    curves_path = run_dir / "curves.jsonl"
    checkpoint_path = run_dir / "checkpoint.pt"
    model.train()
    with open(curves_path, "w") as curves_f:
        for step in range(1, cfg.training.max_steps + 1):
            if cfg.training.conditions_per_batch_source == "random":
                batch_specs = sample_batch(rng, cfg.training.batch_size)
            else:
                idx = rng.choice(len(canonical), cfg.training.batch_size, replace=True)
                batch_specs = [canonical[int(i)] for i in idx]

            obj_feats, targets, visible_mask, valid_mask = build_batch(batch_specs, device)
            outputs, _ = model(obj_feats)
            loss = compute_loss(
                outputs, targets, cfg.training.loss_variant, visible_mask, valid_mask
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            if step % cfg.training.log_every == 0 or step == 1:
                line = json.dumps({"step": step, "train_loss": float(loss.item())})
                curves_f.write(line + "\n")
                curves_f.flush()
                logger.info(f"step {step:>6d}  loss {loss.item():.6f}")

            if step % cfg.training.checkpoint_every == 0:
                _save_checkpoint(model, optimizer, step, checkpoint_path)

    _save_checkpoint(model, optimizer, cfg.training.max_steps, checkpoint_path)
    return model


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Hidden-state dump on the canonical 79


def dump_hidden_states(model: torch.nn.Module, run_dir: Path, logger: logging.Logger) -> None:
    """Forward the trained model on the 79 eval conditions; save hidden_states.npz."""
    device = next(model.parameters()).device
    specs = load_conditions()
    obj_feats, targets, visible_mask, valid_mask = build_batch(specs, device)
    model.eval()
    with torch.no_grad():
        outputs, effect_receivers = model(obj_feats)
    out_path = run_dir / "hidden_states.npz"
    np.savez(
        out_path,
        effect_receivers=effect_receivers.cpu().numpy(),
        outputs=outputs.cpu().numpy(),
        targets=targets.cpu().numpy(),
        visible_mask=visible_mask.cpu().numpy(),
        valid_mask=valid_mask.cpu().numpy(),
        meta_index=np.array([s.meta_index for s in specs], dtype=np.int64),
    )
    logger.info(
        f"wrote {out_path}: effect_receivers shape "
        f"{tuple(effect_receivers.shape)}, std {effect_receivers.std().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IN on Mental Pong")
    parser.add_argument("--config", type=str, required=True, help="path to a YAML config")
    parser.add_argument("--seed", type=int, default=None, help="overrides the seed in the config")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="overrides training.max_steps (useful for smoke tests)",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="root directory for run artifacts",
    )
    parser.add_argument(
        "--effect-dim",
        type=int,
        default=None,
        help="overrides model.effect_dim (HP sweeps)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="overrides training.lr (HP sweeps)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="overrides training.grad_clip_norm (HP sweeps)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="overrides training.weight_decay (HP sweeps)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.effect_dim is not None:
        cfg = replace(cfg, model=replace(cfg.model, effect_dim=args.effect_dim))
    train_overrides: dict = {}
    if args.max_steps is not None:
        train_overrides["max_steps"] = args.max_steps
    if args.lr is not None:
        train_overrides["lr"] = args.lr
    if args.grad_clip_norm is not None:
        train_overrides["grad_clip_norm"] = args.grad_clip_norm
    if args.weight_decay is not None:
        train_overrides["weight_decay"] = args.weight_decay
    if train_overrides:
        cfg = replace(cfg, training=replace(cfg.training, **train_overrides))

    seed_everything(cfg.seed)
    # CPU determinism: PyTorch ops are deterministic on CPU by default for the
    # ops we use. CUDA determinism would require torch.use_deterministic_algorithms,
    # which is not enabled by default since we mostly run on CPU for the pilot.
    if not torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    run_dir = make_run_dir(cfg, args.runs_root)
    logger = setup_logging(run_dir / "log.txt")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"config: {cfg.to_dict()}")

    model = train(cfg, run_dir, logger)
    dump_hidden_states(model, run_dir, logger)
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

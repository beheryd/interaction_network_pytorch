"""Tests for the training loop, run-artifact writer, and seed determinism.

CPU-only per CONSTITUTION:80. The model is exercised at a tiny step budget; the
real training validation is the pilot run, not these tests.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dmfc.envs.conditions import load_conditions
from dmfc.models.interaction_network import (
    OBS_DIM,
    MentalPongIN,
    observation_to_object_features,
)
from dmfc.training.config import load_config

REQUIRED_ARTIFACT_FILES: tuple[str, ...] = (
    "config.yaml",
    "git.txt",
    "seed.txt",
    "log.txt",
    "curves.jsonl",
    "checkpoint.pt",
    "hidden_states.npz",
)


def _run_train(runs_root: Path, seed: int = 0, max_steps: int = 5) -> Path:
    """Invoke `python -m dmfc.training.train` in a subprocess and return the run dir."""
    cmd = [
        sys.executable,
        "-m",
        "dmfc.training.train",
        "--config",
        "configs/in_vis_and_occ.yaml",
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
        "--runs-root",
        str(runs_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert (
        result.returncode == 0
    ), f"train subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    runs = sorted(runs_root.iterdir())
    assert len(runs) >= 1
    return runs[-1]


def test_run_artifact_contains_required_files(tmp_path: Path) -> None:
    run_dir = _run_train(tmp_path)
    for fname in REQUIRED_ARTIFACT_FILES:
        p = run_dir / fname
        assert p.exists(), f"missing artifact {fname}"
        assert p.stat().st_size > 0, f"empty artifact {fname}"


def test_run_dir_naming_carries_variant_seed_and_hidden(tmp_path: Path) -> None:
    run_dir = _run_train(tmp_path, seed=3)
    name = run_dir.name
    assert name.startswith("in_sim-mov_h10_s3_")  # config is in_vis_and_occ.yaml -> sim-mov


def test_config_yaml_round_trips(tmp_path: Path) -> None:
    run_dir = _run_train(tmp_path, seed=7)
    cfg = load_config(run_dir / "config.yaml")
    assert cfg.seed == 7
    assert cfg.training.loss_variant == "sim-mov"
    assert cfg.model.effect_dim == 10


def test_curves_jsonl_is_well_formed(tmp_path: Path) -> None:
    run_dir = _run_train(tmp_path, max_steps=5)
    lines = (run_dir / "curves.jsonl").read_text().splitlines()
    assert len(lines) >= 1
    for line in lines:
        rec = json.loads(line)
        assert set(rec) == {"step", "train_loss"}
        assert isinstance(rec["step"], int)
        assert isinstance(rec["train_loss"], float)


def test_hidden_states_shape_and_alignment(tmp_path: Path) -> None:
    run_dir = _run_train(tmp_path)
    d = np.load(run_dir / "hidden_states.npz")
    canonical = load_conditions()
    assert d["effect_receivers"].shape[0] == 79
    assert d["effect_receivers"].shape[2] == 2  # ball, paddle
    assert d["effect_receivers"].shape[3] == 10  # effect_dim from config
    assert d["outputs"].shape[2] == 7
    assert d["meta_index"].tolist() == [s.meta_index for s in canonical]
    assert d["valid_mask"].sum() > 0


def test_seed_determinism_on_cpu(tmp_path: Path) -> None:
    """Two runs with the same config + seed must produce byte-identical checkpoints."""
    if torch.cuda.is_available():
        pytest.skip("CUDA enables non-deterministic kernels by default")
    run_a = _run_train(tmp_path / "a", seed=0, max_steps=5)
    run_b = _run_train(tmp_path / "b", seed=0, max_steps=5)
    sha_a = hashlib.sha256((run_a / "checkpoint.pt").read_bytes()).hexdigest()
    sha_b = hashlib.sha256((run_b / "checkpoint.pt").read_bytes()).hexdigest()
    assert sha_a == sha_b, "same config + same seed produced different checkpoints"


def test_model_forward_shapes_in_isolation() -> None:
    """Sanity: a single forward pass returns the documented shapes."""
    m = MentalPongIN(effect_dim=10)
    b, t = 2, 7
    ball = torch.zeros(b, t, 5)
    paddle = torch.zeros(b, t, 3)
    feats = observation_to_object_features(ball, paddle)
    assert feats.shape == (b, t, 2, OBS_DIM)
    out, eff = m(feats)
    assert out.shape == (b, t, 7)
    assert eff.shape == (b, t, 2, 10)

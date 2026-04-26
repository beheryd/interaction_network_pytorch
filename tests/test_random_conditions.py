"""Tests for the random Mental Pong condition generator.

Env-tier (top of CONSTITUTION test priority): generated conditions must be
deterministic under a seed and must produce trajectories the env can consume.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmfc.envs.conditions import load_conditions
from dmfc.envs.mental_pong import OCCLUDER_X, PADDLE_X, integrate_trajectory
from dmfc.envs.random_conditions import (
    SAMPLED_META_INDEX,
    _envelope,
    sample_batch,
    sample_random_condition,
)


def test_determinism_under_seed() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = sample_batch(rng_a, 16)
    b = sample_batch(rng_b, 16)
    for ca, cb in zip(a, b, strict=False):
        assert ca == cb


def test_envelope_matches_canonical_79() -> None:
    env = _envelope()
    canonical = load_conditions()
    x0 = np.array([c.x0 for c in canonical])
    y0 = np.array([c.y0 for c in canonical])
    assert env.x0_min == pytest.approx(x0.min())
    assert env.x0_max == pytest.approx(x0.max())
    assert env.y0_min == pytest.approx(y0.min())
    assert env.y0_max == pytest.approx(y0.max())
    assert env.t_f_steps_min >= 1
    assert env.t_f_steps_max <= 200


def test_sampled_conditions_are_valid_pong() -> None:
    """Sampled conditions integrate cleanly and respect Mental Pong invariants."""
    rng = np.random.default_rng(0)
    batch = sample_batch(rng, 64)
    for c in batch:
        assert c.meta_index == SAMPLED_META_INDEX
        assert c.dx > 0  # rightward motion
        assert 0 <= c.n_bounce <= 1
        assert c.t_occ_steps < c.t_f_steps
        # The integrator the env uses must reach the paddle on the recorded final step.
        traj = integrate_trajectory(c)
        assert traj.shape == (c.t_f_steps + 1, 4)
        assert traj[c.t_f_steps, 0] >= PADDLE_X - 1e-9
        # And cross the occluder at the recorded occlusion step.
        assert traj[c.t_occ_steps, 0] >= OCCLUDER_X - 1e-9
        if c.t_occ_steps > 0:
            assert traj[c.t_occ_steps - 1, 0] < OCCLUDER_X


def test_sampled_oracles_match_trajectory() -> None:
    """y_occ_oracle and y_f_oracle on a sampled spec must come from the integrator."""
    rng = np.random.default_rng(7)
    for c in sample_batch(rng, 16):
        traj = integrate_trajectory(c)
        assert traj[c.t_occ_steps, 1] == pytest.approx(c.y_occ_oracle)
        assert traj[c.t_f_steps, 1] == pytest.approx(c.y_f_oracle)


def test_one_shot_sample_and_batch_agree() -> None:
    """sample_batch is just sample_random_condition in a loop with the same RNG."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    batch = sample_batch(rng_a, 4)
    one_by_one = [sample_random_condition(rng_b) for _ in range(4)]
    for c1, c2 in zip(batch, one_by_one, strict=False):
        assert c1 == c2

"""Tests for `dmfc.envs.mental_pong`.

Priority order (CONSTITUTION: env > data > analysis > model):
    1. Determinism — same (seed, condition_id) is bit-identical across instances.
    2. Endpoint oracle — all 79 conditions match `y_occ_rnn_mwk` and `y_f_rnn_mwk`
       within 1e-3 deg. This is the gating test; if it fails, the integrator is wrong.
    3. No-bounce trajectory — dy never sign-flips when n_bounce=0.
    4. One-bounce trajectory — exactly one dy sign-flip, at |y| ~= BALL_REFLECT_Y.
    5. Occluder timing — visible_flag goes 1 -> 0 at the bin where ball_x crosses 5.
    6. Mask integrity — occluded bins have ball_state[:4] all zeroed.
    7. Interception geometry — trial ends with ball_x near +10 within one dt_ms of t_f.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmfc.envs.conditions import RNN_STEP_MS, load_conditions
from dmfc.envs.mental_pong import (
    BALL_REFLECT_Y,
    OCCLUDER_X,
    PADDLE_X,
    MentalPongEnv,
    integrate_trajectory,
)


@pytest.fixture(scope="module")
def specs():
    return load_conditions()


def _rollout(env: MentalPongEnv, condition_id: int, seed: int = 0) -> np.ndarray:
    """Rollout an env with paddle held at 0 and return stacked observations.

    Returns array shape (n_bins, 8): ball_state (5) + paddle_state (3).
    """
    obs = env.reset(condition_id=condition_id, seed=seed)
    rows = [np.concatenate([obs.ball_state, obs.paddle_state])]
    done = False
    while not done:
        obs, _, done, _ = env.step(action=0.0)
        rows.append(np.concatenate([obs.ball_state, obs.paddle_state]))
    return np.array(rows)


def test_determinism_same_seed_and_condition(specs):
    a = _rollout(MentalPongEnv(conditions=specs), condition_id=0, seed=7)
    b = _rollout(MentalPongEnv(conditions=specs), condition_id=0, seed=7)
    c = _rollout(MentalPongEnv(conditions=specs), condition_id=0, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(b, c)


def test_endpoint_oracle_all_79_conditions(specs):
    """Integrated trajectory must hit y_occ and y_f oracles to <=1e-3 deg.

    This is the gating correctness test. If it fails do not move on.
    """
    tol = 1e-3
    failures: list[tuple[int, float, float]] = []
    for spec in specs:
        traj = integrate_trajectory(spec)
        y_occ_err = abs(traj[spec.t_occ_steps, 1] - spec.y_occ_oracle)
        y_f_err = abs(traj[-1, 1] - spec.y_f_oracle)
        if y_occ_err > tol or y_f_err > tol:
            failures.append((spec.meta_index, y_occ_err, y_f_err))
    assert (
        not failures
    ), f"endpoint oracle failures (meta_index, y_occ_err, y_f_err): {failures[:5]}"


def test_no_bounce_condition_dy_does_not_flip(specs):
    spec = next(s for s in specs if s.n_bounce == 0)
    traj = integrate_trajectory(spec)
    dy = traj[:, 3]
    # All dy values should have the same sign (ignore floating-point near-zero)
    signs = np.sign(dy[np.abs(dy) > 1e-9])
    assert np.all(signs == signs[0]), f"dy sign flipped in n_bounce=0 condition {spec.meta_index}"


def test_one_bounce_condition_has_exactly_one_dy_sign_flip(specs):
    spec = next(s for s in specs if s.n_bounce == 1)
    traj = integrate_trajectory(spec)
    dy = traj[:, 3]
    flips = np.sum(np.diff(np.sign(dy)) != 0)
    assert (
        flips == 1
    ), f"expected 1 dy sign flip for n_bounce=1, got {flips} (meta={spec.meta_index})"
    # And that flip should occur near the reflection threshold, not somewhere weird
    flip_step = int(np.argmax(np.abs(np.diff(np.sign(dy))))) + 1
    y_at_flip = traj[flip_step - 1, 1]
    assert (
        abs(abs(y_at_flip) - BALL_REFLECT_Y) < 1.0
    ), f"flip occurred at y={y_at_flip}, far from +/-{BALL_REFLECT_Y}"


def test_occluder_visible_flag_transitions_once_when_ball_crosses_5(specs):
    """Visibility goes True...True, False...False with exactly one transition,
    and the bin where it transitions has true ball_x crossing OCCLUDER_X."""
    env = MentalPongEnv(conditions=specs)
    spec = specs[0]
    env.reset(condition_id=0, seed=0)
    visibilities = []
    done = False
    obs = env._observation()  # initial bin
    visibilities.append(obs.visible)
    while not done:
        obs, _, done, _ = env.step(action=0.0)
        visibilities.append(obs.visible)
    transitions = sum(
        1 for i in range(1, len(visibilities)) if visibilities[i] != visibilities[i - 1]
    )
    assert transitions == 1, f"expected 1 visibility transition, got {transitions}"
    # The transition bin's true (un-masked) ball_x must be the first bin where x >= OCCLUDER_X
    traj = integrate_trajectory(spec)
    # Resample to bin times to find the first bin where x >= OCCLUDER_X
    n_bins = len(visibilities)
    bin_times_ms = np.minimum(np.arange(n_bins) * env.dt_ms, spec.t_f_steps * RNN_STEP_MS)
    step_float = bin_times_ms / RNN_STEP_MS
    lo = np.floor(step_float).astype(int)
    hi = np.minimum(lo + 1, traj.shape[0] - 1)
    alpha = step_float - lo
    bin_x = (1 - alpha) * traj[lo, 0] + alpha * traj[hi, 0]
    expected_first_occluded = int(np.argmax(bin_x >= OCCLUDER_X))
    actual_first_occluded = visibilities.index(False)
    assert (
        actual_first_occluded == expected_first_occluded
    ), f"visibility flipped at bin {actual_first_occluded}, expected {expected_first_occluded}"


def test_mask_integrity_during_occluded_epoch(specs):
    rollout = _rollout(MentalPongEnv(conditions=specs), condition_id=0, seed=0)
    # ball_state cols: x, y, dx, dy, visible_flag
    visible = rollout[:, 4]
    masked_rows = rollout[visible == 0.0]
    visible_rows = rollout[visible == 1.0]
    assert masked_rows.size > 0 and visible_rows.size > 0, "expected both epochs present"
    # Masked rows: first four columns must be exactly zero
    assert np.all(masked_rows[:, :4] == 0.0), "ball_state not zeroed in occluded bins"
    # Visible rows: ball_x and ball_y should be plausibly within arena
    assert np.all(np.abs(visible_rows[:, 0]) <= 11.0)
    assert np.all(np.abs(visible_rows[:, 1]) <= 11.0)


def test_interception_geometry(specs):
    """Trial ends with ball at right wall and t_ms ~= t_f_steps * 41."""
    spec = specs[0]
    env = MentalPongEnv(conditions=specs)
    env.reset(condition_id=0, seed=0)
    done = False
    final_t_ms = None
    while not done:
        obs, _, done, _ = env.step(action=0.0)
        final_t_ms = obs.t_ms
    # Final ball_x at the integrator step (read from raw trajectory, not the masked obs)
    traj = integrate_trajectory(spec)
    assert traj[-1, 0] >= PADDLE_X - 0.5, f"final ball x = {traj[-1, 0]} did not reach paddle plane"
    expected_t_ms = spec.t_f_steps * RNN_STEP_MS
    assert (
        abs(final_t_ms - expected_t_ms) <= 50
    ), f"final t_ms {final_t_ms} vs expected {expected_t_ms}"


def test_paddle_x_is_constant_at_right_wall(specs):
    rollout = _rollout(MentalPongEnv(conditions=specs), condition_id=3, seed=0)
    paddle_x = rollout[:, 5]
    assert np.all(paddle_x == PADDLE_X)


def test_occluder_constants_consistent_with_metadata(specs):
    """Ball x at t_occ should be ~= 5 across all 79 conditions (Zenodo definition)."""
    deviations = []
    for spec in specs:
        traj = integrate_trajectory(spec)
        x_at_occ = traj[spec.t_occ_steps, 0]
        deviations.append(abs(x_at_occ - OCCLUDER_X))
    # Allow up to 0.5 deg slop (one integrator step is dx ~= 0.3 deg)
    assert max(deviations) < 0.5, f"max ball_x deviation at t_occ: {max(deviations)}"

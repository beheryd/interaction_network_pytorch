"""Random Mental Pong condition generator for IN training.

Rajalingham et al. trained their RNNs on ~10000 random conditions and evaluated
on a fixed 79-condition subset. We mirror that protocol: this module samples
fresh `ConditionSpec` instances from the same kinematic envelope as the
canonical 79, filtered to `n_bounce <= 1` to match Rajalingham's trial spec.

Sampled `ConditionSpec` instances carry `meta_index = -1` to signal "not in the
canonical 79". The env (`dmfc.envs.mental_pong`) consumes them transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from dmfc.envs.conditions import RNN_STEP_MS, ConditionSpec, load_conditions
from dmfc.envs.mental_pong import BALL_REFLECT_Y, OCCLUDER_X, PADDLE_X

SAMPLED_META_INDEX: int = -1


@dataclass(frozen=True)
class SamplingEnvelope:
    """Empirical kinematic ranges read from the canonical 79."""

    x0_min: float
    x0_max: float
    y0_min: float
    y0_max: float
    speed_min: float
    speed_max: float
    heading_min_deg: float
    heading_max_deg: float
    t_f_steps_min: int
    t_f_steps_max: int


@lru_cache(maxsize=1)
def _envelope() -> SamplingEnvelope:
    """Read sampling ranges from the canonical 79 conditions."""
    specs = load_conditions()
    x0 = np.array([s.x0 for s in specs])
    y0 = np.array([s.y0 for s in specs])
    dx = np.array([s.dx for s in specs])
    dy = np.array([s.dy for s in specs])
    speed = np.sqrt(dx**2 + dy**2)
    heading = np.degrees(np.arctan2(dy, dx))
    tf = np.array([s.t_f_steps for s in specs])
    return SamplingEnvelope(
        x0_min=float(x0.min()),
        x0_max=float(x0.max()),
        y0_min=float(y0.min()),
        y0_max=float(y0.max()),
        speed_min=float(speed.min()),
        speed_max=float(speed.max()),
        heading_min_deg=float(heading.min()),
        heading_max_deg=float(heading.max()),
        t_f_steps_min=int(tf.min()),
        t_f_steps_max=int(tf.max()),
    )


def _integrate_until_paddle(
    x0: float, y0: float, dx: float, dy: float, max_steps: int = 200
) -> tuple[np.ndarray, int, int, int]:
    """Integrate at the env's native 41 ms step until ball_x >= PADDLE_X.

    Returns (traj, t_occ_steps, t_f_steps, n_bounce) where traj has shape
    (t_f_steps + 1, 4) with columns (x, y, dx, dy) at integer step indices,
    matching the layout of `mental_pong.integrate_trajectory`.

    Raises ValueError if the ball does not reach x=PADDLE_X within max_steps.
    """
    if dx <= 0:
        raise ValueError(f"dx must be > 0 (rightward motion); got {dx}")
    xs = [x0]
    ys = [y0]
    dxs = [dx]
    dys = [dy]
    n_bounce = 0
    t_occ_steps: int | None = None
    t_f_steps: int | None = None
    x, y = x0, y0
    cur_dx, cur_dy = dx, dy
    for step_idx in range(1, max_steps + 1):
        x = x + cur_dx
        y = y + cur_dy
        while y > BALL_REFLECT_Y or y < -BALL_REFLECT_Y:
            if y > BALL_REFLECT_Y:
                y = 2.0 * BALL_REFLECT_Y - y
                cur_dy = -cur_dy
            else:
                y = -2.0 * BALL_REFLECT_Y - y
                cur_dy = -cur_dy
            n_bounce += 1
        xs.append(x)
        ys.append(y)
        dxs.append(cur_dx)
        dys.append(cur_dy)
        if t_occ_steps is None and x >= OCCLUDER_X:
            t_occ_steps = step_idx
        if x >= PADDLE_X:
            t_f_steps = step_idx
            break
    if t_f_steps is None:
        raise ValueError(f"trajectory did not reach paddle within {max_steps} steps")
    if t_occ_steps is None:
        # Defensive: a ball reaching x=10 must have crossed x=5 strictly before.
        raise ValueError("trajectory reached paddle without crossing occluder")
    traj = np.array([xs, ys, dxs, dys], dtype=np.float64).T
    return traj, t_occ_steps, t_f_steps, n_bounce


def sample_random_condition(
    rng: np.random.Generator, envelope: SamplingEnvelope | None = None, max_tries: int = 50
) -> ConditionSpec:
    """Sample one valid ConditionSpec.

    Rejects samples with n_bounce > 1 or trajectories that fall outside the
    envelope's t_f_steps range. Raises RuntimeError if `max_tries` rejections
    are hit (indicates a bug in the envelope or the sampler).
    """
    env = envelope if envelope is not None else _envelope()
    for _ in range(max_tries):
        x0 = float(rng.uniform(env.x0_min, env.x0_max))
        y0 = float(rng.uniform(env.y0_min, env.y0_max))
        speed = float(rng.uniform(env.speed_min, env.speed_max))
        heading = float(rng.uniform(env.heading_min_deg, env.heading_max_deg))
        dx = speed * np.cos(np.radians(heading))
        dy = speed * np.sin(np.radians(heading))
        if dx <= 0:
            continue
        try:
            traj, t_occ_steps, t_f_steps, n_bounce = _integrate_until_paddle(x0, y0, dx, dy)
        except ValueError:
            continue
        if n_bounce > 1:
            continue
        if t_f_steps < env.t_f_steps_min or t_f_steps > env.t_f_steps_max:
            continue
        # Oracles (y at t_occ and t_f) come from the integrated trajectory.
        y_occ_oracle = float(traj[t_occ_steps, 1])
        y_f_oracle = float(traj[t_f_steps, 1])
        return ConditionSpec(
            meta_index=SAMPLED_META_INDEX,
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            t_f_steps=int(t_f_steps),
            t_occ_steps=int(t_occ_steps),
            y_occ_oracle=y_occ_oracle,
            y_f_oracle=y_f_oracle,
            n_bounce=int(n_bounce),
        )
    raise RuntimeError(
        f"failed to sample a valid condition in {max_tries} tries; check the envelope"
    )


def sample_batch(
    rng: np.random.Generator, batch_size: int, envelope: SamplingEnvelope | None = None
) -> list[ConditionSpec]:
    """Convenience: sample `batch_size` independent conditions."""
    env = envelope if envelope is not None else _envelope()
    return [sample_random_condition(rng, envelope=env) for _ in range(batch_size)]


__all__ = [
    "RNN_STEP_MS",
    "SAMPLED_META_INDEX",
    "SamplingEnvelope",
    "sample_batch",
    "sample_random_condition",
]

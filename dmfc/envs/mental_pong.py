"""Mental Pong environment.

Deterministic Gym-style simulator reproducing the 79-condition stimulus set from
Rajalingham et al. 2025. The ball trajectory is integrated at the native 41 ms
RNN-step resolution (matching the Zenodo data) and exposed to the agent at 50 ms
bins (matching the neural-data binning).

Coordinates: visual degrees (MWK frame). Arena is x, y in [-10, +10]. The
paddle is at x = +10 (right wall, full vertical span). The occluder spans
x in [+5, +10]: when ball_x >= 5, ball state is masked to zero and
visible_flag = 0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from dmfc.envs.conditions import (
    RNN_STEP_MS,
    ConditionSpec,
    load_conditions,
)

ARENA_HALF: float = 10.0  # visible frame edge (rendering only)
BALL_REFLECT_Y: float = 9.6  # ball center reflects here, not at the visible wall
# (Empirically fit to all 79 conditions: y_occ_rnn_mwk and y_f_rnn_mwk match to
# ~1e-14 with this value; y=10.0 produces a 0.8 deg post-bounce error.
# Likely accounts for the rendered ball's finite extent vs. wall extent.)
OCCLUDER_X: float = 5.0
PADDLE_X: float = 10.0
PADDLE_HALF_HEIGHT: float = 1.25  # paper renders 0.5 x 2.5 deg paddle
MAX_PADDLE_SPEED_DEG_PER_MS: float = 0.01  # paper: 0.17 deg / 16 ms = 0.01 deg/ms


@dataclass(frozen=True)
class Observation:
    """Per-tick env observation.

    `ball_state` columns: (ball_x, ball_y, ball_dx, ball_dy, visible_flag).
    During the occluded epoch (ball_x >= OCCLUDER_X) the first four are zeroed
    and `visible_flag = 0`.

    `paddle_state` columns: (paddle_x, paddle_y, paddle_vy). `paddle_x` is the
    constant +10°.
    """

    ball_state: np.ndarray  # shape (5,), float64
    paddle_state: np.ndarray  # shape (3,), float64
    t_ms: float
    visible: bool


def integrate_trajectory(spec: ConditionSpec) -> np.ndarray:
    """Integrate the ball at 41 ms steps with y = +/- 10 wall reflection.

    Returns array of shape (t_f_steps + 1, 4) with columns (x, y, dx, dy) at
    integer step indices 0..t_f_steps. Index 0 is the initial state.
    """
    n = spec.t_f_steps + 1
    traj = np.zeros((n, 4), dtype=np.float64)
    x, y, dx, dy = spec.x0, spec.y0, spec.dx, spec.dy
    traj[0] = (x, y, dx, dy)
    for k in range(1, n):
        x = x + dx
        y = y + dy
        while y > BALL_REFLECT_Y or y < -BALL_REFLECT_Y:
            if y > BALL_REFLECT_Y:
                y = 2.0 * BALL_REFLECT_Y - y
                dy = -dy
            else:
                y = -2.0 * BALL_REFLECT_Y - y
                dy = -dy
        traj[k] = (x, y, dx, dy)
    return traj


def _resample_to_bins(
    traj_steps: np.ndarray, dt_ms: int, integrator_dt_ms: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate a (n_steps, 4) integer-step trajectory onto dt_ms bins.

    Returns (times_ms, traj_bins) where times_ms is shape (n_bins,) and
    traj_bins is shape (n_bins, 4). Bins span t = 0 to t_f_ms inclusive in
    steps of dt_ms; the final bin is clamped to t_f_ms exactly so the
    interpolation never reads past the integrated trajectory.
    """
    n_steps = traj_steps.shape[0] - 1
    t_f_ms = n_steps * integrator_dt_ms
    times_ms = np.arange(0, t_f_ms + dt_ms, dt_ms, dtype=np.float64)
    times_ms = np.minimum(times_ms, float(t_f_ms))
    step_float = times_ms / integrator_dt_ms
    lo = np.floor(step_float).astype(np.int64)
    hi = np.minimum(lo + 1, n_steps)
    alpha = (step_float - lo)[:, None]
    traj_bins = (1.0 - alpha) * traj_steps[lo] + alpha * traj_steps[hi]
    return times_ms, traj_bins


def _mask_ball(state_xy_dxdy: np.ndarray) -> np.ndarray:
    """Apply occluder mask: when x >= OCCLUDER_X, zero the (x, y, dx, dy) and set
    visible_flag = 0; else pass through and set visible_flag = 1.

    Input: (n_bins, 4). Output: (n_bins, 5) with appended visible_flag column.
    """
    visible = (state_xy_dxdy[:, 0] < OCCLUDER_X).astype(np.float64)
    masked = state_xy_dxdy * visible[:, None]
    return np.concatenate([masked, visible[:, None]], axis=1)


class MentalPongEnv:
    """Gym-style Mental Pong environment.

    One env instance simulates one condition at a time. Use `reset(condition_id,
    seed)` to switch conditions. The full ball trajectory is integrated up
    front; `step` indexes into it.
    """

    def __init__(
        self,
        dt_ms: int = 50,
        integrator_dt_ms: int = RNN_STEP_MS,
        conditions: list[ConditionSpec] | None = None,
    ) -> None:
        if integrator_dt_ms != RNN_STEP_MS:
            raise ValueError(
                f"integrator_dt_ms must be {RNN_STEP_MS} (Zenodo native); got {integrator_dt_ms}"
            )
        self.dt_ms = dt_ms
        self.integrator_dt_ms = integrator_dt_ms
        self._conditions = conditions if conditions is not None else load_conditions()
        self._spec: ConditionSpec | None = None
        self._traj_steps: np.ndarray | None = None
        self._times_ms: np.ndarray | None = None
        self._ball_bins: np.ndarray | None = None  # (n_bins, 5) with visible_flag
        self._paddle_y: float = 0.0
        self._paddle_vy: float = 0.0
        self._bin_idx: int = 0
        self._rng: np.random.Generator | None = None

    @property
    def conditions(self) -> list[ConditionSpec]:
        return self._conditions

    def reset(self, condition_id: int, seed: int = 0) -> Observation:
        if not 0 <= condition_id < len(self._conditions):
            raise IndexError(
                f"condition_id {condition_id} out of range [0, {len(self._conditions)})"
            )
        self._spec = self._conditions[condition_id]
        self._traj_steps = integrate_trajectory(self._spec)
        self._times_ms, traj_bins = _resample_to_bins(
            self._traj_steps, self.dt_ms, self.integrator_dt_ms
        )
        self._ball_bins = _mask_ball(traj_bins)
        self._paddle_y = 0.0
        self._paddle_vy = 0.0
        self._bin_idx = 0
        self._rng = np.random.default_rng(seed)
        return self._observation()

    def step(self, action: float) -> tuple[Observation, float, bool, dict]:
        if self._spec is None:
            raise RuntimeError("step() called before reset()")
        target_y = float(np.clip(action, -ARENA_HALF, ARENA_HALF))
        max_delta = MAX_PADDLE_SPEED_DEG_PER_MS * self.dt_ms
        delta = float(np.clip(target_y - self._paddle_y, -max_delta, max_delta))
        self._paddle_vy = delta / self.dt_ms
        self._paddle_y = self._paddle_y + delta
        assert self._ball_bins is not None  # for mypy
        self._bin_idx = min(self._bin_idx + 1, self._ball_bins.shape[0] - 1)
        obs = self._observation()
        done = self._bin_idx >= self._ball_bins.shape[0] - 1
        reward = 0.0  # no shaping for Milestone 2
        info: dict = {
            "bin_idx": self._bin_idx,
            "is_occluded": not obs.visible,
            "meta_index": self._spec.meta_index,
        }
        return obs, reward, done, info

    def _observation(self) -> Observation:
        assert self._spec is not None
        assert self._ball_bins is not None
        assert self._times_ms is not None
        ball = self._ball_bins[self._bin_idx]
        paddle_state = np.array([PADDLE_X, self._paddle_y, self._paddle_vy], dtype=np.float64)
        return Observation(
            ball_state=ball.copy(),
            paddle_state=paddle_state,
            t_ms=float(self._times_ms[self._bin_idx]),
            visible=bool(ball[4] > 0.5),
        )

    def render(self, ax=None, show_trajectory: bool = True):
        """Render arena + occluder + paddle + ball trajectory to a matplotlib Axes.

        If `ax` is None, creates a new figure. Returns the Axes for further use.
        """
        import matplotlib.pyplot as plt  # local import keeps import-time cheap
        from matplotlib.patches import Rectangle

        if self._spec is None or self._traj_steps is None:
            raise RuntimeError("render() called before reset()")
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        # arena box
        ax.plot(
            [-ARENA_HALF, ARENA_HALF, ARENA_HALF, -ARENA_HALF, -ARENA_HALF],
            [ARENA_HALF, ARENA_HALF, -ARENA_HALF, -ARENA_HALF, ARENA_HALF],
            "k-",
            lw=2,
        )
        # occluder
        ax.add_patch(
            Rectangle(
                (OCCLUDER_X, -ARENA_HALF),
                ARENA_HALF - OCCLUDER_X,
                2.0 * ARENA_HALF,
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.6,
            )
        )
        # paddle (centered at current y)
        ax.plot(
            [PADDLE_X, PADDLE_X],
            [self._paddle_y - PADDLE_HALF_HEIGHT, self._paddle_y + PADDLE_HALF_HEIGHT],
            "m-",
            lw=4,
            label=f"paddle y={self._paddle_y:.1f}",
        )
        # trajectory + endpoints
        if show_trajectory:
            ax.plot(self._traj_steps[:, 0], self._traj_steps[:, 1], "b-", lw=1, alpha=0.7)
            ax.scatter(self._spec.x0, self._spec.y0, c="green", s=40, zorder=5, label="start")
            t_occ = self._spec.t_occ_steps
            ax.scatter(
                self._traj_steps[t_occ, 0],
                self._traj_steps[t_occ, 1],
                c="orange",
                s=40,
                zorder=5,
                label="t_occ (oracle)",
            )
            ax.scatter(
                self._traj_steps[-1, 0],
                self._traj_steps[-1, 1],
                c="red",
                s=40,
                zorder=5,
                label=f"t_f y={self._spec.y_f_oracle:.2f}",
            )
        ax.set_xlim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_ylim(-ARENA_HALF - 1, ARENA_HALF + 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x (deg)")
        ax.set_ylabel("y (deg)")
        ax.set_title(
            f"meta_index={self._spec.meta_index}  "
            f"n_bounce={self._spec.n_bounce}  t_f={self._spec.t_f_ms}ms"
        )
        ax.legend(loc="lower left", fontsize=8)
        return ax


def animate_condition(
    condition_id: int, out_path: str, fps: int = 20, conditions: list[ConditionSpec] | None = None
) -> None:
    """Write a GIF showing one condition playing in real time.

    Ball is hidden when it enters the occluder (visible_flag = 0). A faint
    full trajectory is drawn underneath so the path is visible after-the-fact.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    env = MentalPongEnv(conditions=conditions)
    env.reset(condition_id=condition_id, seed=0)
    spec = env._conditions[condition_id]
    assert env._ball_bins is not None and env._times_ms is not None and env._traj_steps is not None
    bins = env._ball_bins  # (n_bins, 5) — masked
    traj_steps = env._traj_steps  # (n_steps+1, 4) — un-masked (for trail)
    times_ms = env._times_ms
    n_bins = bins.shape[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        [-ARENA_HALF, ARENA_HALF, ARENA_HALF, -ARENA_HALF, -ARENA_HALF],
        [ARENA_HALF, ARENA_HALF, -ARENA_HALF, -ARENA_HALF, ARENA_HALF],
        "k-",
        lw=2,
    )
    ax.add_patch(
        Rectangle(
            (OCCLUDER_X, -ARENA_HALF),
            ARENA_HALF - OCCLUDER_X,
            2.0 * ARENA_HALF,
            facecolor="lightgray",
            edgecolor="gray",
            alpha=0.6,
        )
    )
    # Faint full trajectory underneath (so the path is recoverable visually)
    ax.plot(traj_steps[:, 0], traj_steps[:, 1], "b-", lw=0.8, alpha=0.25)
    ax.set_xlim(-ARENA_HALF - 1, ARENA_HALF + 1)
    ax.set_ylim(-ARENA_HALF - 1, ARENA_HALF + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")

    (paddle_line,) = ax.plot([], [], "m-", lw=4)
    (ball_dot,) = ax.plot([], [], "go", ms=10)
    title = ax.set_title("")
    fig.tight_layout()

    def init():
        ball_dot.set_data([], [])
        paddle_line.set_data([], [])
        return ball_dot, paddle_line, title

    def update(frame_idx: int):
        ball_x, ball_y, _, _, vis = bins[frame_idx]
        if vis > 0.5:
            ball_dot.set_data([ball_x], [ball_y])
        else:
            ball_dot.set_data([], [])
        paddle_line.set_data(
            [PADDLE_X, PADDLE_X],
            [-PADDLE_HALF_HEIGHT, PADDLE_HALF_HEIGHT],  # paddle stays at y=0 in this CLI demo
        )
        epoch = "visible" if vis > 0.5 else "OCCLUDED"
        title.set_text(
            f"meta={spec.meta_index}  n_bounce={spec.n_bounce}  "
            f"t={times_ms[frame_idx]:.0f}ms  {epoch}"
        )
        return ball_dot, paddle_line, title

    anim = FuncAnimation(
        fig, update, frames=n_bins, init_func=init, interval=1000 / fps, blit=False
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_grid(out_path: str, conditions: list[ConditionSpec] | None = None) -> None:
    """Single PNG showing all 79 trajectories in a 9x9 subplot grid.

    Trajectories color-coded by n_bounce (blue=0, red=1). Useful for spotting
    bad conditions or unusual geometries at a glance.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    specs = conditions if conditions is not None else load_conditions()
    n = len(specs)
    cols = 9
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes_flat = axes.flatten()
    for i, spec in enumerate(specs):
        ax = axes_flat[i]
        traj = integrate_trajectory(spec)
        ax.add_patch(
            Rectangle(
                (OCCLUDER_X, -ARENA_HALF),
                ARENA_HALF - OCCLUDER_X,
                2.0 * ARENA_HALF,
                facecolor="lightgray",
                edgecolor="none",
                alpha=0.5,
            )
        )
        color = "tab:red" if spec.n_bounce == 1 else "tab:blue"
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=0.9)
        ax.scatter([spec.x0], [spec.y0], c="green", s=6, zorder=5)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], c="black", s=6, zorder=5)
        ax.set_xlim(-ARENA_HALF, ARENA_HALF)
        ax.set_ylim(-ARENA_HALF, ARENA_HALF)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i} m={spec.meta_index}", fontsize=6)
    for j in range(n, rows * cols):
        axes_flat[j].axis("off")
    fig.suptitle("Mental Pong — all 79 conditions (blue=no-bounce, red=one-bounce)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Render Mental Pong conditions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--condition", type=int, default=0, help="index into canonical 79-condition set"
    )
    parser.add_argument(
        "--render", action="store_true", help="write a static PNG for one condition"
    )
    parser.add_argument(
        "--animate", action="store_true", help="write an animated GIF for one condition"
    )
    parser.add_argument(
        "--grid", action="store_true", help="write a single PNG with all 79 trajectories"
    )
    parser.add_argument("--out", type=str, default="mental_pong_render.png")
    parser.add_argument("--fps", type=int, default=20, help="animation frames per second")
    args = parser.parse_args()

    if args.grid:
        render_grid(args.out)
        print(f"wrote {args.out}")
        return 0

    if args.animate:
        animate_condition(condition_id=args.condition, out_path=args.out, fps=args.fps)
        print(f"wrote {args.out}")
        return 0

    env = MentalPongEnv()
    obs = env.reset(condition_id=args.condition, seed=args.seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(action=0.0)  # paddle held at center for the smoke render

    if args.render:
        import matplotlib.pyplot as plt

        env.render()
        plt.tight_layout()
        plt.savefig(args.out, dpi=120)
        print(f"wrote {args.out}")
    else:
        print(f"final obs t_ms={obs.t_ms:.1f} visible={obs.visible} ball_state={obs.ball_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

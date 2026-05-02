"""
Hybrid Double CartPole controller.

Swing-up phase  → swingup DP policy (Trial 012 best, full ±π angle range)
Balance phase   → balance DP policy (double_cartpole_cuda, ±23° angle range)

Switch logic:
  Enter balance mode  when |θ1|<0.28 AND |θ2|<0.28 AND |ω1|<3.5 AND |ω2|<3.5
  Return to swingup   when |θ1|>0.35  OR |θ2|>0.35  OR |ω1|>4.5  OR |ω2|>4.5
  (hysteresis prevents rapid toggling at the boundary)
"""

import argparse
from pathlib import Path

import numpy as np

# ── Reuse the swingup dynamics (full angle range, no angle-based termination) ──
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from runners.double_cartpole_swingup_cuda import _step_python

# ── Policy lookup (barycentric interpolation over the DP grid) ─────────────────
from utils.barycentric import get_optimal_action


# ── Policy file paths ──────────────────────────────────────────────────────────
BALANCE_POLICY_PATH = Path("results/double_cartpole_cuda_policy.npz")
SWINGUP_POLICY_PATH = Path("results/double_cartpole_swingup_cuda_policy.npz")


class DPPolicy:
    """Thin wrapper around a saved .npz policy file."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.policy      = d["policy"]
        self.action_space = d["action_space"]
        self.bounds_low  = d["bounds_low"]
        self.bounds_high = d["bounds_high"]
        self.grid_shape  = d["grid_shape"]
        self.strides     = d["strides"]
        self.corner_bits = d["corner_bits"]

    def action(self, state: np.ndarray) -> float:
        return float(get_optimal_action(
            state,
            self.policy, self.action_space,
            self.bounds_low, self.bounds_high,
            self.grid_shape, self.strides, self.corner_bits,
        ))


# ── Switch thresholds ──────────────────────────────────────────────────────────
# Balance DP grid covers ±0.401 rad (±23°). Stay well inside with margin.
_ENTER_TH  = 0.32   # rad (~18°) — enter balance mode (within ±23° grid bound)
_ENTER_W   = 4.0    # rad/s      (within ±5 rad/s grid bound)
_EXIT_TH   = 0.38   # rad (~22°) — hysteresis: exit balance mode
_EXIT_W    = 5.0    # rad/s


def _use_balance(th1, w1, th2, w2, currently_balancing: bool) -> bool:
    if currently_balancing:
        # Stay in balance until clearly outside
        return not (abs(th1) > _EXIT_TH or abs(th2) > _EXIT_TH or
                    abs(w1)  > _EXIT_W  or abs(w2)  > _EXIT_W)
    else:
        return (abs(th1) < _ENTER_TH and abs(th2) < _ENTER_TH and
                abs(w1)  < _ENTER_W  and abs(w2)  < _ENTER_W)


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(n_episodes: int = 3, steps: int = 1000,
             render: bool = False, record_path: Path = None,
             seed: int = 42) -> None:

    balance = DPPolicy(BALANCE_POLICY_PATH)
    swingup = DPPolicy(SWINGUP_POLICY_PATH)

    rng = np.random.default_rng(seed)

    recording  = record_path is not None
    do_render  = render or recording
    all_frames = []

    if do_render:
        import pygame
        from runners.double_cartpole_swingup_cuda import _render_frame_pygame, _SCREEN_W, _SCREEN_H
        pygame.init()
        flags = 0 if render else pygame.NOFRAME
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H), flags)
        pygame.display.set_caption("Hybrid Double CartPole")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0], dtype=np.float32)
        state[:2] += rng.uniform(-0.05, 0.05, size=2).astype(np.float32)

        balancing     = False
        balance_steps = 0
        total_reward  = 0.0
        terminated    = False

        for step in range(steps):
            if do_render:
                _render_frame_pygame(screen, clock, state)
                if recording:
                    frame = pygame.surfarray.array3d(screen)
                    all_frames.append(frame.transpose(1, 0, 2))

            x, xd, th1, w1, th2, w2 = state
            balancing = _use_balance(th1, w1, th2, w2, balancing)

            policy = balance if balancing else swingup
            force  = policy.action(state)

            if balancing:
                balance_steps += 1

            state, reward, terminated = _step_python(state, force)
            total_reward += reward

            if terminated:
                break

        x, xd, th1, w1, th2, w2 = state
        mode = "BALANCE" if balancing else "SWINGUP"
        print(
            f"Ep {ep+1}: {step+1} steps | reward={total_reward:.0f} | "
            f"balance_steps={balance_steps} | last_mode={mode} | "
            f"th1={np.degrees(th1):+.1f}° th2={np.degrees(th2):+.1f}° "
            f"w1={w1:+.2f} w2={w2:+.2f}"
        )

    if do_render:
        pygame.quit()

    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.suffix.lower() == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=50, loop=0)
        elif record_path.suffix.lower() == ".mp4":
            writer = imageio.get_writer(str(record_path), fps=50, codec="libx264",
                                        quality=8, pixelformat="yuv420p")
            for frame in all_frames:
                writer.append_data(frame)
            writer.close()
        else:
            imageio.mimsave(str(record_path), all_frames, fps=50)
        print(f"Video saved to {record_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Double CartPole - DP swing-up + DP balance"
    )
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes with pygame")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save video to PATH (.gif or .mp4)")
    parser.add_argument("--episodes",  type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--steps",     type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default: 42)")
    # The --random, --bins, --no-plot, --retrain, --save-path flags do not
    # apply to the hybrid runner: it loads two pre-trained DP policies and
    # never trains anything itself. They are accepted but ignored, so the
    # CLI remains uniform across runners.
    parser.add_argument("--random",    type=int, nargs="?", const=5, default=None,
                        metavar="N",
                        help="(ignored) hybrid runner does not support random baseline")
    parser.add_argument("--bins",      type=int, default=None,
                        help="(ignored) policies are loaded with their saved bins")
    parser.add_argument("--no-plot",   action="store_true",
                        help="(ignored) hybrid runner does not produce plots")
    parser.add_argument("--retrain",   action="store_true",
                        help="(ignored) hybrid runner only loads policies, never trains")
    parser.add_argument("--save-path", type=Path, default=None,
                        help="(ignored) hybrid uses fixed swingup/balance policy paths")
    args = parser.parse_args()

    if args.random is not None:
        print("[!] --random is not supported for the hybrid runner — exiting.")
    else:
        evaluate(
            n_episodes=args.episodes,
            steps=args.steps,
            render=args.render,
            record_path=args.record,
            seed=args.seed,
        )

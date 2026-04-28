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
             render: bool = False, seed: int = 42) -> None:

    balance = DPPolicy(BALANCE_POLICY_PATH)
    swingup = DPPolicy(SWINGUP_POLICY_PATH)

    rng = np.random.default_rng(seed)

    if render:
        # Reuse the swingup renderer (same cart-pole geometry)
        import pygame
        from runners.double_cartpole_swingup_cuda import _render_frame_pygame, _SCREEN_W, _SCREEN_H
        pygame.init()
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H))
        pygame.display.set_caption("Hybrid Double CartPole")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        # Start hanging down (swing-up task)
        state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0], dtype=np.float32)
        state[:2] += rng.uniform(-0.05, 0.05, size=2).astype(np.float32)

        balancing     = False
        balance_steps = 0
        total_reward  = 0.0
        terminated    = False

        for step in range(steps):
            if render:
                _render_frame_pygame(screen, clock, state)

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

    if render:
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps",    type=int, default=1000)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    evaluate(args.episodes, args.steps, args.render, args.seed)

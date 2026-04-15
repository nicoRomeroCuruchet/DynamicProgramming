"""
runners/continuous_mountain_car_cuda.py — CUDA Policy Iteration for
MountainCarContinuous-v0.

Embeds dynamics directly in a CUDA kernel. Continuous action space is
discretized to 21 evenly-spaced force values in [-1, 1].

State  : [position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]]
Actions: 21 values in [-1.0, 1.0]  (applied force, clipped in dynamics)
Reward : -0.1 * force² + 100 * terminated
Terminal: position >= 0.45 and velocity >= 0
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration2D, CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

BINS_SPACE = {
    "position": np.linspace(-1.2,  0.6,  200, dtype=np.float32),
    "velocity": np.linspace(-0.07, 0.07, 200, dtype=np.float32),
}

# Discretized force values (continuous action space [-1, 1])
ACTION_SPACE = np.linspace(-1.0, 1.0, 21, dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class ContinuousMountainCarCuda(CudaPolicyIteration2D):
    """
    CudaPolicyIteration2D for MountainCarContinuous-v0.

    Dynamics (exact match to gymnasium Continuous_MountainCarEnv.step):
        force    = clip(force, -1.0, 1.0)
        velocity += force * 0.0015 - 0.0025 * cos(3 * position)
        velocity  = clip(velocity, -0.07, 0.07)
        position += velocity
        position  = clip(position, -1.2, 0.6)
        if position <= -1.2: velocity = 0   (left-wall bounce)
        terminated = (position >= 0.45) and (velocity >= 0)
        reward = -0.1 * force² + 100 * terminated

    Note: reward uses the clipped force (matching gymnasium exactly).
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        __device__ void step_dynamics(
            float pos, float vel, float force,
            float* next_pos, float* next_vel,
            float* reward, bool* terminated
        ) {
            force = fmaxf(-1.0f, fminf(1.0f, force));

            vel += force * 0.0015f - 0.0025f * cosf(3.0f * pos);
            vel  = fmaxf(-0.07f, fminf(0.07f, vel));
            pos += vel;
            pos  = fmaxf(-1.2f,  fminf(0.6f,  pos));
            if (pos <= -1.2f) vel = 0.0f;

            bool goal = (pos >= 0.45f) && (vel >= 0.0f);

            *next_pos   = pos;
            *next_vel   = vel;
            *terminated = goal;
            *reward     = -0.1f * force * force + (goal ? 100.0f : 0.0f);
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        pos = states[:, 0]
        vel = states[:, 1]
        mask = (pos >= 0.45) & (vel >= 0.0)
        # Terminal value = 0; the +100 goal reward is given during the
        # transition that *reaches* the goal, not as a standing terminal bonus.
        return mask, 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/continuous_mountain_car_cuda_policy.npz"),
) -> ContinuousMountainCarCuda:
    config = CudaPIConfig(
        gamma        = 0.99,
        theta        = 1e-4,
        max_eval_iter= 5_000,
        max_pi_iter  = 50,
        log_interval = 200,
    )

    pi = ContinuousMountainCarCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pi: ContinuousMountainCarCuda, n_episodes: int = 5, render: bool = False
) -> None:
    import gymnasium as gym
    from utils.barycentric import get_optimal_action
    mode = "human" if render else None
    env  = gym.make("MountainCarContinuous-v0", render_mode=mode)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for step in range(999):
            action_val = float(get_optimal_action(
                np.array(obs, dtype=np.float32),
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))
            obs, reward, terminated, truncated, _ = env.step(
                np.array([action_val], dtype=np.float32)
            )
            total_reward += reward
            if terminated or truncated:
                break

        outcome = "SUCCESS" if terminated else "timeout"
        print(
            f"Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.2f} | {outcome}"
        )

    env.close()


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_function(
    pi: ContinuousMountainCarCuda,
    save_path: Path = Path("results/continuous_mountain_car_cuda_vf.png"),
) -> None:
    pos_bins = np.unique(pi.states_space[:, 0])
    vel_bins = np.unique(pi.states_space[:, 1])
    V = pi.value_function.reshape(len(pos_bins), len(vel_bins))

    P, Vl = np.meshgrid(pos_bins, vel_bins, indexing="ij")

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, Vl, V, cmap="plasma", alpha=0.85)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value")
    ax.set_title("Continuous Mountain Car CUDA — Value Function")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value function saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Mountain Car — CUDA Policy Iteration")
    parser.add_argument("--render",    action="store_true", help="Render evaluation episodes")
    parser.add_argument("--episodes",  type=int, default=5, help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--retrain",   action="store_true", help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path, default=Path("results/continuous_mountain_car_cuda_policy.npz"))
    args = parser.parse_args()

    if args.save_path.exists() and not args.retrain:
        print(f"[+] Loading existing policy from {args.save_path}")
        pi = ContinuousMountainCarCuda.load(args.save_path)
    else:
        print("[*] Training new policy...")
        pi = train(args.save_path)

    evaluate(pi, n_episodes=args.episodes, render=args.render)
    plot_value_function(pi)

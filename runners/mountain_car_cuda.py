"""
runners/mountain_car_cuda.py — CUDA Policy Iteration for MountainCar-v0.

Embeds MountainCar dynamics directly in a CUDA kernel (no env.step() calls
during DP). Follows the same architecture as PolicyIterationStall.

State  : [position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]]
Actions: {-1.0, 0.0, 1.0}  (push left / no push / push right)
Reward : -1 per step
Terminal: position >= 0.5 and velocity >= 0
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

# Actions: actual force values applied to the car
ACTION_SPACE = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class MountainCarCuda(CudaPolicyIteration2D):
    """
    CudaPolicyIteration2D for MountainCar-v0.

    Dynamics (exact match to gymnasium MountainCarEnv.step):
        velocity += force * 0.001 - 0.0025 * cos(3 * position)
        velocity  = clip(velocity, -0.07, 0.07)
        position += velocity
        position  = clip(position, -1.2, 0.6)
        if position <= -1.2: velocity = 0   (left-wall bounce)
        terminated = (position >= 0.5) and (velocity >= 0)
        reward = -1  (always)
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        __device__ void step_dynamics(
            float pos, float vel, float force,
            float* next_pos, float* next_vel,
            float* reward, bool* terminated
        ) {
            vel += force * 0.001f - 0.0025f * cosf(3.0f * pos);
            vel  = fmaxf(-0.07f, fminf(0.07f, vel));
            pos += vel;
            pos  = fmaxf(-1.2f,  fminf(0.6f,  pos));
            if (pos <= -1.2f) vel = 0.0f;

            *next_pos   = pos;
            *next_vel   = vel;
            *terminated = (pos >= 0.5f) && (vel >= 0.0f);
            *reward     = -1.0f;
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        pos = states[:, 0]
        vel = states[:, 1]
        mask = (pos >= 0.5) & (vel >= 0.0)
        # Terminal value = 0: no more steps, no more -1 rewards
        return mask, 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/mountain_car_cuda_policy.npz"),
) -> MountainCarCuda:
    config = CudaPIConfig(
        gamma        = 0.99,
        theta        = 1e-4,
        max_eval_iter= 5_000,
        max_pi_iter  = 50,
        log_interval = 200,
    )

    pi = MountainCarCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(pi: MountainCarCuda, n_episodes: int = 5, render: bool = False) -> None:
    import gymnasium as gym
    from utils.barycentric import get_optimal_action
    mode = "human" if render else None
    env  = gym.make("MountainCar-v0", render_mode=mode)

    # Map continuous action value to discrete gym action index {0,1,2}
    def action_to_gym(a: float) -> int:
        # a ∈ {-1.0, 0.0, 1.0} → gym action {0, 1, 2}
        return int(round(a + 1.0))

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for step in range(200):
            action_val = get_optimal_action(
                np.array(obs, dtype=np.float32),
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            )
            obs, reward, terminated, truncated, _ = env.step(action_to_gym(float(action_val)))
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.1f}")

    env.close()


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_function(
    pi: MountainCarCuda,
    save_path: Path = Path("results/mountain_car_cuda_vf.png"),
) -> None:
    pos_bins = np.unique(pi.states_space[:, 0])
    vel_bins = np.unique(pi.states_space[:, 1])
    V = pi.value_function.reshape(len(pos_bins), len(vel_bins))

    P, Vl = np.meshgrid(pos_bins, vel_bins, indexing="ij")

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, Vl, V, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value")
    ax.set_title("Mountain Car CUDA — Value Function")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value function saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    save_path = Path("results/mountain_car_cuda_policy.npz")

    if save_path.exists():
        print(f"[+] Loading existing policy from {save_path}")
        pi = MountainCarCuda.load(save_path)
    else:
        print("[*] Training new policy...")
        pi = train(save_path)

    evaluate(pi)
    plot_value_function(pi)

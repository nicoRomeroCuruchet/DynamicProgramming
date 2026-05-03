"""
runners/pendulum_cuda.py — CUDA Policy Iteration for Pendulum-v1.

Embeds Pendulum dynamics directly in a CUDA kernel. Works on the underlying
(theta, theta_dot) state — no cos/sin wrapping as in the Gymnasium observation.

State  : [theta ∈ [-π, π], theta_dot ∈ [-8, 8]]
           theta = 0 → pointing UP (goal / unstable equilibrium)
           theta = ±π → pointing DOWN (stable rest)
Actions: 21 torque values in [-2.0, 2.0]
Reward : -(theta² + 0.1 * theta_dot² + 0.001 * torque²)
           theta is angle_normalized to [-π, π] before squaring
Terminal: none (episodic with fixed horizon; PI runs without termination)

Dynamics (exact match to gymnasium PendulumEnv.step):
    torque       = clip(torque, -2, 2)
    theta_ddot   = -3*g/(2*l) * sin(theta + π) + 3/(m*l²) * torque
                 = 15 * sin(theta) + 3 * torque        [g=10, m=1, l=1]
    theta_dot   += theta_ddot * dt                      [dt=0.05]
    theta_dot    = clip(theta_dot, -8, 8)
    theta       += theta_dot * dt
    theta        = angle_normalize(theta)               [→ (-π, π]]
    reward       = -(angle_normalize(theta_before)² + 0.1*theta_dot_before² + 0.001*torque²)

Note: Gymnasium computes reward on the state BEFORE the integration step.
      For policy iteration V(s) = R(s,a) + γV(s'), this is the standard
      immediate reward for being in state s and applying action a.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration2D, CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

BINS_PER_DIM = 200

BINS_SPACE = {
    "theta":     np.linspace(-np.pi, np.pi, BINS_PER_DIM, dtype=np.float32),
    "theta_dot": np.linspace(-8.0,   8.0,   BINS_PER_DIM, dtype=np.float32),
}

# Discretized torque values (continuous action space [-2, 2])
ACTION_SPACE = np.linspace(-2.0, 2.0, 21, dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class PendulumCuda(CudaPolicyIteration2D):
    """
    CudaPolicyIteration2D for Pendulum-v1.

    State convention: theta=0 is upright (goal). Matches gymnasium PendulumEnv
    where the observation is [cos(theta), sin(theta), theta_dot] but the
    internal state is (theta, theta_dot).
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        #define PEND_G      10.0f
        #define PEND_M       1.0f
        #define PEND_L       1.0f
        #define PEND_DT      0.05f
        #define PEND_MAX_SPD 8.0f
        #define PEND_MAX_TRQ 2.0f
        #define PEND_PI      3.14159265358979323846f

        __device__ float pend_angle_normalize(float x) {
            // Maps angle to (-pi, pi].
            // fmodf has same sign as dividend in CUDA, so we add 2*pi if negative.
            float r = fmodf(x + PEND_PI, 2.0f * PEND_PI);
            if (r < 0.0f) r += 2.0f * PEND_PI;
            return r - PEND_PI;
        }

        __device__ void step_dynamics(
            float theta, float theta_dot, float torque,
            float* next_theta, float* next_theta_dot,
            float* reward, bool* terminated
        ) {
            torque = fmaxf(-PEND_MAX_TRQ, fminf(PEND_MAX_TRQ, torque));

            // Reward on current state (matches gymnasium PendulumEnv.step exactly)
            float th_norm = pend_angle_normalize(theta);
            *reward = -(th_norm * th_norm
                        + 0.1f * theta_dot * theta_dot
                        + 0.001f * torque * torque);

            // Euler integration (gymnasium uses single Euler step, not RK4)
            // theta_ddot = -3g/(2l) * sin(theta + pi) + 3/(m*l^2) * torque
            //            =  3g/(2l) * sin(theta)       + 3/(m*l^2) * torque  [since sin(x+pi)=-sin(x)]
            float theta_ddot = (3.0f * PEND_G / (2.0f * PEND_L)) * sinf(theta)
                               + (3.0f / (PEND_M * PEND_L * PEND_L)) * torque;

            float new_theta_dot = theta_dot + theta_ddot * PEND_DT;
            new_theta_dot = fmaxf(-PEND_MAX_SPD, fminf(PEND_MAX_SPD, new_theta_dot));

            float new_theta = pend_angle_normalize(theta + new_theta_dot * PEND_DT);

            *next_theta     = new_theta;
            *next_theta_dot = new_theta_dot;
            *terminated     = false;  // Pendulum has no terminal state
        }
        '''

    # _terminal_fn is not overridden → default: no terminal states


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/pendulum_cuda_policy.npz"),
) -> PendulumCuda:
    config = CudaPIConfig(
        gamma        = 0.99,
        theta        = 1e-4,
        max_eval_iter= 5_000,
        max_pi_iter  = 50,
        log_interval = 200,
    )

    pi = PendulumCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pi: PendulumCuda,
    n_episodes: int = 5,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
    max_steps: int = 1000,
) -> None:
    import gymnasium as gym
    from utils.barycentric import get_optimal_action

    recording   = record_path is not None
    render_mode = "human" if (render and not recording) else ("rgb_array" if recording else None)
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    all_frames  = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(max_steps):
            # Convert observation [cos(theta), sin(theta), theta_dot] → [theta, theta_dot]
            cos_th, sin_th, theta_dot = obs
            theta = float(np.arctan2(sin_th, cos_th))
            state = np.array([theta, theta_dot], dtype=np.float32)

            torque = float(get_optimal_action(
                state,
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))

            obs, reward, terminated, truncated, _ = env.step(
                np.array([torque], dtype=np.float32)
            )
            total_reward += reward
            if recording:
                all_frames.append(env.render())
            if terminated or truncated:
                break

        print(f"Episode {ep + 1}: {step + 1} steps | total reward = {total_reward:.2f}")

    env.close()

    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.suffix.lower() == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=50, loop=0)
        else:
            imageio.mimsave(str(record_path), all_frames, fps=50, macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


def evaluate_random(n_episodes: int = 5, seed: int = 42, max_steps: int = 1000) -> None:
    """Run episodes with a uniformly random policy — use as baseline comparison."""
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for step in range(max_steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        print(f"[random] Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.2f}")
    env.close()


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_function(
    pi: PendulumCuda,
    save_path: Path = Path("results/pendulum_cuda_vf.png"),
) -> None:
    theta_bins     = np.unique(pi.states_space[:, 0])
    theta_dot_bins = np.unique(pi.states_space[:, 1])
    V = pi.value_function.reshape(len(theta_bins), len(theta_dot_bins))

    T, Td = np.meshgrid(theta_bins, theta_dot_bins, indexing="ij")

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, Td, V, cmap="coolwarm", alpha=0.85)
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_zlabel("Value")
    ax.set_title("Pendulum CUDA — Value Function")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value function saved to {save_path.resolve()}")


def plot_policy(
    pi: PendulumCuda,
    save_path: Path = Path("results/pendulum_cuda_policy.png"),
) -> None:
    theta_bins     = np.unique(pi.states_space[:, 0])
    theta_dot_bins = np.unique(pi.states_space[:, 1])
    torque_map     = pi.action_space[pi.policy].reshape(
        len(theta_bins), len(theta_dot_bins)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        np.degrees(theta_bins), theta_dot_bins, torque_map.T,
        cmap="RdBu", vmin=-2, vmax=2,
    )
    plt.colorbar(im, ax=ax, label="Torque (N·m)")
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title("Pendulum CUDA — Optimal Policy")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Policy plot saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pendulum — CUDA Policy Iteration")
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes")
    parser.add_argument("--random",    type=int, nargs="?", const=5, default=None, metavar="N",
                        help="Run N random-policy episodes as baseline (default N=5 if flag given)")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save evaluation video to PATH (.gif or .mp4). "
                             "MP4 requires: pip install imageio[ffmpeg]")
    parser.add_argument("--episodes",  type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--steps",     type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--bins",      type=int, default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM})")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for episode resets (default: 42)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip saving value/policy plots")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path, default=Path("results/pendulum_cuda_policy.npz"))
    args = parser.parse_args()

    if args.random is not None:
        evaluate_random(n_episodes=args.random, seed=args.seed, max_steps=args.steps)
    else:
        if args.bins != BINS_PER_DIM:
            for key in BINS_SPACE:
                lo, hi = BINS_SPACE[key][0], BINS_SPACE[key][-1]
                BINS_SPACE[key] = np.linspace(lo, hi, args.bins, dtype=np.float32)

        if args.save_path.exists() and not args.retrain:
            print(f"[+] Loading existing policy from {args.save_path}")
            pi = PendulumCuda.load(args.save_path)
        else:
            print("[*] Training new policy...")
            pi = train(args.save_path)

        evaluate(pi, n_episodes=args.episodes, render=args.render,
                 record_path=args.record, seed=args.seed, max_steps=args.steps)
        if not args.no_plot:
            plot_value_function(pi)
            plot_policy(pi)

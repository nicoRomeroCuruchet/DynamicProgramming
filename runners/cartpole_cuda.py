"""
runners/cartpole_cuda.py — CUDA Policy Iteration for CartPole-v1.

Embeds CartPole dynamics directly in a CUDA kernel. State space is the
full 4D continuous space (x, x_dot, theta, theta_dot).

State  : [x        ∈ [-2.5, 2.5]   cart position (terminates at |x| > 2.4)
           x_dot   ∈ [-5.0, 5.0]   cart velocity
           theta   ∈ [-0.25, 0.25] pole angle (terminates at |theta| > 0.2094 rad)
           theta_dot ∈ [-5.0, 5.0] pole angular velocity]
Actions: 2 force values {-10.0, +10.0}  →  gym actions {0, 1}
Reward : +1.0 for every step (including the termination step)
Terminal: |x| > 2.4 OR |theta| > 12 deg (0.20943951 rad)

Dynamics (Euler, exact match to gymnasium CartPoleEnv.step):
    force = +10.0 if action==1 else -10.0
    temp       = (force + polemass_length * theta_dot^2 * sin(theta)) / total_mass
    thetaacc   = (g * sin(theta) - cos(theta) * temp) /
                 (length * (4/3 - masspole * cos^2(theta) / total_mass))
    xacc       = temp - polemass_length * thetaacc * cos(theta) / total_mass
    x         += tau * x_dot
    x_dot     += tau * xacc
    theta     += tau * theta_dot
    theta_dot += tau * thetaacc

Constants: g=9.8, masscart=1.0, masspole=0.1, total_mass=1.1,
           length=0.5 (half-pole), polemass_length=0.05, tau=0.02
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration4D, CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

# Theta threshold: 12 degrees in radians
_THETA_THRESH = 12.0 * 2.0 * np.pi / 360.0   # 0.20943951 rad

BINS_PER_DIM = 30

BINS_SPACE = {
    "x":         np.linspace(-2.5,  2.5,  BINS_PER_DIM, dtype=np.float32),
    "x_dot":     np.linspace(-5.0,  5.0,  BINS_PER_DIM, dtype=np.float32),
    "theta":     np.linspace(-0.25, 0.25, BINS_PER_DIM, dtype=np.float32),
    "theta_dot": np.linspace(-5.0,  5.0,  BINS_PER_DIM, dtype=np.float32),
}

# Force values that map directly to gym discrete actions {0, 1}
ACTION_SPACE = np.array([-10.0, 10.0], dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class CartPoleCuda(CudaPolicyIteration4D):
    """
    CudaPolicyIteration4D for CartPole-v1.

    State convention: [x, x_dot, theta, theta_dot]
      theta=0 is upright (goal). Termination at |x|>2.4 or |theta|>12 deg.
    Action space: force in {-10, +10} (maps to gym discrete actions {0, 1}).
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        #define CP_GRAVITY      9.8f
        #define CP_MASSCART     1.0f
        #define CP_MASSPOLE     0.1f
        #define CP_TOTALMASS    1.1f
        #define CP_LENGTH       0.5f
        #define CP_POLEMASSLEN  0.05f
        #define CP_TAU          0.02f
        #define CP_X_THRESH     2.4f
        #define CP_THETA_THRESH 0.20943951f

        __device__ void step_dynamics(
            float x, float x_dot, float theta, float theta_dot, float force,
            float* nx, float* nx_dot, float* ntheta, float* ntheta_dot,
            float* reward, bool* terminated
        ) {
            float costheta = cosf(theta);
            float sintheta = sinf(theta);

            // Equations of motion (Florian 2007 / Barto, Sutton, Anderson 1983)
            float temp = (force + CP_POLEMASSLEN * theta_dot * theta_dot * sintheta)
                         / CP_TOTALMASS;
            float thetaacc = (CP_GRAVITY * sintheta - costheta * temp)
                             / (CP_LENGTH * (4.0f / 3.0f
                                - CP_MASSPOLE * costheta * costheta / CP_TOTALMASS));
            float xacc = temp - CP_POLEMASSLEN * thetaacc * costheta / CP_TOTALMASS;

            // Euler integration (matches gymnasium CartPoleEnv default)
            float new_x         = x         + CP_TAU * x_dot;
            float new_x_dot     = x_dot     + CP_TAU * xacc;
            float new_theta     = theta     + CP_TAU * theta_dot;
            float new_theta_dot = theta_dot + CP_TAU * thetaacc;

            *nx         = new_x;
            *nx_dot     = new_x_dot;
            *ntheta     = new_theta;
            *ntheta_dot = new_theta_dot;

            // Reward: +1 for every step taken (including the one that terminates)
            *reward = 1.0f;

            *terminated = (new_x < -CP_X_THRESH)     || (new_x > CP_X_THRESH)
                       || (new_theta < -CP_THETA_THRESH) || (new_theta > CP_THETA_THRESH);
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        """Mark grid cells outside the failure region as absorbing terminals."""
        x     = states[:, 0]
        theta = states[:, 2]
        mask = (
            (x < -2.4) | (x > 2.4) |
            (theta < -_THETA_THRESH) | (theta > _THETA_THRESH)
        )
        return mask, 0.0   # terminal value = 0 (no future reward)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/cartpole_cuda_policy.npz"),
) -> CartPoleCuda:
    config = CudaPIConfig(
        gamma         = 0.99,
        theta         = 1e-4,
        max_eval_iter = 10_000,
        max_pi_iter   = 100,
        log_interval  = 500,
    )

    pi = CartPoleCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pi: CartPoleCuda,
    n_episodes: int = 5,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
) -> None:
    import gymnasium as gym
    from utils.barycentric import get_optimal_action

    recording   = record_path is not None
    render_mode = "human" if (render and not recording) else ("rgb_array" if recording else None)
    env = gym.make("CartPole-v1", render_mode=render_mode)
    all_frames  = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(500):
            # obs = [x, x_dot, theta, theta_dot]
            state = np.array(obs, dtype=np.float32)

            force = float(get_optimal_action(
                state,
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))
            # Map continuous force to gym discrete action: force < 0 → push left (0)
            gym_action = 1 if force >= 0.0 else 0

            obs, reward, terminated, truncated, _ = env.step(gym_action)
            total_reward += reward
            if recording:
                all_frames.append(env.render())
            if terminated or truncated:
                break

        outcome = "SURVIVED" if not terminated else "FELL"
        print(
            f"Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.0f} | {outcome}"
        )

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


def evaluate_random(n_episodes: int = 5, seed: int = 42) -> None:
    """Run episodes with a uniformly random policy — use as baseline comparison."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for step in range(500):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        print(f"[random] Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.0f}")
    env.close()


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_slice(
    pi: CartPoleCuda,
    save_path: Path = Path("results/cartpole_cuda_vf_slice.png"),
) -> None:
    """Plot V(theta, theta_dot) slice at x=0, x_dot=0."""
    x_bins         = np.unique(pi.states_space[:, 0])
    x_dot_bins     = np.unique(pi.states_space[:, 1])
    theta_bins     = np.unique(pi.states_space[:, 2])
    theta_dot_bins = np.unique(pi.states_space[:, 3])

    n0, n1, n2, n3 = len(x_bins), len(x_dot_bins), len(theta_bins), len(theta_dot_bins)
    V4 = pi.value_function.reshape(n0, n1, n2, n3)

    # Fix x closest to 0 and x_dot closest to 0
    ix   = int(np.argmin(np.abs(x_bins)))
    ixd  = int(np.argmin(np.abs(x_dot_bins)))
    V2 = V4[ix, ixd, :, :]   # shape (n_theta, n_theta_dot)

    T, Td = np.meshgrid(np.degrees(theta_bins), theta_dot_bins, indexing="ij")

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, Td, V2, cmap="viridis", alpha=0.85)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_zlabel("Value")
    ax.set_title("CartPole CUDA — V(theta, theta_dot) at x=0, x_dot=0")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value slice saved to {save_path.resolve()}")


def plot_policy_slice(
    pi: CartPoleCuda,
    save_path: Path = Path("results/cartpole_cuda_policy_slice.png"),
) -> None:
    """Plot optimal policy (force direction) as function of (theta, theta_dot) at x=0, x_dot=0."""
    x_bins         = np.unique(pi.states_space[:, 0])
    x_dot_bins     = np.unique(pi.states_space[:, 1])
    theta_bins     = np.unique(pi.states_space[:, 2])
    theta_dot_bins = np.unique(pi.states_space[:, 3])

    n0, n1, n2, n3 = len(x_bins), len(x_dot_bins), len(theta_bins), len(theta_dot_bins)
    P4 = pi.action_space[pi.policy].reshape(n0, n1, n2, n3)

    ix  = int(np.argmin(np.abs(x_bins)))
    ixd = int(np.argmin(np.abs(x_dot_bins)))
    P2  = P4[ix, ixd, :, :]   # shape (n_theta, n_theta_dot)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        np.degrees(theta_bins), theta_dot_bins, P2.T,
        cmap="RdBu", vmin=-10, vmax=10,
    )
    plt.colorbar(im, ax=ax, label="Force (N): -10=Left, +10=Right")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_title("CartPole CUDA — Policy at x=0, x_dot=0")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Policy slice saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CartPole — CUDA Policy Iteration")
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes")
    parser.add_argument("--random",    type=int, nargs="?", const=5, default=None, metavar="N",
                        help="Run N random-policy episodes as baseline (default N=5 if flag given)")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save evaluation video to PATH (.gif or .mp4). "
                             "MP4 requires: pip install imageio[ffmpeg]")
    parser.add_argument("--episodes",  type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--bins",      type=int, default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM})")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for episode resets (default: 42)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip saving value/policy plots")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path, default=Path("results/cartpole_cuda_policy.npz"))
    args = parser.parse_args()

    if args.random is not None:
        evaluate_random(n_episodes=args.random, seed=args.seed)
    else:
        if args.bins != BINS_PER_DIM:
            for key in BINS_SPACE:
                lo, hi = BINS_SPACE[key][0], BINS_SPACE[key][-1]
                BINS_SPACE[key] = np.linspace(lo, hi, args.bins, dtype=np.float32)

        if args.save_path.exists() and not args.retrain:
            print(f"[+] Loading existing policy from {args.save_path}")
            pi = CartPoleCuda.load(args.save_path)
        else:
            print("[*] Training new policy...")
            pi = train(args.save_path)

        evaluate(pi, n_episodes=args.episodes, render=args.render,
                 record_path=args.record, seed=args.seed)
        if not args.no_plot:
            plot_value_slice(pi)
            plot_policy_slice(pi)

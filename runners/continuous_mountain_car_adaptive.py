"""
runners/continuous_mountain_car_adaptive.py — Hessian-based Adaptive Mesh
for MountainCarContinuous-v0.

Runs the full adaptive policy iteration loop:
  1. Coarse uniform solve
  2. Estimate ‖∇²V‖ on GPU
  3. Equidistribute nodes — more density where value function curves more
  4. Transfer V, re-solve on adaptive grid
  5. Repeat for n_refine cycles

State  : [position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]]
Actions: 21 values in [-1.0, 1.0]  (discretized continuous force)
Reward : -0.1 * force² + 100 * terminated
Terminal: position >= 0.45 and velocity >= 0
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive_cuda_pi import AdaptiveCudaPI2D, AdaptivePIConfig
from src.cuda_policy_iteration import CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

# Discretized force values (continuous action space [-1, 1])
ACTION_SPACE = np.linspace(-1.0, 1.0, 21, dtype=np.float32)


# ── Adaptive Continuous Mountain Car subclass ─────────────────────────────────

class AdaptiveContinuousMountainCar(AdaptiveCudaPI2D):
    """
    AdaptiveCudaPI2D for MountainCarContinuous-v0.

    Dynamics (exact match to gymnasium Continuous_MountainCarEnv.step):
        force    = clip(force, -1.0, 1.0)
        velocity += force * 0.0015 - 0.0025 * cos(3 * position)
        velocity  = clip(velocity, -0.07, 0.07)
        position += velocity
        position  = clip(position, -1.2, 0.6)
        if position <= -1.2: velocity = 0   (left-wall bounce)
        terminated = (position >= 0.45) and (velocity >= 0)
        reward = -0.1 * force^2 + 100 * terminated
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
        return mask, 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/continuous_mountain_car_adaptive_policy.npz"),
    coarse_bins: int = 100,
    refined_bins: int = 300,
    n_refine: int = 2,
    epsilon: float = 0.01,
    nu: float = 0.1,
) -> AdaptiveContinuousMountainCar:

    bins_space = {
        "position": np.linspace(-1.2,  0.6,  coarse_bins, dtype=np.float32),
        "velocity": np.linspace(-0.07, 0.07, coarse_bins, dtype=np.float32),
    }

    pi_cfg = CudaPIConfig(
        gamma=0.99, theta=1e-4, max_eval_iter=5_000, max_pi_iter=50, log_interval=200,
    )
    adaptive_cfg = AdaptivePIConfig(
        epsilon=epsilon, nu=nu, n_refine=n_refine, n_nodes_refined=refined_bins,
    )

    pi = AdaptiveContinuousMountainCar(bins_space, ACTION_SPACE, pi_cfg, adaptive_cfg)
    pi.run_adaptive()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def _make_interpolator(pi: AdaptiveContinuousMountainCar) -> RegularGridInterpolator:
    action_values = pi.action_space[pi.policy]
    policy_2d = action_values.reshape(pi.grid_shape)
    return RegularGridInterpolator(
        (pi.coords_per_dim[0], pi.coords_per_dim[1]),
        policy_2d,
        method='nearest',
        bounds_error=False,
        fill_value=None,
    )


def evaluate(
    pi: AdaptiveContinuousMountainCar,
    n_episodes: int = 5,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
) -> None:
    import gymnasium as gym

    interp = _make_interpolator(pi)

    recording   = record_path is not None
    render_mode = "human" if (render and not recording) else ("rgb_array" if recording else None)
    env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
    all_frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(999):
            action_val = float(interp([[obs[0], obs[1]]])[0])
            obs, reward, terminated, truncated, _ = env.step(
                np.array([action_val], dtype=np.float32)
            )
            total_reward += reward
            if recording:
                all_frames.append(env.render())
            if terminated or truncated:
                break

        outcome = "SUCCESS" if terminated else "timeout"
        print(
            f"Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.2f} | {outcome}"
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


# ── Visualization ─────────────────────────────────────────────────────────────

def _compute_hessian(pi: AdaptiveContinuousMountainCar) -> np.ndarray:
    """Vectorized Frobenius Hessian norm on the adaptive grid."""
    c0 = pi.coords_per_dim[0]
    c1 = pi.coords_per_dim[1]
    V  = pi.value_function.reshape(pi.grid_shape)
    n0, n1 = pi.grid_shape

    hl0 = np.diff(c0)[:-1, np.newaxis]
    hr0 = np.diff(c0)[1:,  np.newaxis]
    hl1 = np.diff(c1)[np.newaxis, :-1]
    hr1 = np.diff(c1)[np.newaxis, 1:]

    Vi = V[1:-1, 1:-1]
    d2_dx2  = 2*(V[2:,  1:-1]/hr0 - Vi*(hl0+hr0)/(hl0*hr0) + V[:-2, 1:-1]/hl0) / (hl0+hr0)
    d2_dy2  = 2*(V[1:-1, 2:] /hr1 - Vi*(hl1+hr1)/(hl1*hr1) + V[1:-1, :-2]/hl1) / (hl1+hr1)
    dx      = 0.5*(hl0+hr0)
    dy      = 0.5*(hl1+hr1)
    d2_dxdy = (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4*dx*dy)

    H = np.zeros((n0, n1))
    H[1:-1, 1:-1] = np.sqrt(d2_dx2**2 + 2*d2_dxdy**2 + d2_dy2**2)
    return H


def plot_adaptive_results(
    pi: AdaptiveContinuousMountainCar,
    save_path: Path = Path("results/continuous_mountain_car_adaptive.png"),
) -> None:
    """
    4-panel figure:
      A) Value function surface (3D)
      B) Hessian norm heatmap (‖∇²V‖)
      C) Node spacing in position dimension
      D) Node spacing in velocity dimension
    """
    c0 = pi.coords_per_dim[0]
    c1 = pi.coords_per_dim[1]
    V  = pi.value_function.reshape(pi.grid_shape)
    H  = _compute_hessian(pi)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # ── A) Value function surface ──
    ax_vf = fig.add_subplot(gs[0, 0], projection='3d')
    P, Vl = np.meshgrid(c0, c1, indexing='ij')
    ax_vf.plot_surface(P, Vl, V, cmap='plasma', alpha=0.85)
    ax_vf.set_xlabel("Position"); ax_vf.set_ylabel("Velocity"); ax_vf.set_zlabel("V")
    ax_vf.set_title("A) Value Function (adaptive grid)")

    # ── B) Hessian norm ──
    ax_hess = fig.add_subplot(gs[0, 1])
    im = ax_hess.pcolormesh(c0, c1, H.T, cmap='hot', shading='auto')
    fig.colorbar(im, ax=ax_hess, label='||d2V||')
    ax_hess.set_xlabel("Position"); ax_hess.set_ylabel("Velocity")
    ax_hess.set_title("B) Hessian Norm — drives mesh refinement")

    # ── C) Position node spacing ──
    ax_c0 = fig.add_subplot(gs[1, 0])
    spacing0 = np.diff(c0)
    ax_c0.bar(c0[:-1], spacing0, width=spacing0, align='edge', color='steelblue', alpha=0.8)
    ax_c0.set_xlabel("Position"); ax_c0.set_ylabel("Node spacing (Δpos)")
    ax_c0.set_title(f"C) Position nodes — {len(c0)} total\n(small spacing = high curvature)")

    # ── D) Velocity node spacing ──
    ax_c1 = fig.add_subplot(gs[1, 1])
    spacing1 = np.diff(c1)
    ax_c1.bar(c1[:-1], spacing1, width=spacing1, align='edge', color='coral', alpha=0.8)
    ax_c1.set_xlabel("Velocity"); ax_c1.set_ylabel("Node spacing (Δvel)")
    ax_c1.set_title(f"D) Velocity nodes — {len(c1)} total\n(small spacing = high curvature)")

    plt.suptitle("Continuous Mountain Car — Hessian-based Adaptive Mesh",
                 fontsize=13, fontweight='bold')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Adaptive results saved to {save_path.resolve()}")


def plot_value_heatmap(
    pi: AdaptiveContinuousMountainCar,
    save_path: Path = Path("results/continuous_mountain_car_adaptive_vf_heatmap.png"),
) -> None:
    """2D heatmap of V(position, velocity)."""
    c0 = pi.coords_per_dim[0]
    c1 = pi.coords_per_dim[1]
    V  = pi.value_function.reshape(pi.grid_shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(c0, c1, V.T, cmap='plasma', shading='auto')
    fig.colorbar(im, ax=ax, label='V (value function)')
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(
        f"Continuous Mountain Car — Value Function Heatmap\n"
        f"Adaptive grid {pi.grid_shape[0]}x{pi.grid_shape[1]}  ({pi.n_states:,} states)"
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value heatmap saved to {save_path.resolve()}")


def plot_policy_heatmap(
    pi: AdaptiveContinuousMountainCar,
    save_path: Path = Path("results/continuous_mountain_car_adaptive_policy_heatmap.png"),
) -> None:
    """2D heatmap of the optimal force policy pi(position, velocity)."""
    c0 = pi.coords_per_dim[0]
    c1 = pi.coords_per_dim[1]
    action_values = pi.action_space[pi.policy].reshape(pi.grid_shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(c0, c1, action_values.T, cmap='RdBu', shading='auto',
                       vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label='Optimal force')
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(
        f"Continuous Mountain Car — Optimal Policy Heatmap\n"
        f"Adaptive grid {pi.grid_shape[0]}x{pi.grid_shape[1]}  ({pi.n_states:,} states)"
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Policy heatmap saved to {save_path.resolve()}")


def plot_node_density(
    pi: AdaptiveContinuousMountainCar,
    save_path: Path = Path("results/continuous_mountain_car_adaptive_nodes.png"),
) -> None:
    """Scatter plot of adaptive grid nodes — shows density variation."""
    g0, g1 = np.meshgrid(pi.coords_per_dim[0], pi.coords_per_dim[1], indexing='ij')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(g0.ravel(), g1.ravel(), s=0.5, alpha=0.4, color='navy')
    ax.set_xlabel("Position"); ax.set_ylabel("Velocity")
    ax.set_title(
        f"Adaptive Grid — {pi.n_states:,} nodes  "
        f"({pi.grid_shape[0]}x{pi.grid_shape[1]})\n"
        "Denser regions correspond to higher ||d2V||"
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Node density plot saved to {save_path.resolve()}")


def plot_value_function_interactive(pi: AdaptiveContinuousMountainCar) -> None:
    """Interactive 3D value function — rotate with mouse, close window to exit."""
    for backend in ["TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "wxAgg"]:
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue

    c0 = pi.coords_per_dim[0]
    c1 = pi.coords_per_dim[1]
    V  = pi.value_function.reshape(pi.grid_shape)
    P, Vl = np.meshgrid(c0, c1, indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, Vl, V, cmap='plasma', alpha=0.9)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("V")
    ax.set_title(
        f"Continuous Mountain Car — Value Function\n"
        f"Adaptive grid {pi.grid_shape[0]}x{pi.grid_shape[1]} | Click+drag to rotate"
    )
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Continuous Mountain Car — Hessian-based Adaptive Mesh Policy Iteration"
    )
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--record",       type=Path, default=None, metavar="PATH")
    parser.add_argument("--episodes",     type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-plot",      action="store_true")
    parser.add_argument("--interactive",  action="store_true",
                        help="Show interactive 3D value function (rotate with mouse)")
    parser.add_argument("--plot-only",    action="store_true",
                        help="Skip training and evaluation, only generate plots")
    parser.add_argument("--retrain",      action="store_true")
    parser.add_argument("--save-path",    type=Path,
                        default=Path("results/continuous_mountain_car_adaptive_policy.npz"))
    parser.add_argument("--coarse-bins",  type=int,   default=100,
                        help="Bins per dim for initial uniform solve (default: 100)")
    parser.add_argument("--refined-bins", type=int,   default=300,
                        help="Nodes per dim after each refinement (default: 300)")
    parser.add_argument("--n-refine",     type=int,   default=2,
                        help="Number of mesh refinement cycles (default: 2)")
    parser.add_argument("--epsilon",      type=float, default=0.01,
                        help="Interpolation error tolerance epsilon (default: 0.01)")
    parser.add_argument("--nu",           type=float, default=0.1,
                        help="Regularization nu (default: 0.1)")
    args = parser.parse_args()

    if args.save_path.exists() and not args.retrain:
        print(f"[+] Loading existing policy from {args.save_path}")
        pi = AdaptiveContinuousMountainCar.load(args.save_path)
    else:
        print("[*] Training adaptive Continuous Mountain Car policy...")
        pi = train(
            save_path    = args.save_path,
            coarse_bins  = args.coarse_bins,
            refined_bins = args.refined_bins,
            n_refine     = args.n_refine,
            epsilon      = args.epsilon,
            nu           = args.nu,
        )

    if not args.plot_only:
        evaluate(pi, n_episodes=args.episodes, render=args.render,
                 record_path=args.record, seed=args.seed)

    if args.interactive:
        plot_value_function_interactive(pi)
    elif not args.no_plot:
        plot_adaptive_results(pi)
        plot_value_heatmap(pi)
        plot_policy_heatmap(pi)
        plot_node_density(pi)

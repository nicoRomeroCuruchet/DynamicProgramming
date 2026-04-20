"""
runners/mountain_car_adaptive.py — Hessian-based Adaptive Mesh for MountainCar-v0.

Runs the full adaptive policy iteration loop:
  1. Coarse uniform solve (100×100 by default)
  2. Estimate ‖∇²V‖ on GPU
  3. Equidistribute nodes — more density where value function curves more
  4. Transfer V, re-solve on adaptive grid
  5. Repeat for n_refine cycles

State  : [position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]]
Actions: {-1.0, 0.0, 1.0}
Reward : -1 per step
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

COARSE_BINS = 100   # bins per dim for the initial uniform solve

BINS_SPACE = {
    "position": np.linspace(-1.2,  0.6,  COARSE_BINS, dtype=np.float32),
    "velocity": np.linspace(-0.07, 0.07, COARSE_BINS, dtype=np.float32),
}

ACTION_SPACE = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


# ── Adaptive Mountain Car subclass ────────────────────────────────────────────

class AdaptiveMountainCar(AdaptiveCudaPI2D):
    """
    AdaptiveCudaPI2D for MountainCar-v0.
    Dynamics identical to MountainCarCuda — only the mesh is adaptive.
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
        return (pos >= 0.5) & (vel >= 0.0), 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/mountain_car_adaptive_policy.npz"),
    coarse_bins: int = COARSE_BINS,
    refined_bins: int = 300,
    n_refine: int = 2,
    epsilon: float = 0.01,
    nu: float = 0.1,
) -> AdaptiveMountainCar:

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

    pi = AdaptiveMountainCar(bins_space, ACTION_SPACE, pi_cfg, adaptive_cfg)
    pi.run_adaptive()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def _make_interpolator(pi: AdaptiveMountainCar) -> RegularGridInterpolator:
    """
    Build a scipy interpolator over the non-uniform adaptive grid.
    Used during episode rollout instead of the uniform barycentric kernel.
    """
    # policy stores action indices; we need the action values per state
    action_values = pi.action_space[pi.policy]
    policy_2d = action_values.reshape(pi.grid_shape)

    return RegularGridInterpolator(
        (pi.coords_per_dim[0], pi.coords_per_dim[1]),
        policy_2d,
        method='nearest',   # policy is discrete; nearest avoids blending actions
        bounds_error=False,
        fill_value=None,
    )


def evaluate(
    pi: AdaptiveMountainCar,
    n_episodes: int = 5,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
) -> None:
    import gymnasium as gym

    interp = _make_interpolator(pi)

    recording   = record_path is not None
    render_mode = "human" if (render and not recording) else ("rgb_array" if recording else None)
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    all_frames = []

    def action_to_gym(a: float) -> int:
        return int(round(float(a) + 1.0))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(200):
            action_val = float(interp([[obs[0], obs[1]]])[0])
            obs, reward, terminated, truncated, _ = env.step(action_to_gym(action_val))
            total_reward += reward
            if recording:
                all_frames.append(env.render())
            if terminated or truncated:
                break

        print(f"Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.1f}")

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

def plot_adaptive_results(
    pi: AdaptiveMountainCar,
    save_path: Path = Path("results/mountain_car_adaptive.png"),
) -> None:
    """
    4-panel figure:
      A) Value function on the adaptive grid
      B) Hessian norm heatmap (‖∇²V‖)
      C) Node spacing in position dimension
      D) Node spacing in velocity dimension
    """
    c0 = pi.coords_per_dim[0]   # position nodes
    c1 = pi.coords_per_dim[1]   # velocity nodes
    V  = pi.value_function.reshape(pi.grid_shape)

    # Hessian norm via vectorized non-uniform finite differences (no Python loops)
    n0, n1 = pi.grid_shape
    hl0 = np.diff(c0)[:-1, np.newaxis]   # (n0-2, 1)  left spacing in dim 0
    hr0 = np.diff(c0)[1:,  np.newaxis]   # (n0-2, 1)  right spacing in dim 0
    hl1 = np.diff(c1)[np.newaxis, :-1]   # (1, n1-2)  left spacing in dim 1
    hr1 = np.diff(c1)[np.newaxis, 1:]    # (1, n1-2)  right spacing in dim 1

    Vi = V[1:-1, 1:-1]                   # interior block (n0-2, n1-2)

    d2_dx2 = 2*(V[2:,  1:-1]/hr0 - Vi*(hl0+hr0)/(hl0*hr0) + V[:-2, 1:-1]/hl0) / (hl0+hr0)
    d2_dy2 = 2*(V[1:-1, 2:] /hr1 - Vi*(hl1+hr1)/(hl1*hr1) + V[1:-1, :-2]/hl1) / (hl1+hr1)
    dx     = 0.5*(hl0+hr0)
    dy     = 0.5*(hl1+hr1)
    d2_dxdy = (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4*dx*dy)

    H = np.zeros((n0, n1))
    H[1:-1, 1:-1] = np.sqrt(d2_dx2**2 + 2*d2_dxdy**2 + d2_dy2**2)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # ── A) Value function ──
    ax_vf = fig.add_subplot(gs[0, 0], projection='3d')
    P, Vl = np.meshgrid(c0, c1, indexing='ij')
    ax_vf.plot_surface(P, Vl, V, cmap='viridis', alpha=0.85)
    ax_vf.set_xlabel("Position"); ax_vf.set_ylabel("Velocity"); ax_vf.set_zlabel("V")
    ax_vf.set_title("A) Value Function (adaptive grid)")

    # ── B) Hessian norm ──
    ax_hess = fig.add_subplot(gs[0, 1])
    im = ax_hess.pcolormesh(c0, c1, H.T, cmap='hot', shading='auto')
    fig.colorbar(im, ax=ax_hess, label='‖∇²V‖')
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

    plt.suptitle("Mountain Car — Hessian-based Adaptive Mesh", fontsize=13, fontweight='bold')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Adaptive results saved to {save_path.resolve()}")


def plot_value_function_interactive(pi: AdaptiveMountainCar) -> None:
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
    ax.plot_surface(P, Vl, V, cmap='viridis', alpha=0.9)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("V")
    ax.set_title(
        f"Value Function — adaptive grid {pi.grid_shape[0]}x{pi.grid_shape[1]}\n"
        "Click + drag to rotate | scroll to zoom"
    )
    plt.tight_layout()
    plt.show()


def plot_value_heatmap(
    pi: AdaptiveMountainCar,
    save_path: Path = Path("results/mountain_car_adaptive_vf_heatmap.png"),
) -> None:
    """2D heatmap of the value function V(position, velocity)."""
    c0 = pi.coords_per_dim[0]   # position nodes
    c1 = pi.coords_per_dim[1]   # velocity nodes
    V  = pi.value_function.reshape(pi.grid_shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(c0, c1, V.T, cmap='viridis', shading='auto')
    fig.colorbar(im, ax=ax, label='V (value function)')
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(
        f"Mountain Car — Value Function Heatmap\n"
        f"Adaptive grid {pi.grid_shape[0]}×{pi.grid_shape[1]}  ({pi.n_states:,} states)"
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value heatmap saved to {save_path.resolve()}")


def plot_node_density(
    pi: AdaptiveMountainCar,
    save_path: Path = Path("results/mountain_car_adaptive_nodes.png"),
) -> None:
    """Scatter plot of adaptive grid nodes — visually shows density variation."""
    g0, g1 = np.meshgrid(pi.coords_per_dim[0], pi.coords_per_dim[1], indexing='ij')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(g0.ravel(), g1.ravel(), s=0.5, alpha=0.4, color='navy')
    ax.set_xlabel("Position"); ax.set_ylabel("Velocity")
    ax.set_title(
        f"Adaptive Grid — {pi.n_states:,} nodes  "
        f"({pi.grid_shape[0]}×{pi.grid_shape[1]})\n"
        "Denser regions correspond to higher ‖∇²V‖"
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Node density plot saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mountain Car — Hessian-based Adaptive Mesh Policy Iteration"
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
                        default=Path("results/mountain_car_adaptive_policy.npz"))
    parser.add_argument("--coarse-bins",  type=int,   default=100,
                        help="Bins per dim for initial uniform solve (default: 100)")
    parser.add_argument("--refined-bins", type=int,   default=300,
                        help="Nodes per dim after each refinement (default: 300)")
    parser.add_argument("--n-refine",     type=int,   default=2,
                        help="Number of mesh refinement cycles (default: 2)")
    parser.add_argument("--epsilon",      type=float, default=0.01,
                        help="Interpolation error tolerance ε (default: 0.01)")
    parser.add_argument("--nu",           type=float, default=0.1,
                        help="Regularization ν (default: 0.1)")
    args = parser.parse_args()

    if args.save_path.exists() and not args.retrain:
        print(f"[+] Loading existing policy from {args.save_path}")
        pi = AdaptiveMountainCar.load(args.save_path)
    else:
        print("[*] Training adaptive policy...")
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
        plot_node_density(pi)

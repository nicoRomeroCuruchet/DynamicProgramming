"""
runners/cartpole_adaptive.py — Hessian-based Adaptive Mesh for CartPole-v1.

State  : [x ∈ [-2.5,2.5], x_dot ∈ [-5,5], theta ∈ [-0.25,0.25], theta_dot ∈ [-5,5]]
Actions: {-10.0, +10.0} (force)
Reward : +1 per step
Terminal: |x| > 2.4 OR |theta| > 12 deg

Memory guide (float32, n bins/dim → n^4 states):
  n=20 →  160k states ~  5 MB VRAM   (coarse)
  n=30 →  810k states ~ 25 MB VRAM
  n=40 → 2.56M states ~ 80 MB VRAM   (refined, safe default)
  n=50 → 6.25M states ~200 MB VRAM
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive_cuda_pi import AdaptiveCudaPI4D, AdaptivePIConfig
from src.cuda_policy_iteration import CudaPIConfig

_THETA_THRESH = 12.0 * 2.0 * np.pi / 360.0   # 0.20943951 rad

ACTION_SPACE = np.array([-10.0, 10.0], dtype=np.float32)


# ── Adaptive CartPole subclass ────────────────────────────────────────────────

class AdaptiveCartPole(AdaptiveCudaPI4D):
    """AdaptiveCudaPI4D for CartPole-v1. Dynamics identical to CartPoleCuda."""

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
            float temp = (force + CP_POLEMASSLEN * theta_dot * theta_dot * sintheta)
                         / CP_TOTALMASS;
            float thetaacc = (CP_GRAVITY * sintheta - costheta * temp)
                             / (CP_LENGTH * (4.0f/3.0f
                                - CP_MASSPOLE * costheta * costheta / CP_TOTALMASS));
            float xacc = temp - CP_POLEMASSLEN * thetaacc * costheta / CP_TOTALMASS;

            *nx         = x         + CP_TAU * x_dot;
            *nx_dot     = x_dot     + CP_TAU * xacc;
            *ntheta     = theta     + CP_TAU * theta_dot;
            *ntheta_dot = theta_dot + CP_TAU * thetaacc;
            *reward     = 1.0f;
            *terminated = (*nx < -CP_X_THRESH) || (*nx > CP_X_THRESH)
                       || (*ntheta < -CP_THETA_THRESH) || (*ntheta > CP_THETA_THRESH);
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        x     = states[:, 0]
        theta = states[:, 2]
        mask  = (x < -2.4) | (x > 2.4) | (theta < -_THETA_THRESH) | (theta > _THETA_THRESH)
        return mask, 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/cartpole_adaptive_policy.npz"),
    coarse_bins: int = 20,
    refined_bins: int = 35,
    n_refine: int = 2,
    epsilon: float = 0.01,
    nu: float = 0.1,
) -> AdaptiveCartPole:

    bins_space = {
        "x":         np.linspace(-2.5,  2.5,  coarse_bins, dtype=np.float32),
        "x_dot":     np.linspace(-5.0,  5.0,  coarse_bins, dtype=np.float32),
        "theta":     np.linspace(-0.25, 0.25, coarse_bins, dtype=np.float32),
        "theta_dot": np.linspace(-5.0,  5.0,  coarse_bins, dtype=np.float32),
    }

    pi_cfg = CudaPIConfig(
        gamma=0.99, theta=1e-4, max_eval_iter=10_000, max_pi_iter=100, log_interval=500,
    )
    adaptive_cfg = AdaptivePIConfig(
        epsilon=epsilon, nu=nu, n_refine=n_refine, n_nodes_refined=refined_bins,
    )

    pi = AdaptiveCartPole(bins_space, ACTION_SPACE, pi_cfg, adaptive_cfg)
    pi.run_adaptive()
    pi.save(save_path)
    return pi


# ── Evaluation ────────────────────────────────────────────────────────────────

def _make_policy_interpolator(pi: AdaptiveCartPole) -> RegularGridInterpolator:
    action_values = pi.action_space[pi.policy]
    policy_4d = action_values.reshape(pi.grid_shape)
    return RegularGridInterpolator(
        tuple(pi.coords_per_dim),
        policy_4d,
        method='nearest',
        bounds_error=False,
        fill_value=None,
    )


def evaluate(
    pi: AdaptiveCartPole,
    n_episodes: int = 5,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
) -> None:
    import gymnasium as gym

    interp = _make_policy_interpolator(pi)

    recording   = record_path is not None
    render_mode = "human" if (render and not recording) else ("rgb_array" if recording else None)
    env = gym.make("CartPole-v1", render_mode=render_mode)
    all_frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(500):
            force = float(interp([obs])[0])
            gym_action = 1 if force >= 0.0 else 0
            obs, reward, terminated, truncated, _ = env.step(gym_action)
            total_reward += reward
            if recording:
                all_frames.append(env.render())
            if terminated or truncated:
                break

        outcome = "SURVIVED" if not terminated else "FELL"
        print(f"Episode {ep+1}: {step+1} steps | reward = {total_reward:.0f} | {outcome}")

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

def _hessian_diag_cpu(pi: AdaptiveCartPole) -> np.ndarray:
    """Vectorized diagonal Hessian norm on CPU from saved V."""
    V = pi.value_function.reshape(pi.grid_shape)
    coords = pi.coords_per_dim
    H = np.zeros_like(V)

    for d in range(4):
        sl_p = [slice(None)]*4; sl_p[d] = slice(2, None)
        sl_c = [slice(None)]*4; sl_c[d] = slice(1, -1)
        sl_m = [slice(None)]*4; sl_m[d] = slice(None, -2)
        # interior slice: only trim dimension d, keep others full
        sl_i = [slice(None)]*4; sl_i[d] = slice(1, -1)

        hl = np.diff(coords[d])[:-1]
        hr = np.diff(coords[d])[1:]

        shape = [1]*4; shape[d] = -1
        hl = hl.reshape(shape); hr = hr.reshape(shape)

        Vp = V[tuple(sl_p)]; Vc = V[tuple(sl_c)]; Vm = V[tuple(sl_m)]
        d2 = 2*(Vp/hr - Vc*(hl+hr)/(hl*hr) + Vm/hl) / (hl+hr)
        H[tuple(sl_i)] += d2**2

    return np.sqrt(H)


def plot_results(
    pi: AdaptiveCartPole,
    save_path: Path = Path("results/cartpole_adaptive.png"),
) -> None:
    """
    6-panel figure:
      A) V(theta, theta_dot) slice at x=0, x_dot=0
      B) Hessian norm projected onto (theta, theta_dot)
      C-F) Node spacing per dimension
    """
    coords = pi.coords_per_dim
    names  = pi.dim_names
    V = pi.value_function.reshape(pi.grid_shape)

    # V slice at x~0, x_dot~0
    ix  = int(np.argmin(np.abs(coords[0])))
    ixd = int(np.argmin(np.abs(coords[1])))
    V2  = V[ix, ixd, :, :]

    # Hessian projected onto (theta, theta_dot) — max over x, x_dot
    H = _hessian_diag_cpu(pi)
    H2 = H.max(axis=(0, 1))

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # A) Value slice
    ax_vf = fig.add_subplot(gs[0, 0], projection='3d')
    T, Td = np.meshgrid(np.degrees(coords[2]), coords[3], indexing='ij')
    ax_vf.plot_surface(T, Td, V2, cmap='viridis', alpha=0.85)
    ax_vf.set_xlabel("theta (deg)"); ax_vf.set_ylabel("theta_dot"); ax_vf.set_zlabel("V")
    ax_vf.set_title("A) V(theta, theta_dot)\nat x=0, x_dot=0")

    # B) Hessian heatmap on (theta, theta_dot)
    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.pcolormesh(np.degrees(coords[2]), coords[3], H2.T, cmap='hot', shading='auto')
    fig.colorbar(im, ax=ax_h, label='||d2V||')
    ax_h.set_xlabel("theta (deg)"); ax_h.set_ylabel("theta_dot")
    ax_h.set_title("B) Hessian norm\nprojected on (theta, theta_dot)")

    # C-F) Node spacing per dimension
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    labels = [names[i] for i in range(4)]
    positions = [gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2]]

    for d in range(4):
        ax = fig.add_subplot(positions[d])
        c  = coords[d]
        sp = np.diff(c)
        ax.bar(c[:-1], sp, width=sp, align='edge', color=colors[d], alpha=0.8)
        ax.set_xlabel(labels[d])
        ax.set_ylabel("spacing")
        ax.set_title(f"{'CDEF'[d]}) {labels[d]} nodes — {len(c)} total")

    plt.suptitle("CartPole — Hessian-based Adaptive Mesh", fontsize=13, fontweight='bold')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results saved to {save_path.resolve()}")


def plot_value_interactive(pi: AdaptiveCartPole) -> None:
    """Interactive V(theta, theta_dot) slice — rotate with mouse."""
    for backend in ["TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"]:
        try:
            plt.switch_backend(backend); break
        except Exception:
            continue

    coords = pi.coords_per_dim
    ix  = int(np.argmin(np.abs(coords[0])))
    ixd = int(np.argmin(np.abs(coords[1])))
    V2  = pi.value_function.reshape(pi.grid_shape)[ix, ixd, :, :]

    T, Td = np.meshgrid(np.degrees(coords[2]), coords[3], indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, Td, V2, cmap='viridis', alpha=0.9)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_zlabel("V")
    ax.set_title(
        f"CartPole — V(theta, theta_dot) at x~0, x_dot~0\n"
        f"Adaptive grid {pi.grid_shape.tolist()} | Click+drag to rotate"
    )
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CartPole — Hessian-based Adaptive Mesh Policy Iteration"
    )
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--record",       type=Path, default=None, metavar="PATH")
    parser.add_argument("--episodes",     type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-plot",      action="store_true")
    parser.add_argument("--plot-only",    action="store_true",
                        help="Skip training and evaluation, only generate plots")
    parser.add_argument("--interactive",  action="store_true",
                        help="Show interactive 3D value slice (rotate with mouse)")
    parser.add_argument("--retrain",      action="store_true")
    parser.add_argument("--save-path",    type=Path,
                        default=Path("results/cartpole_adaptive_policy.npz"))
    parser.add_argument("--coarse-bins",  type=int, default=20,
                        help="Bins/dim for initial uniform solve (default: 20 → 160k states)")
    parser.add_argument("--refined-bins", type=int, default=35,
                        help="Nodes/dim after refinement (default: 35 → 1.5M states)")
    parser.add_argument("--n-refine",     type=int, default=2,
                        help="Refinement cycles (default: 2)")
    parser.add_argument("--epsilon",      type=float, default=0.01)
    parser.add_argument("--nu",           type=float, default=0.1)
    args = parser.parse_args()

    if args.save_path.exists() and not args.retrain:
        print(f"[+] Loading existing policy from {args.save_path}")
        pi = AdaptiveCartPole.load(args.save_path)
    else:
        print("[*] Training adaptive CartPole policy...")
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
        plot_value_interactive(pi)
    elif not args.no_plot:
        plot_results(pi)

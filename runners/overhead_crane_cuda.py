"""
runners/overhead_crane_cuda.py -- CUDA Policy Iteration for Overhead Crane (anti-sway).

A trolley moves horizontally along a fixed rail. A load hangs from the trolley
on a rope of fixed length. Goal: bring the trolley to the centre (x=0) while
damping load swing (theta -> 0).

State  : [x          in [-3.0,  3.0]   trolley position along rail (m)
           x_dot     in [-2.0,  2.0]   trolley velocity (m/s)
           theta     in [-1.15, 1.15]  rope angle from vertical (rad, 0=straight down)
           theta_dot in [-3.0,  3.0]   rope angular velocity (rad/s)]

Goal   : x -> 0, theta -> 0  (centred, no swing)
Actions: 3 force values {-10.0, 0.0, +10.0} N applied horizontally to trolley
Reward : 1.0 - 0.4*(x/X_MAX)^2 - 0.3*(theta/TH_MAX)^2 - 0.3*(x_dot/XD_MAX)^2
         (penalises position, swing AND trolley velocity — forces braking at goal)
Termination: |x| >= 3.0 (trolley hits rail end, V=0)

Dynamics (Lagrangian, 2-DOF underactuated):
  Mass matrix H (2x2, always positive-definite):
    H = [[M+m,            m*L*cos(theta)],
         [m*L*cos(theta), m*L^2         ]]
  det(H) = m*L^2*(M + m*sin^2(theta)) > 0 always

  Generalised forces:
    rhs1 = F + m*L*theta_dot^2*sin(theta)   (centrifugal from swing)
    rhs2 = -m*g*L*sin(theta)                (gravity restoring torque)

  Analytic inverse (2x2):
    x_ddot     = ( d*rhs1 - b*rhs2) / det
    theta_ddot = (-b*rhs1 + a*rhs2) / det
    where a=M+m, b=m*L*cos(theta), d=m*L^2

Constants: g=9.81, M=1.0 kg (trolley), m=0.5 kg (load), L=1.5 m (rope), tau=0.02 s
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration4D, CudaPIConfig


# ---- Grid & action space --------------------------------------------------

_X_MAX    = 3.0                   # rail half-length (m) — terminate here
_TH_MAX   = np.pi / 3.0           # 60 deg, reward normalisation
_TH_BOUND = _TH_MAX * 1.1         # ~66 deg, grid bounds with margin

BINS_PER_DIM = 30   # default; override with --bins at runtime

BINS_SPACE = {
    "x":         np.linspace(-_X_MAX,    _X_MAX,    BINS_PER_DIM, dtype=np.float32),
    "x_dot":     np.linspace(-2.0,       2.0,       BINS_PER_DIM, dtype=np.float32),
    "theta":     np.linspace(-_TH_BOUND, _TH_BOUND, BINS_PER_DIM, dtype=np.float32),
    "theta_dot": np.linspace(-3.0,       3.0,       BINS_PER_DIM, dtype=np.float32),
}

ACTION_SPACE = np.array([-10.0, 0.0, 10.0], dtype=np.float32)


# ---- CUDA subclass --------------------------------------------------------

class OverheadCraneCuda(CudaPolicyIteration4D):
    """
    CudaPolicyIteration4D for overhead crane anti-sway control.

    State : [x, x_dot, theta, theta_dot]
    theta = 0  -> load hanging straight down (stable goal).
    theta > 0  -> load displaced to the right.
    Force > 0  -> push trolley to the right.
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        // --- Physical constants -------------------------------------------
        #define OC_G      9.81f
        #define OC_M      1.0f     // trolley mass (kg)
        #define OC_m      0.5f     // load mass (kg)
        #define OC_L      1.5f     // rope length (m)
        #define OC_TAU    0.02f    // integration timestep (s)
        #define OC_X_MAX    3.0f     // rail half-length: terminate at |x| >= X_MAX
        #define OC_TH_MAX   1.04720f // 60 deg = pi/3, reward normalisation
        #define OC_XD_MAX   2.0f     // max trolley velocity, reward normalisation
        #define OC_W_X      0.4f     // position penalty weight
        #define OC_W_TH     0.3f     // angle penalty weight
        #define OC_W_XD     0.3f     // velocity penalty weight
        #define OC_GOAL_X   0.15f    // goal position tolerance (m)
        #define OC_GOAL_TH  0.05f    // goal angle tolerance (rad)
        #define OC_GOAL_XD  0.10f    // goal velocity tolerance (m/s)

        __device__ void step_dynamics(
            float x, float xd, float theta, float thetad, float force,
            float* nx, float* nxd, float* ntheta, float* nthetad,
            float* reward, bool* terminated
        ) {
            float costh = cosf(theta);
            float sinth = sinf(theta);

            // --- 2x2 mass matrix: H = [[a, b], [b, d]] --------------------
            float a = OC_M + OC_m;
            float b = OC_m * OC_L * costh;
            float d = OC_m * OC_L * OC_L;

            // --- Generalised forces + centrifugal -------------------------
            float rhs1 = force + OC_m * OC_L * thetad * thetad * sinth;
            float rhs2 = -OC_m * OC_G * OC_L * sinth;

            // --- 2x2 analytic inverse: H^{-1} = [[d,-b],[-b,a]] / det ----
            // det = a*d - b^2 = m*L^2*(M + m*sin^2(theta)) > 0 always
            float det     = a * d - b * b;
            float inv_det = 1.0f / det;

            float xacc     = ( d * rhs1 - b * rhs2) * inv_det;
            float thetaacc = (-b * rhs1 + a * rhs2) * inv_det;

            // --- Euler integration ----------------------------------------
            *nx      = x      + OC_TAU * xd;
            *nxd     = xd     + OC_TAU * xacc;
            *ntheta  = theta  + OC_TAU * thetad;
            *nthetad = thetad + OC_TAU * thetaacc;

            // --- Reward: penalise position, angle AND trolley velocity ----
            float xn  = *nx     / OC_X_MAX;
            float thn = *ntheta / OC_TH_MAX;
            float xdn = *nxd    / OC_XD_MAX;
            *reward = 1.0f - OC_W_X * xn * xn - OC_W_TH * thn * thn - OC_W_XD * xdn * xdn;

            // --- Terminate: rail end (failure) OR goal reached (success) --
            bool hit_wall = (*nx <= -OC_X_MAX) || (*nx >= OC_X_MAX);
            bool at_goal  = (fabsf(*nx)   <= OC_GOAL_X)
                         && (fabsf(*ntheta) <= OC_GOAL_TH)
                         && (fabsf(*nxd)    <= OC_GOAL_XD);
            *terminated = hit_wall || at_goal;
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        """
        Two kinds of terminal states, both marked here with value=0.
        Goal states are upgraded to V=1/(1-gamma) in _allocate_tensors_and_compile.
        """
        x   = states[:, 0]
        xd  = states[:, 1]
        th  = states[:, 2]
        fail_mask = (x <= -_X_MAX) | (x >= _X_MAX)
        goal_mask = (
            (np.abs(x)  <= 0.15) &
            (np.abs(th) <= 0.05) &
            (np.abs(xd) <= 0.10)
        )
        self._goal_mask = goal_mask   # stash for use in override below
        return (fail_mask | goal_mask), 0.0

    def _allocate_tensors_and_compile(self) -> None:
        """Extend base allocation to set goal states to the maximum value."""
        import cupy as cp
        super()._allocate_tensors_and_compile()
        if hasattr(self, "_goal_mask") and np.any(self._goal_mask):
            goal_value = float(1.0 / (1.0 - self.config.gamma))
            d_goal = cp.asarray(self._goal_mask, dtype=cp.bool_)
            self.d_value_function[d_goal]     = goal_value
            self.d_new_value_function[d_goal] = goal_value
            from loguru import logger
            logger.info(
                f"Goal states: {self._goal_mask.sum():,} "
                f"(value={goal_value:.1f} = 1/(1-gamma))"
            )


# ---- Python step (for inference, no CUDA needed) -------------------------

def _step_python(state, force):
    """Single Euler step matching CUDA dynamics — used at inference time."""
    x, xd, theta, thetad = state
    M, m, L, g, tau = 1.0, 0.5, 1.5, 9.81, 0.02

    costh = np.cos(theta)
    sinth = np.sin(theta)

    a = M + m
    b = m * L * costh
    d = m * L * L

    rhs1 = force + m * L * thetad**2 * sinth
    rhs2 = -m * g * L * sinth

    det      = a * d - b * b
    xacc     = ( d * rhs1 - b * rhs2) / det
    thetaacc = (-b * rhs1 + a * rhs2) / det

    nx      = x      + tau * xd
    nxd     = xd     + tau * xacc
    ntheta  = theta  + tau * thetad
    nthetad = thetad + tau * thetaacc

    next_state = np.array([nx, nxd, ntheta, nthetad], dtype=np.float32)
    hit_wall  = abs(nx) >= _X_MAX
    at_goal   = abs(nx) <= 0.15 and abs(ntheta) <= 0.05 and abs(nxd) <= 0.10
    terminated = hit_wall or at_goal
    xn  = nx     / _X_MAX
    thn = ntheta / _TH_MAX
    xdn = nxd    / 2.0
    reward = 1.0 - 0.4 * xn**2 - 0.3 * thn**2 - 0.3 * xdn**2
    return next_state, reward, terminated


# ---- Training ------------------------------------------------------------

def train(
    save_path: Path = Path("results/overhead_crane_cuda_policy.npz"),
) -> OverheadCraneCuda:
    config = CudaPIConfig(
        gamma         = 0.995,
        theta         = 1e-4,
        max_eval_iter = 10_000,
        max_pi_iter   = 100,
        log_interval  = 500,
    )
    pi = OverheadCraneCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ---- Rendering (pygame) --------------------------------------------------

_SCREEN_W   = 700
_SCREEN_H   = 420
_RENDER_FPS = 50
_pygame_font = None   # lazy-initialised inside render loop


def _render_frame_pygame(screen, clock, state, target_x: float = 0.0, step: int = 0):
    """
    Draw overhead crane: fixed rail at top, trolley sliding on rail,
    rope hanging at angle theta, load circle at rope end.
    """
    global _pygame_font
    import pygame
    from pygame import gfxdraw

    x, _, theta, _ = state

    # World -> screen mapping
    scale  = 500.0 / (2.0 * _X_MAX)   # px per meter (~83 px/m for X_MAX=3)
    rail_y = 80                         # y-pixel of rail top surface

    cx = int(_SCREEN_W / 2.0 + x * scale)          # trolley centre x
    L_px = int(1.5 * scale)                          # rope pixel length

    # Load position: theta=0 -> straight down (natural in pygame y-down coords)
    load_x = int(cx + L_px * np.sin(theta))
    load_y = rail_y + 24 + L_px   # attach_y + rope

    # Recompute attach point exactly
    attach_y = rail_y + 24          # bottom of trolley box
    rope_end_x = int(cx        + L_px * np.sin(theta))
    rope_end_y = int(attach_y  + L_px * np.cos(theta))

    target_cx = int(_SCREEN_W / 2.0 + target_x * scale)

    surf = pygame.Surface((_SCREEN_W, _SCREEN_H))
    surf.fill((240, 240, 245))

    # --- Ceiling / support structure ---
    pygame.draw.rect(surf, (100, 100, 100), (0, 0, _SCREEN_W, rail_y - 12))

    # --- Rail (I-beam) ---
    rail_left  = 25
    rail_right = _SCREEN_W - 25
    pygame.draw.rect(surf, (70, 70, 70),
                     (rail_left, rail_y - 12, rail_right - rail_left, 12))   # top flange
    pygame.draw.rect(surf, (90, 90, 90),
                     (rail_left, rail_y - 12, rail_right - rail_left, 6))    # highlight

    # Rail end stops
    pygame.draw.rect(surf, (50, 50, 50), (rail_left - 8,  rail_y - 24, 10, 36))
    pygame.draw.rect(surf, (50, 50, 50), (rail_right - 2, rail_y - 24, 10, 36))

    # --- Target marker: green downward triangle on the rail ---
    ty = rail_y
    gfxdraw.filled_trigon(
        surf,
        target_cx - 8, ty,
        target_cx + 8, ty,
        target_cx,     ty + 18,
        (40, 180, 60),
    )
    gfxdraw.aatrigon(
        surf,
        target_cx - 8, ty,
        target_cx + 8, ty,
        target_cx,     ty + 18,
        (20, 140, 40),
    )

    # --- Trolley box (slides on underside of rail) ---
    tw, th_box = 54, 24
    pygame.draw.rect(surf, (80, 95, 120),
                     (cx - tw // 2, rail_y, tw, th_box), border_radius=5)
    pygame.draw.rect(surf, (50, 65, 90),
                     (cx - tw // 2, rail_y, tw, th_box), width=2, border_radius=5)

    # Trolley wheel circles (sit on the rail flange)
    for wx in [cx - 16, cx + 16]:
        gfxdraw.filled_circle(surf, wx, rail_y, 6, (150, 155, 165))
        gfxdraw.aacircle(surf,     wx, rail_y, 6, (90, 95, 105))

    # --- Rope ---
    pygame.draw.line(surf, (100, 80, 50),
                     (cx, attach_y), (rope_end_x, rope_end_y), 3)

    # --- Load (filled circle with outline) ---
    load_r = 18
    gfxdraw.filled_circle(surf, rope_end_x, rope_end_y, load_r, (195, 75, 30))
    gfxdraw.aacircle(surf,      rope_end_x, rope_end_y, load_r, (130, 45, 10))
    # Load highlight
    gfxdraw.filled_circle(surf, rope_end_x - 5, rope_end_y - 5, 5, (230, 130, 80))

    # --- Ground line ---
    pygame.draw.line(surf, (60, 60, 60), (0, _SCREEN_H - 20), (_SCREEN_W, _SCREEN_H - 20), 2)

    # --- Info overlay ---
    if _pygame_font is None:
        _pygame_font = pygame.font.SysFont("monospace", 15)
    font = _pygame_font

    x_goal_dist = abs(state[0])
    th_deg = np.degrees(theta)
    info = [
        f"step : {step:4d}",
        f"x    : {state[0]:+.3f} m  (goal: 0.00)",
        f"theta: {th_deg:+.2f} deg",
        f"dist : {x_goal_dist:.3f} m",
    ]
    for i, line in enumerate(info):
        color = (30, 30, 30) if i < 3 else (100, 100, 100)
        surf.blit(font.render(line, True, color), (10, _SCREEN_H - 90 + i * 20))

    screen.blit(surf, (0, 0))
    pygame.event.pump()
    clock.tick(_RENDER_FPS)
    pygame.display.flip()


# ---- Evaluation ----------------------------------------------------------

def evaluate(
    pi: OverheadCraneCuda,
    n_episodes: int = 5,
    render: bool = False,
    start_x: float = 2.0,
    record_path: Path = None,
) -> None:
    from utils.barycentric import get_optimal_action

    recording = record_path is not None
    do_render  = render or recording   # recording always needs a surface

    rng = np.random.default_rng(seed=42)

    if do_render:
        import pygame
        pygame.init()
        pygame.display.init()
        flags = 0 if render else pygame.NOFRAME
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H), flags)
        pygame.display.set_caption("Overhead Crane -- CUDA Policy Iteration")
        clock = pygame.time.Clock()

    all_frames = []   # collected across all episodes when recording

    for ep in range(n_episodes):
        x0     = float(np.clip(start_x + rng.uniform(-0.1, 0.1), -2.8, 2.8))
        theta0 = float(rng.uniform(-0.05, 0.05))
        state  = np.array([x0, 0.0, theta0, 0.0], dtype=np.float32)
        total_reward = 0.0
        reached_goal = False

        for step in range(1000):
            if do_render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                _render_frame_pygame(screen, clock, state, target_x=0.0, step=step)

                if recording:
                    # pygame surface is (W, H, 3); imageio expects (H, W, 3)
                    frame = pygame.surfarray.array3d(screen)
                    all_frames.append(frame.transpose(1, 0, 2))

            force = float(get_optimal_action(
                state,
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))
            state, reward, terminated = _step_python(state, force)
            total_reward += reward

            if terminated:
                reached_goal = (
                    abs(state[0]) <= 0.15 and
                    abs(state[2]) <= 0.05 and
                    abs(state[1]) <= 0.10
                )
                break

        outcome = "GOAL" if reached_goal else ("HIT WALL" if terminated else "TIMEOUT")
        print(
            f"Episode {ep + 1}: {step + 1:4d} steps | "
            f"reward = {total_reward:7.1f} | {outcome:9s} | "
            f"x={state[0]:+.3f} m  theta={np.degrees(state[2]):+.2f} deg"
        )

    if do_render:
        import pygame
        pygame.quit()

    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = record_path.suffix.lower()
        if suffix == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=_RENDER_FPS)
        else:
            # MP4 / AVI — requires imageio[ffmpeg]
            imageio.mimsave(str(record_path), all_frames, fps=_RENDER_FPS,
                            macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


# ---- Visualisation -------------------------------------------------------

def plot_value_slice(
    pi: OverheadCraneCuda,
    save_path: Path = Path("results/overhead_crane_cuda_vf_slice.png"),
) -> None:
    """Plot V(x, theta) at x_dot=0, theta_dot=0."""
    x_bins   = np.unique(pi.states_space[:, 0])
    xd_bins  = np.unique(pi.states_space[:, 1])
    th_bins  = np.unique(pi.states_space[:, 2])
    thd_bins = np.unique(pi.states_space[:, 3])
    n0, n1, n2, n3 = len(x_bins), len(xd_bins), len(th_bins), len(thd_bins)
    V4 = pi.value_function.reshape(n0, n1, n2, n3)

    ixd  = int(np.argmin(np.abs(xd_bins)))
    ithd = int(np.argmin(np.abs(thd_bins)))
    V2   = V4[:, ixd, :, ithd]   # shape (n_x, n_theta)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(np.degrees(th_bins), x_bins, V2, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Value")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("x (m)")
    ax.set_title("Overhead Crane CUDA -- V(x, theta) at x_dot=0, theta_dot=0")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value slice saved to {save_path.resolve()}")


def plot_policy_slice(
    pi: OverheadCraneCuda,
    save_path: Path = Path("results/overhead_crane_cuda_policy_slice.png"),
) -> None:
    """Plot optimal force as function of (x, theta) at x_dot=0, theta_dot=0."""
    x_bins   = np.unique(pi.states_space[:, 0])
    xd_bins  = np.unique(pi.states_space[:, 1])
    th_bins  = np.unique(pi.states_space[:, 2])
    thd_bins = np.unique(pi.states_space[:, 3])
    n0, n1, n2, n3 = len(x_bins), len(xd_bins), len(th_bins), len(thd_bins)
    P4 = pi.action_space[pi.policy].reshape(n0, n1, n2, n3)

    ixd  = int(np.argmin(np.abs(xd_bins)))
    ithd = int(np.argmin(np.abs(thd_bins)))
    P2   = P4[:, ixd, :, ithd]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(np.degrees(th_bins), x_bins, P2,
                       cmap="RdBu_r", vmin=-10, vmax=10)
    plt.colorbar(im, ax=ax, label="Force (N): -10=Left  0=Hold  +10=Right")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("x (m)")
    ax.set_title("Overhead Crane CUDA -- Policy at x_dot=0, theta_dot=0")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Policy slice saved to {save_path.resolve()}")


# ---- Entry point ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overhead Crane -- CUDA Policy Iteration (anti-sway control)"
    )
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes with pygame")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save evaluation video to PATH (.gif or .mp4). "
                             "MP4 requires: pip install imageio[ffmpeg]")
    parser.add_argument("--episodes",  type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--start-x",   type=float, default=2.0,
                        help="Initial trolley position for evaluation in metres (default: 2.0)")
    parser.add_argument("--bins",      type=int, default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM}). "
                             "Memory: 20->~25MB, 30->~130MB, 40->~408MB, 50->~1GB")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path,
                        default=Path("results/overhead_crane_cuda_policy.npz"))
    args = parser.parse_args()

    if args.bins != BINS_PER_DIM:
        for key in BINS_SPACE:
            lo, hi = BINS_SPACE[key][0], BINS_SPACE[key][-1]
            BINS_SPACE[key] = np.linspace(lo, hi, args.bins, dtype=np.float32)

    if args.save_path.exists() and not args.retrain:
        print(f"[+] Loading existing policy from {args.save_path}")
        pi = OverheadCraneCuda.load(args.save_path)
    else:
        print("[*] Training new policy...")
        pi = train(args.save_path)

    evaluate(pi, n_episodes=args.episodes, render=args.render,
             start_x=args.start_x, record_path=args.record)
    plot_value_slice(pi)
    plot_policy_slice(pi)

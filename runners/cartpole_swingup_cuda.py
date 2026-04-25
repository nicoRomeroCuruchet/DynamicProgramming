"""
runners/cartpole_swingup_cuda.py — CUDA Policy Iteration for CartPole Swing-Up.

Same dynamics as cartpole_cuda.py, but now the task is to swing the pole up
from the hanging-down position (theta = pi) and balance it upright (theta = 0).

State  : [x        ∈ [-2.5,  2.5]   cart position (terminates at |x| > 2.4)
           x_dot   ∈ [-5.0,  5.0]   cart velocity
           theta   ∈ [-pi,   pi]    pole angle from vertical (full range, wrapped)
           th_dot  ∈ [-8.0,  8.0]   pole angular velocity]

Termination: |x| > 2.4 only — pole is free to spin through full [-pi, pi].

Actions: {-10.0, 0.0, +10.0} N
           Three actions give the policy the option to do nothing during the
           swing phase, which is often optimal.

Reward : cos(theta) - 0.1 * (x / 2.4)^2
           +1.0  when pole upright and cart centred
           -1.0  when pole hanging straight down
           Smooth everywhere — natural gradient toward upright.

Memory (4D scales very gently):
  BINS_PER_DIM=50  →  50^4 =   6.25M states  →  ~  25 MB VRAM
  BINS_PER_DIM=100 → 100^4 = 100.0M states  →  ~ 400 MB VRAM

Dynamics: identical to cartpole_cuda.py (Euler, Florian 2007).
  Constants: g=9.8, masscart=1.0, masspole=0.1, length=0.5, tau=0.02.
  theta is wrapped to (-pi, pi] after each Euler step.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration4D, CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

BINS_PER_DIM = 50   # 50^4 ~ 6.25M states ~ 25 MB VRAM (try 100 for better balance)

BINS_SPACE = {
    "x":      np.linspace(-2.5,    2.5,    BINS_PER_DIM, dtype=np.float32),
    "x_dot":  np.linspace(-5.0,    5.0,    BINS_PER_DIM, dtype=np.float32),
    "theta":  np.linspace(-np.pi,  np.pi,  BINS_PER_DIM, dtype=np.float32),
    "th_dot": np.linspace(-8.0,    8.0,    BINS_PER_DIM, dtype=np.float32),
}

# Three actions: push left / do nothing / push right
ACTION_SPACE = np.array([-10.0, 0.0, 10.0], dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class CartPoleSwingUpCuda(CudaPolicyIteration4D):
    """
    CudaPolicyIteration4D for CartPole swing-up.

    State: [x, x_dot, theta, th_dot]
    theta = 0 → upright (goal).  theta = ±pi → hanging down (start).
    Pole angle is wrapped to (-pi, pi] after each Euler step.
    Only the cart-out-of-bounds condition terminates an episode.
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        #define CP_G          9.8f
        #define CP_MASSCART   1.0f
        #define CP_MASSPOLE   0.1f
        #define CP_TOTALMASS  1.1f
        #define CP_LENGTH     0.5f
        #define CP_POLEMASLEN 0.05f
        #define CP_TAU        0.02f
        #define CP_XMAX       2.4f
        #define CP_PI         3.14159265358979323846f

        __device__ float cp_wrap(float x) {
            float r = fmodf(x + CP_PI, 2.0f * CP_PI);
            if (r < 0.0f) r += 2.0f * CP_PI;
            return r - CP_PI;
        }

        __device__ void step_dynamics(
            float x, float xd, float theta, float thd, float force,
            float* nx, float* nxd, float* ntheta, float* nthd,
            float* reward, bool* terminated
        ) {
            float costh = cosf(theta);
            float sinth = sinf(theta);

            float temp     = (force + CP_POLEMASLEN * thd * thd * sinth)
                             / CP_TOTALMASS;
            float thacc    = (CP_G * sinth - costh * temp)
                             / (CP_LENGTH * (4.0f / 3.0f
                                - CP_MASSPOLE * costh * costh / CP_TOTALMASS));
            float xacc     = temp - CP_POLEMASLEN * thacc * costh / CP_TOTALMASS;

            *nx     = x     + CP_TAU * xd;
            *nxd    = xd    + CP_TAU * xacc;
            *ntheta = cp_wrap(theta + CP_TAU * thd);
            *nthd   = thd   + CP_TAU * thacc;

            // cos(theta) reward: +1 upright, -1 hanging down
            float xn   = *nx / CP_XMAX;
            *reward    = cosf(*ntheta) - 0.1f * xn * xn;

            // Only terminate on cart bounds
            *terminated = (*nx < -CP_XMAX) || (*nx > CP_XMAX);
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        """Cart-out-of-bounds is the only terminal condition."""
        x    = states[:, 0]
        mask = (x < -2.4) | (x > 2.4)
        return mask, 0.0


# ── Python step function (for evaluate) ──────────────────────────────────────

def _step_python(state, force):
    """Single Euler step matching the CUDA dynamics — used at inference."""
    x, xd, theta, thd = state
    g, masscart, masspole, length, tau = 9.8, 1.0, 0.1, 0.5, 0.02
    total_mass    = masscart + masspole
    polemasslen   = masspole * length

    costh  = np.cos(theta)
    sinth  = np.sin(theta)
    temp   = (force + polemasslen * thd**2 * sinth) / total_mass
    thacc  = (g * sinth - costh * temp) / (
        length * (4.0/3.0 - masspole * costh**2 / total_mass)
    )
    xacc   = temp - polemasslen * thacc * costh / total_mass

    nx     = x     + tau * xd
    nxd    = xd    + tau * xacc
    ntheta = ((theta + tau * thd + np.pi) % (2 * np.pi)) - np.pi
    nthd   = thd   + tau * thacc

    xn         = nx / 2.4
    reward     = np.cos(ntheta) - 0.1 * xn**2
    terminated = abs(nx) > 2.4
    next_state = np.array([nx, nxd, ntheta, nthd], dtype=np.float32)
    return next_state, reward, terminated


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/cartpole_swingup_cuda_policy.npz"),
) -> CartPoleSwingUpCuda:
    config = CudaPIConfig(
        gamma         = 0.999,
        theta         = 1e-4,
        max_eval_iter = 10_000,
        max_pi_iter   = 200,
        log_interval  = 500,
    )

    pi = CartPoleSwingUpCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Rendering (pygame) ────────────────────────────────────────────────────────

_SCREEN_W   = 600
_SCREEN_H   = 400
_RENDER_FPS = 50

def _render_frame_pygame(screen, clock, state):
    """Draw cart + pole on a pygame surface. Pole can swing below the cart."""
    import pygame
    from pygame import gfxdraw

    x, _, theta, _ = state

    world_width = 2.4 * 2
    scale       = _SCREEN_W / world_width   # 125 px/m

    cart_width  = 50.0
    cart_height = 30.0
    axle_offset = cart_height / 4.0
    cart_y      = 200              # center of screen — pole can hang below

    pole_w = 10.0
    pole_l = scale * (2 * 0.5)    # 125 px

    cart_x = x * scale + _SCREEN_W / 2.0

    surf = pygame.Surface((_SCREEN_W, _SCREEN_H))
    surf.fill((255, 255, 255))

    # Ground line
    gfxdraw.hline(surf, 0, _SCREEN_W, cart_y, (180, 180, 180))

    # Cart
    l = -cart_width / 2;  r = cart_width / 2
    t =  cart_height / 2; b = -cart_height / 2
    cart_coords = [(c[0] + cart_x, c[1] + cart_y)
                   for c in [(l, b), (l, t), (r, t), (r, b)]]
    gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
    gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

    axle_x = int(cart_x)
    axle_y = int(cart_y + axle_offset)

    # Pole — theta=0 points up, theta=pi points down
    p_coords = []
    for coord in [(-pole_w/2, -pole_w/2),
                  (-pole_w/2,  pole_l - pole_w/2),
                  ( pole_w/2,  pole_l - pole_w/2),
                  ( pole_w/2, -pole_w/2)]:
        v = pygame.math.Vector2(coord).rotate_rad(theta)
        p_coords.append((v.x + cart_x, v.y + axle_y))
    gfxdraw.aapolygon(surf, p_coords, (70, 130, 180))
    gfxdraw.filled_polygon(surf, p_coords, (70, 130, 180))

    gfxdraw.aacircle(surf, axle_x, axle_y, int(pole_w / 2), (129, 132, 203))
    gfxdraw.filled_circle(surf, axle_x, axle_y, int(pole_w / 2), (129, 132, 203))

    # Info overlay
    font = pygame.font.SysFont("monospace", 14)
    lines = [
        f"x     : {state[0]:+.3f} m",
        f"theta : {np.degrees(state[2]):+.1f} deg",
        f"th_dot: {state[3]:+.2f} rad/s",
    ]
    for i, txt in enumerate(lines):
        surf.blit(font.render(txt, True, (60, 60, 60)), (8, 8 + i * 18))

    screen.blit(surf, (0, 0))
    pygame.event.pump()
    clock.tick(_RENDER_FPS)
    pygame.display.flip()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pi: CartPoleSwingUpCuda,
    n_episodes: int = 3,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
    max_steps: int = 1000,
) -> None:
    from utils.barycentric import get_optimal_action

    recording = record_path is not None
    do_render = render or recording
    rng = np.random.default_rng(seed=seed)
    all_frames = []

    if do_render:
        import pygame
        pygame.init()
        pygame.display.init()
        flags = 0 if render else pygame.NOFRAME
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H), flags)
        pygame.display.set_caption("CartPole Swing-Up — CUDA Policy Iteration")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        # Start with pole hanging down + small perturbation
        theta0 = np.pi + rng.uniform(-0.05, 0.05)
        theta0 = ((theta0 + np.pi) % (2 * np.pi)) - np.pi
        state  = np.array([
            rng.uniform(-0.1, 0.1),   # x
            0.0,                       # x_dot
            theta0,                    # theta ≈ pi (hanging down)
            0.0,                       # th_dot
        ], dtype=np.float32)

        total_reward = 0.0
        terminated   = False

        for step in range(max_steps):
            if do_render:
                _render_frame_pygame(screen, clock, state)
                if recording:
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
                break

        outcome = "cart out-of-bounds" if terminated else f"{max_steps} steps"
        print(
            f"Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.1f} | {outcome}"
        )
        print(
            f"  final: x={state[0]:+.3f}  theta={np.degrees(state[2]):+.1f}°"
            f"  th_dot={state[3]:+.2f} rad/s"
        )

    if do_render:
        pygame.quit()

    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.suffix.lower() == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=50, loop=0)
        else:
            imageio.mimsave(str(record_path), all_frames, fps=50, macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


def evaluate_random(n_episodes: int = 3, seed: int = 42) -> None:
    """Random policy rollout — physics sanity check, no training needed."""
    rng = np.random.default_rng(seed=seed)
    for ep in range(n_episodes):
        state = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)
        total_reward = 0.0
        for step in range(500):
            force = float(rng.choice(ACTION_SPACE))
            state, reward, terminated = _step_python(state, force)
            total_reward += reward
            if terminated:
                break
        print(f"[random] Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.1f}")


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_slice(
    pi: CartPoleSwingUpCuda,
    save_path: Path = Path("results/cartpole_swingup_cuda_vf_slice.png"),
) -> None:
    """V(theta, th_dot) at x=0, x_dot=0 — shows full angle landscape."""
    x_bins  = np.unique(pi.states_space[:, 0])
    xd_bins = np.unique(pi.states_space[:, 1])
    th_bins = np.unique(pi.states_space[:, 2])
    wd_bins = np.unique(pi.states_space[:, 3])

    n0, n1, n2, n3 = len(x_bins), len(xd_bins), len(th_bins), len(wd_bins)
    V4 = pi.value_function.reshape(n0, n1, n2, n3)

    ix  = int(np.argmin(np.abs(x_bins)))
    ixd = int(np.argmin(np.abs(xd_bins)))
    V2  = V4[ix, ixd, :, :]   # (n_theta, n_th_dot)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(
        np.degrees(th_bins), wd_bins, V2.T,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Value")
    ax.axvline(  0, color="white", ls="--", lw=1.5, label="upright (goal)")
    ax.axvline( 180, color="red",  ls=":",  lw=1,   label="hanging down (start)")
    ax.axvline(-180, color="red",  ls=":",  lw=1)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("th_dot (rad/s)")
    ax.set_title("CartPole Swing-Up — V(theta, th_dot) at x=0, x_dot=0")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value slice saved to {save_path.resolve()}")


def plot_trajectory(
    pi: CartPoleSwingUpCuda,
    save_path: Path = Path("results/cartpole_swingup_cuda_trajectory.png"),
    seed: int = 42,
    max_steps: int = 1000,
) -> None:
    """theta(t), x(t), reward(t) from a hanging-down start."""
    from utils.barycentric import get_optimal_action

    rng   = np.random.default_rng(seed=seed)
    th0   = np.pi + rng.uniform(-0.05, 0.05)
    th0   = ((th0 + np.pi) % (2 * np.pi)) - np.pi
    state = np.array([0.0, 0.0, th0, 0.0], dtype=np.float32)

    xs, ths, rs = [], [], []

    for _ in range(max_steps):
        force = float(get_optimal_action(
            state, pi.policy, pi.action_space,
            pi.bounds_low, pi.bounds_high,
            pi.grid_shape, pi.strides, pi.corner_bits,
        ))
        state, reward, terminated = _step_python(state, force)
        xs.append(state[0])
        ths.append(np.degrees(state[2]))
        rs.append(reward)
        if terminated:
            break

    t = np.arange(len(xs)) * 0.02

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, ths, color="steelblue")
    axes[0].axhline(  0, color="green", ls="--", lw=1, label="upright (goal)")
    axes[0].axhline( 180, color="red",  ls=":",  lw=1, label="hanging down")
    axes[0].axhline(-180, color="red",  ls=":",  lw=1)
    axes[0].set_ylabel("theta (deg)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("CartPole Swing-Up — trajectory from hanging-down")

    axes[1].plot(t, xs, color="black")
    axes[1].axhline( 0,   color="green", ls="--", lw=1)
    axes[1].axhline( 2.4, color="red",   ls=":",  lw=1)
    axes[1].axhline(-2.4, color="red",   ls=":",  lw=1)
    axes[1].set_ylabel("Cart position (m)")

    axes[2].plot(t, rs, color="purple")
    axes[2].axhline(1.0, color="green", ls="--", lw=1, label="max reward")
    axes[2].axhline(0.0, color="gray",  ls="--", lw=1)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CartPole Swing-Up — CUDA Policy Iteration"
    )
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes with pygame")
    parser.add_argument("--random",    type=int, nargs="?", const=3, default=None, metavar="N",
                        help="Run N random-policy episodes (physics check, no training)")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save evaluation video to PATH (.gif or .mp4)")
    parser.add_argument("--episodes",  type=int, default=3,
                        help="Number of evaluation episodes (default: 3)")
    parser.add_argument("--steps",     type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--bins",      type=int, default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM}). "
                             "Memory: 50->~25MB, 100->~400MB")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip saving plots")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path,
                        default=Path("results/cartpole_swingup_cuda_policy.npz"))
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
            pi = CartPoleSwingUpCuda.load(args.save_path)
        else:
            print("[*] Training new policy...")
            pi = train(args.save_path)

        evaluate(pi, n_episodes=args.episodes, render=args.render,
                 record_path=args.record, seed=args.seed, max_steps=args.steps)
        if not args.no_plot:
            plot_value_slice(pi)
            plot_trajectory(pi, seed=args.seed, max_steps=args.steps)

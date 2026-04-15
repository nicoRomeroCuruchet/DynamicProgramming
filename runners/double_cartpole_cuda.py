"""
runners/double_cartpole_cuda.py — CUDA Policy Iteration for Double CartPole.

A cart carrying a double inverted pendulum (two poles in series). The second
pole is attached to the tip of the first. Both angles are measured from
vertical (absolute, not relative). Dynamics derived via Lagrangian mechanics.

State  : [x        ∈ [-2.5,  2.5]   cart position (terminates at |x| > 2.4)
           x_dot   ∈ [-5.0,  5.0]   cart velocity
           theta1  ∈ [-0.7,  0.7]   first pole angle from vertical (rad)
           th1_dot ∈ [-5.0,  5.0]   first pole angular velocity
           theta2  ∈ [-0.7,  0.7]   second pole angle from vertical (rad)
           th2_dot ∈ [-5.0,  5.0]   second pole angular velocity]

Termination: |x| > 2.4  OR  |theta1| > 36 deg  OR  |theta2| > 36 deg

Actions: 2 force values {-10.0, +10.0} applied to the cart
Reward : +1.0 for every step

Memory estimate (BINS_PER_DIM bins per dimension):
  BINS_PER_DIM=12 → 12^6 =  3.0M states  →  ~90 MB VRAM
  BINS_PER_DIM=15 → 15^6 = 11.4M states  → ~420 MB VRAM
  BINS_PER_DIM=20 → 20^6 = 64.0M states  → ~2.1 GB VRAM

Dynamics (Euler integration, Lagrangian derivation):
  Cart mass M=1.0, pole masses m1=m2=0.1, pole lengths l1=l2=0.5, tau=0.02

  Mass matrix H (3x3, symmetric):
    H = [[M+m1+m2,        (m1+m2)*l1*c1,          m2*l2*c2         ],
         [(m1+m2)*l1*c1,  (m1+m2)*l1^2,           m2*l1*l2*c12     ],
         [m2*l2*c2,        m2*l1*l2*c12,           m2*l2^2          ]]
  where c1=cos(th1), c2=cos(th2), c12=cos(th1-th2)

  RHS vector (generalized forces + centrifugal/Coriolis):
    rhs1 = F + (m1+m2)*l1*th1d^2*s1 + m2*l2*th2d^2*s2
    rhs2 = (m1+m2)*g*l1*s1 - m2*l1*l2*th2d^2*s12
    rhs3 = m2*g*l2*s2      + m2*l1*l2*th1d^2*s12
  where s1=sin(th1), s2=sin(th2), s12=sin(th1-th2)

  Solved analytically via 3x3 matrix inverse (cofactor method):
    [xacc, th1acc, th2acc] = H^{-1} * rhs
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration6D, CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

BINS_PER_DIM = 15   # increase if your GPU has enough VRAM (see memory estimates above)

_TH_THRESH = 36.0 * np.pi / 180.0   # 36 degrees = 0.6283 rad

BINS_SPACE = {
    "x":       np.linspace(-2.5,       2.5,       BINS_PER_DIM, dtype=np.float32),
    "x_dot":   np.linspace(-5.0,       5.0,       BINS_PER_DIM, dtype=np.float32),
    "theta1":  np.linspace(-0.7,       0.7,       BINS_PER_DIM, dtype=np.float32),
    "th1_dot": np.linspace(-5.0,       5.0,       BINS_PER_DIM, dtype=np.float32),
    "theta2":  np.linspace(-0.7,       0.7,       BINS_PER_DIM, dtype=np.float32),
    "th2_dot": np.linspace(-5.0,       5.0,       BINS_PER_DIM, dtype=np.float32),
}

ACTION_SPACE = np.array([-10.0, 10.0], dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class DoubleCartPoleCuda(CudaPolicyIteration6D):
    """
    CudaPolicyIteration6D for double inverted pendulum on a cart.

    State: [x, x_dot, theta1, th1_dot, theta2, th2_dot]
    Both angles are absolute (from vertical, 0 = upright).
    The second pole is attached to the tip of the first.
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        // --- Physical constants -------------------------------------------
        #define DCP_G         9.8f
        #define DCP_M         1.0f    // cart mass
        #define DCP_M1        0.1f    // first pole mass (at tip)
        #define DCP_M2        0.1f    // second pole mass (at tip)
        #define DCP_L1        0.5f    // first pole half-length
        #define DCP_L2        0.5f    // second pole half-length
        #define DCP_TAU       0.02f
        #define DCP_X_THRESH  2.4f
        #define DCP_TH_THRESH 0.62831853f   // 36 deg in rad = pi/5

        __device__ void step_dynamics(
            float x,   float xd,
            float th1, float th1d,
            float th2, float th2d,
            float force,
            float* nx,   float* nxd,
            float* nth1, float* nth1d,
            float* nth2, float* nth2d,
            float* reward, bool* terminated
        ) {
            float m12  = DCP_M1 + DCP_M2;
            float cos1 = cosf(th1), sin1 = sinf(th1);
            float cos2 = cosf(th2), sin2 = sinf(th2);
            float dth  = th1 - th2;
            float cos12 = cosf(dth), sin12 = sinf(dth);

            // --- Mass matrix elements (H is 3x3 symmetric) ---------------
            // H = [[a, b, c],
            //      [b, d, e],
            //      [c, e, f]]
            float a = DCP_M + m12;
            float b = m12   * DCP_L1 * cos1;
            float c = DCP_M2 * DCP_L2 * cos2;
            float d = m12   * DCP_L1 * DCP_L1;
            float e = DCP_M2 * DCP_L1 * DCP_L2 * cos12;
            float f = DCP_M2 * DCP_L2 * DCP_L2;

            // --- Generalized forces + centrifugal/Coriolis ----------------
            // Derived from Lagrangian d/dt(dL/dq_dot) - dL/dq = Q
            float rhs1 = force
                       + m12   * DCP_L1 * th1d * th1d * sin1
                       + DCP_M2 * DCP_L2 * th2d * th2d * sin2;
            float rhs2 = m12   * DCP_G  * DCP_L1 * sin1
                       - DCP_M2 * DCP_L1 * DCP_L2 * th2d * th2d * sin12;
            float rhs3 = DCP_M2 * DCP_G  * DCP_L2 * sin2
                       + DCP_M2 * DCP_L1 * DCP_L2 * th1d * th1d * sin12;

            // --- Analytic 3x3 inverse via cofactors -----------------------
            // Cofactors of symmetric H:
            float cof11 = d * f - e * e;
            float cof12 = e * c - b * f;   // = C_12 = C_21
            float cof13 = b * e - d * c;   // = C_13 = C_31
            float cof22 = a * f - c * c;
            float cof23 = b * c - a * e;   // = C_23 = C_32
            float cof33 = a * d - b * b;

            float det     = a * cof11 + b * cof12 + c * cof13;
            float inv_det = 1.0f / det;

            float xacc   = (cof11 * rhs1 + cof12 * rhs2 + cof13 * rhs3) * inv_det;
            float th1acc = (cof12 * rhs1 + cof22 * rhs2 + cof23 * rhs3) * inv_det;
            float th2acc = (cof13 * rhs1 + cof23 * rhs2 + cof33 * rhs3) * inv_det;

            // --- Euler integration ----------------------------------------
            *nx    = x    + DCP_TAU * xd;
            *nxd   = xd   + DCP_TAU * xacc;
            *nth1  = th1  + DCP_TAU * th1d;
            *nth1d = th1d + DCP_TAU * th1acc;
            *nth2  = th2  + DCP_TAU * th2d;
            *nth2d = th2d + DCP_TAU * th2acc;

            *reward = 1.0f;
            *terminated = (*nx   < -DCP_X_THRESH)  || (*nx   > DCP_X_THRESH)
                       || (*nth1 < -DCP_TH_THRESH) || (*nth1 > DCP_TH_THRESH)
                       || (*nth2 < -DCP_TH_THRESH) || (*nth2 > DCP_TH_THRESH);
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        """Mark grid cells outside failure region as absorbing terminals (V=0)."""
        x    = states[:, 0]
        th1  = states[:, 2]
        th2  = states[:, 4]
        mask = (
            (x   < -2.4)        | (x   > 2.4) |
            (th1 < -_TH_THRESH) | (th1 > _TH_THRESH) |
            (th2 < -_TH_THRESH) | (th2 > _TH_THRESH)
        )
        return mask, 0.0


# ── Python step function (for evaluate, no gym env exists) ────────────────────

def _step_python(state, force):
    """Single Euler step matching the CUDA dynamics — used at inference."""
    x, xd, th1, th1d, th2, th2d = state
    M, m1, m2, l1, l2 = 1.0, 0.1, 0.1, 0.5, 0.5
    g, tau = 9.8, 0.02

    m12   = m1 + m2
    cos1  = np.cos(th1); sin1  = np.sin(th1)
    cos2  = np.cos(th2); sin2  = np.sin(th2)
    dth   = th1 - th2
    cos12 = np.cos(dth); sin12 = np.sin(dth)

    a = M + m12
    b = m12 * l1 * cos1
    c = m2  * l2 * cos2
    d = m12 * l1 * l1
    e = m2  * l1 * l2 * cos12
    f = m2  * l2 * l2

    rhs1 = force + m12 * l1 * th1d**2 * sin1 + m2 * l2 * th2d**2 * sin2
    rhs2 = m12 * g * l1 * sin1 - m2 * l1 * l2 * th2d**2 * sin12
    rhs3 = m2  * g * l2 * sin2 + m2 * l1 * l2 * th1d**2 * sin12

    cof11 = d*f - e*e
    cof12 = e*c - b*f
    cof13 = b*e - d*c
    cof22 = a*f - c*c
    cof23 = b*c - a*e
    cof33 = a*d - b*b
    det   = a*cof11 + b*cof12 + c*cof13

    xacc   = (cof11*rhs1 + cof12*rhs2 + cof13*rhs3) / det
    th1acc = (cof12*rhs1 + cof22*rhs2 + cof23*rhs3) / det
    th2acc = (cof13*rhs1 + cof23*rhs2 + cof33*rhs3) / det

    nx    = x    + tau * xd
    nxd   = xd   + tau * xacc
    nth1  = th1  + tau * th1d
    nth1d = th1d + tau * th1acc
    nth2  = th2  + tau * th2d
    nth2d = th2d + tau * th2acc

    next_state  = np.array([nx, nxd, nth1, nth1d, nth2, nth2d], dtype=np.float32)
    terminated  = (
        abs(nx)   > 2.4 or
        abs(nth1) > _TH_THRESH or
        abs(nth2) > _TH_THRESH
    )
    return next_state, 1.0, terminated


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/double_cartpole_cuda_policy.npz"),
) -> DoubleCartPoleCuda:
    config = CudaPIConfig(
        gamma         = 0.99,
        theta         = 1e-4,
        max_eval_iter = 10_000,
        max_pi_iter   = 100,
        log_interval  = 500,
    )

    pi = DoubleCartPoleCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Rendering (pygame, same style as gymnasium CartPoleEnv) ───────────────────

_SCREEN_W   = 600
_SCREEN_H   = 400
_RENDER_FPS = 50

def _render_frame_pygame(screen, clock, state):
    """Draw cart + double pendulum on the pygame surface. Mirrors CartPoleEnv.render()."""
    import pygame
    from pygame import gfxdraw

    x, _, th1, _, th2, _ = state

    world_width = 2.4 * 2          # x bounds: -2.4 to 2.4
    scale       = _SCREEN_W / world_width   # pixels per meter (125 px/m)

    cart_width  = 50.0
    cart_height = 30.0
    axle_offset = cart_height / 4.0
    cart_y      = 100               # top of cart from top of screen (pre-flip)

    pole_w  = 10.0
    pole_l  = scale * (2 * 0.5)    # 125 px  (2 * half_length * scale)
    pole2_w = 8.0                   # second pole slightly thinner

    cart_x = x * scale + _SCREEN_W / 2.0

    surf = pygame.Surface((_SCREEN_W, _SCREEN_H))
    surf.fill((255, 255, 255))

    # --- Cart ---
    l = -cart_width / 2;  r = cart_width / 2
    t =  cart_height / 2; b = -cart_height / 2
    cart_coords = [(c[0] + cart_x, c[1] + cart_y)
                   for c in [(l, b), (l, t), (r, t), (r, b)]]
    gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
    gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

    axle_x = int(cart_x)
    axle_y = int(cart_y + axle_offset)

    # --- Pole 1 (blue, from cart axle) ---
    p1_coords = []
    for coord in [(-pole_w/2, -pole_w/2),
                  (-pole_w/2,  pole_l - pole_w/2),
                  ( pole_w/2,  pole_l - pole_w/2),
                  ( pole_w/2, -pole_w/2)]:
        v = pygame.math.Vector2(coord).rotate_rad(-th1)
        p1_coords.append((v.x + cart_x, v.y + cart_y + axle_offset))
    gfxdraw.aapolygon(surf, p1_coords, (70, 130, 180))
    gfxdraw.filled_polygon(surf, p1_coords, (70, 130, 180))

    # Cart axle circle
    gfxdraw.aacircle(surf, axle_x, axle_y, int(pole_w / 2), (129, 132, 203))
    gfxdraw.filled_circle(surf, axle_x, axle_y, int(pole_w / 2), (129, 132, 203))

    # --- Joint between pole 1 and pole 2 ---
    tip1 = pygame.math.Vector2(0, pole_l).rotate_rad(-th1)
    joint_x = int(cart_x + tip1.x)
    joint_y = int(cart_y + axle_offset + tip1.y)

    # --- Pole 2 (orange, from tip of pole 1) ---
    p2_coords = []
    for coord in [(-pole2_w/2, -pole2_w/2),
                  (-pole2_w/2,  pole_l - pole2_w/2),
                  ( pole2_w/2,  pole_l - pole2_w/2),
                  ( pole2_w/2, -pole2_w/2)]:
        v = pygame.math.Vector2(coord).rotate_rad(-th2)
        p2_coords.append((v.x + joint_x, v.y + joint_y))
    gfxdraw.aapolygon(surf, p2_coords, (210, 105, 30))
    gfxdraw.filled_polygon(surf, p2_coords, (210, 105, 30))

    # Joint circle
    gfxdraw.aacircle(surf, joint_x, joint_y, int(pole2_w / 2), (129, 132, 203))
    gfxdraw.filled_circle(surf, joint_x, joint_y, int(pole2_w / 2), (129, 132, 203))

    # --- Ground line ---
    gfxdraw.hline(surf, 0, _SCREEN_W, cart_y, (0, 0, 0))

    # Flip vertically (pygame y-down → visual y-up) and blit
    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    pygame.event.pump()
    clock.tick(_RENDER_FPS)
    pygame.display.flip()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(pi: DoubleCartPoleCuda, n_episodes: int = 5, render: bool = False) -> None:
    from utils.barycentric import get_optimal_action

    rng = np.random.default_rng(seed=42)

    if render:
        import pygame
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H))
        pygame.display.set_caption("Double CartPole — CUDA Policy Iteration")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        state = rng.uniform(-0.05, 0.05, size=6).astype(np.float32)
        total_reward = 0.0

        for step in range(500):
            if render:
                _render_frame_pygame(screen, clock, state)

            force = float(get_optimal_action(
                state,
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))
            gym_action = 10.0 if force >= 0.0 else -10.0
            state, reward, terminated = _step_python(state, gym_action)
            total_reward += reward
            if terminated:
                break

        outcome = "SURVIVED" if not terminated else "FELL"
        print(
            f"Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.0f} | {outcome}"
        )

    if render:
        pygame.quit()


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_slice(
    pi: DoubleCartPoleCuda,
    save_path: Path = Path("results/double_cartpole_cuda_vf_slice.png"),
) -> None:
    """
    Plot V(theta1, theta2) slice at x=0, x_dot=0, th1_dot=0, th2_dot=0.
    Shows the value landscape in the angle-angle plane.
    """
    bins = [np.unique(pi.states_space[:, d]) for d in range(6)]
    shape = [len(b) for b in bins]
    V6 = pi.value_function.reshape(shape)

    # Fix x≈0, x_dot≈0, th1_dot≈0, th2_dot≈0
    ix   = int(np.argmin(np.abs(bins[0])))
    ixd  = int(np.argmin(np.abs(bins[1])))
    it1d = int(np.argmin(np.abs(bins[3])))
    it2d = int(np.argmin(np.abs(bins[5])))

    V2 = V6[ix, ixd, :, it1d, :, it2d]   # shape (n_th1, n_th2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        np.degrees(bins[4]), np.degrees(bins[2]), V2,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Value")
    ax.set_xlabel("theta2 (deg)")
    ax.set_ylabel("theta1 (deg)")
    ax.set_title("Double CartPole CUDA — V(th1, th2) at x=xd=th1d=th2d=0")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value slice saved to {save_path.resolve()}")


def plot_policy_slice(
    pi: DoubleCartPoleCuda,
    save_path: Path = Path("results/double_cartpole_cuda_policy_slice.png"),
) -> None:
    """
    Plot optimal force direction as function of (theta1, theta2)
    at x=0, x_dot=0, th1_dot=0, th2_dot=0.
    """
    bins = [np.unique(pi.states_space[:, d]) for d in range(6)]
    shape = [len(b) for b in bins]
    P6 = pi.action_space[pi.policy].reshape(shape)

    ix   = int(np.argmin(np.abs(bins[0])))
    ixd  = int(np.argmin(np.abs(bins[1])))
    it1d = int(np.argmin(np.abs(bins[3])))
    it2d = int(np.argmin(np.abs(bins[5])))

    P2 = P6[ix, ixd, :, it1d, :, it2d]   # shape (n_th1, n_th2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        np.degrees(bins[4]), np.degrees(bins[2]), P2,
        cmap="RdBu", vmin=-10, vmax=10,
    )
    plt.colorbar(im, ax=ax, label="Force (N): -10=Left, +10=Right")
    ax.set_xlabel("theta2 (deg)")
    ax.set_ylabel("theta1 (deg)")
    ax.set_title("Double CartPole CUDA — Policy at x=xd=th1d=th2d=0")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Policy slice saved to {save_path.resolve()}")


# ── Random rollout (physics sanity check, no policy needed) ──────────────────

def run_random(n_episodes: int = 3, render: bool = True) -> None:
    """
    Run episodes with random actions to verify the physics are correct.
    No training required — useful to check dynamics before running PI.
    """
    rng = np.random.default_rng(seed=0)

    if render:
        import pygame
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H))
        pygame.display.set_caption("Double CartPole — Random policy (physics check)")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        state = rng.uniform(-0.05, 0.05, size=6).astype(np.float32)
        total_reward = 0.0

        for step in range(500):
            if render:
                _render_frame_pygame(screen, clock, state)

            force = float(rng.choice(ACTION_SPACE))
            state, reward, terminated = _step_python(state, force)
            total_reward += reward
            if terminated:
                break

        outcome = "SURVIVED" if not terminated else "FELL"
        print(
            f"[random] Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.0f} | {outcome}"
        )

    if render:
        pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Double CartPole — CUDA Policy Iteration")
    parser.add_argument("--render",      action="store_true",
                        help="Render evaluation episodes with pygame")
    parser.add_argument("--random",      action="store_true",
                        help="Run random actions to check physics (no training needed)")
    parser.add_argument("--episodes",    type=int,  default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--bins",        type=int,  default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM}). "
                             "Memory: 12->~90MB, 15->~420MB, 20->~2.1GB")
    parser.add_argument("--retrain",     action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path",   type=Path,
                        default=Path("results/double_cartpole_cuda_policy.npz"))
    args = parser.parse_args()

    # --random bypasses training entirely
    if args.random:
        run_random(n_episodes=args.episodes, render=args.render)
    else:
        # Rebuild grid if --bins differs from default
        if args.bins != BINS_PER_DIM:
            for key in BINS_SPACE:
                lo, hi = BINS_SPACE[key][0], BINS_SPACE[key][-1]
                BINS_SPACE[key] = np.linspace(lo, hi, args.bins, dtype=np.float32)

        if args.save_path.exists() and not args.retrain:
            print(f"[+] Loading existing policy from {args.save_path}")
            pi = DoubleCartPoleCuda.load(args.save_path)
        else:
            print("[*] Training new policy...")
            pi = train(args.save_path)

        evaluate(pi, n_episodes=args.episodes, render=args.render)
        plot_value_slice(pi)
        plot_policy_slice(pi)

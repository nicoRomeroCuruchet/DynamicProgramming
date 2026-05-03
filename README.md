# CUDA Policy Iteration — Continuous RL via Dynamic Programming

Solve continuous-state, continuous-action control problems with GPU-accelerated Policy Iteration.  
The implementation is based on the theoretical framework described in **[Continuous RL via Dynamic Programming](continuous_RL.pdf)**, which rigorously bridges the gap between continuous physics (Hamilton-Jacobi-Bellman equations) and discrete computation (Markov Decision Processes).

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Environments](#environments)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Running the Runners](#running-the-runners)
6. [CLI Reference](#cli-reference)

---

## Theoretical Background

### The Continuous Optimal Control Problem

The framework targets dynamical systems governed by first-order ODEs:

```
ẋ(t) = f(x(t), u(t)),    x(0) = x₀
```

where `x(t) ∈ Ω ⊂ ℝᵈ` is the continuous state and `u(t) ∈ U` is the control input.  
The goal is to find a policy `π: Ω → U` that maximises the **discounted cumulative reward**:

```
V(x) = sup  ∫ γᵗ · r(x(t), u(t)) dt        γ ∈ (0, 1)
        u(·)
```

`V(x)` is called the **value function** — the maximum expected return starting from state `x`.

---

### Hamilton-Jacobi-Bellman (HJB) Equation

The necessary and sufficient optimality condition is the **HJB PDE** (Theorem 2.5):

```
0 = V(x)·ln(γ)  +  sup  { ∇V(x)·f(x,u)  +  r(x,u) }
                    u∈U
```

This non-linear first-order PDE is analytically intractable for all but the simplest systems. The paper provides a rigorous discretisation scheme that converges to its solution as the grid resolution increases.

---

### Bellman Principle & Discretisation

The **Dynamic Programming Principle** (Theorem 2.3) enables recursive decomposition:

```
V(x) = sup  { r(x,u)·τ  +  γᵗ · V(x + τ·f(x,u)) }  +  O(τ²)
        u∈U
```

The continuous state space is discretised into a **uniform grid** with spacing `δ`. After a forward Euler step, the next state `η = x + τ·f(x,u)` typically falls between grid nodes. Its value is recovered via **barycentric interpolation** over the enclosing hypercube:

```
V(η) ≈ Σᵢ  p(η | ξᵢ) · V(ξᵢ)
```

where the weights `p(η | ξᵢ) ≥ 0` satisfy `Σ p = 1` and `Σ p·ξᵢ = η` (convex combination). This preserves second-order accuracy without explicit error correction.

---

### Convergence Guarantee

The barycentric DP operator `Fᵟ` is a **contraction mapping** (Lemma 2.7):

```
‖Fᵟ[V] − Fᵟ[W]‖∞  ≤  λ · ‖V − W‖∞        λ = γ^τ_min < 1
```

By the **Banach Fixed-Point Theorem**, iterating `V_{k+1} = Fᵟ[V_k]` converges to a unique fixed point `V*` regardless of initialisation — the discrete approximation of the continuous value function.

---

### Policy Iteration Algorithm

**Policy Iteration** (Algorithm 1) alternates two steps until the policy no longer changes:

| Step | What it does |
|---|---|
| **Policy Evaluation** | Repeatedly apply the Bellman expectation operator under the current policy until `‖V_{k+1} − V_k‖∞ < θ` |
| **Policy Improvement** | Update every state greedily: `π(s) ← argmax_a  Σ_{s'} p(s'|s,a) · [r + γ·V(s')]` |

Theorem 3.1 guarantees convergence to the **globally optimal policy** in a finite number of outer iterations.

---

### CUDA Parallelisation

The key insight is that the Bellman update at each grid node is **fully independent** — no communication between nodes is needed. This maps perfectly onto a GPU:

```
for each grid node ξᵢ  (one CUDA thread per node, all in parallel)
    best_Q ← -∞
    for each action u ∈ U
        η      ← ξᵢ + τ · f(ξᵢ, u)          // forward Euler step
        V_next ← barycentric_interp(η, V)     // 2ᵈ-corner lookup
        Q      ← r(ξᵢ, u) + γ^τ · V_next     // Bellman target
        if Q > best_Q:  best_Q ← Q,  best_u ← u
    V_new(ξᵢ) ← best_Q
    π(ξᵢ)     ← best_u
```

The environment dynamics `f` and reward `r` are injected as a raw CUDA C device function compiled at runtime via **NVRTC** — no pre-compilation or Makefiles needed.

---

## Environments

All runners share the same CUDA Policy Iteration core. Each environment injects its own dynamics as a raw CUDA C device function via `_dynamics_cuda_src()`, which is compiled at runtime with **NVRTC**.

### CartPole

Classic inverted pendulum. A force is applied to a cart to keep a pole balanced upright.

| State | Bounds |
|---|---|
| Cart position | [-2.5, 2.5] m |
| Cart velocity | [-5, 5] m/s |
| Pole angle | [-0.25, 0.25] rad |
| Pole angular velocity | [-5, 5] rad/s |

**Actions:** {-10, +10} N &nbsp;|&nbsp; **Grid:** 30⁴ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration4D`

![CartPole](gifs/cartpole.gif)

---

### CartPole Swing-Up

Same dynamics as CartPole but with the **full angle range** [-π, π] and the pole starting hanging down (θ=π). The policy must inject energy via cart movement to swing the pole up to θ=0 and stabilise it.

| State | Bounds |
|---|---|
| Cart position | [-2.5, 2.5] m (terminates at \|x\| > 2.4) |
| Cart velocity | [-5, 5] m/s |
| Pole angle θ | [-π, π] rad (full range, wrapped) |
| Pole angular velocity θ̇ | [-10, 10] rad/s |

**Actions:** {-20, -10, 0, +10, +20} N &nbsp;|&nbsp; **Grid:** 50⁴ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration4D`

**Reward shaping:** `cos(θ) − 0.5·E_err − 0.1·(x/x_max)²`, where `E_err = |E_pole − E_target| / (2·E_target)`. The energy term breaks the gradient flatness at the hanging position and propagates the value function from the upright region into the bottom region.

![CartPole Swing-Up](gifs/cartpole_swingup.gif)

---

### Pendulum

Swing up and stabilise a free pendulum at the upright position.

| State | Bounds |
|---|---|
| Angle θ | [-π, π] rad |
| Angular velocity θ̇ | [-8, 8] rad/s |

**Actions:** 21 torque values in [-2, 2] N·m &nbsp;|&nbsp; **Grid:** 200² nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration2D`

![Pendulum](gifs/pendulum.gif)

---

### Mountain Car

Drive an underpowered car out of a valley. Requires learning to build momentum by oscillating.

| State | Bounds |
|---|---|
| Position | [-1.2, 0.6] |
| Velocity | [-0.07, 0.07] |

**Actions:** {-1, 0, +1} &nbsp;|&nbsp; **Grid:** 200² nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration2D`

![Mountain Car](gifs/mountain_car.gif)

---

### Continuous Mountain Car

Same as Mountain Car but with a continuous action space.

| State | Bounds |
|---|---|
| Position | [-1.2, 0.6] |
| Velocity | [-0.07, 0.07] |

**Actions:** 21 force values in [-1, 1] &nbsp;|&nbsp; **Grid:** 200² nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration2D`

![Continuous Mountain Car](gifs/continuous_mountain_car.gif)

---

### Double Pendulum (Underactuated)

Two coupled links on a **fixed pivot** with a single torque applied at the base joint (link 1). The system is **underactuated** — link 2 has no direct actuator, it can only be moved through the inertial coupling with link 1. Generalisation of the classic Pendulum with a second free link, no cart.

Dynamics follow the **Lagrangian formulation** with a 2×2 mass matrix inverted analytically:

```
M = | (m₁+m₂)·l₁²            m₂·l₁·l₂·cos(θ₁−θ₂) |
    | m₂·l₁·l₂·cos(θ₁−θ₂)    m₂·l₂²              |
```

| State | Bounds |
|---|---|
| Link 1 angle θ₁ | [-π, π] rad (full range, wrapped) |
| Link 1 angular velocity θ̇₁ | [-15, 15] rad/s |
| Link 2 angle θ₂ | [-π, π] rad (full range, wrapped) |
| Link 2 angular velocity θ̇₂ | [-15, 15] rad/s |

**Physical parameters:** m₁ = m₂ = 0.1 kg, l₁ = l₂ = 0.5 m, g = 9.8 m/s², τ = 0.02 s

**Actions:** 11 torque values in [-3, 3] N·m, including ±0.05, ±0.15 for fine balancing &nbsp;|&nbsp; **Grid:** up to 50⁴ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration4D`

**Reward shaping:** combined cosine + asymmetric energy penalty + multiplicative upright gate + smooth "deep stillness" bowl + anti-alignment penalty. The deep bowl is the only term that distinguishes "oscillating around upright" from "stationary at upright", so it forces the policy to find the true fixed point instead of a tight limit cycle.

> **Notes:**
> - Useful as a **fast 4D sandbox** for shaping experiments that would be expensive on the 6D Double CartPole. Training: `--bins 15` ≈ 5 sec, `--bins 35` ≈ 6 min, `--bins 50` ≈ 30 min on RTX 3090.
> - Underactuated swing-up has multiple basins of attraction. Sustained stabilisation typically requires `--bins ≥ 50` plus the included fine-action set.

![Double Pendulum](gifs/pendulum_doble.gif)

---

### Double CartPole

Two poles of different lengths balanced simultaneously on a single cart. 6-dimensional state space.

| State | Bounds |
|---|---|
| Cart position | [-2.4, 2.4] m |
| Cart velocity | [-3, 3] m/s |
| Pole 1 angle θ₁ | [-0.3, 0.3] rad |
| Pole 1 angular velocity θ̇₁ | [-3, 3] rad/s |
| Pole 2 angle θ₂ | [-0.3, 0.3] rad |
| Pole 2 angular velocity θ̇₂ | [-3, 3] rad/s |

**Actions:** {-10, 0, +10} N &nbsp;|&nbsp; **Grid:** 12⁶ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration6D`

> **Note:** Memory scales as B⁶. Recommended: `--bins 12` (~90 MB), `--bins 15` (~420 MB), `--bins 20` (~2.1 GB).

![Double CartPole](gifs/double_cartpole.gif)

---

### Double CartPole Swing-Up

Same dynamics as Double CartPole but with the **full angle range** [-π, π] for both poles, starting from hanging-down (θ₁ = θ₂ = π). The policy must (a) inject energy via the cart to swing both poles up, and (b) stabilise the unstable double-inverted equilibrium. 6-dimensional state space.

| State | Bounds |
|---|---|
| Cart position | [-2.5, 2.5] m (terminates at \|x\| > 2.4) |
| Cart velocity | [-8, 8] m/s |
| Pole 1 angle θ₁ | [-π, π] rad (full range, wrapped) |
| Pole 1 angular velocity θ̇₁ | [-15, 15] rad/s |
| Pole 2 angle θ₂ | [-π, π] rad (full range, wrapped) |
| Pole 2 angular velocity θ̇₂ | [-15, 15] rad/s |

**Actions:** 9 force values in [-60, 60] N (with ±3 N for fine balancing) &nbsp;|&nbsp; **Grid:** 12⁶–20⁶ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration6D`

**Reward shaping:** baseline + cosine sum + per-link credits + multiplicative upright gate + asymmetric energy penalty + cart position/velocity penalties + upright-gated velocity damping + terminal bound penalty. Carefully tuned to escape the "link 1 upright + link 2 free-spinning" false attractor.

> **Note:** This is the hardest task in the repo. With `--bins 20` and the included shaping, training takes ~30 min on RTX 3090 and the policy achieves partial swing-up. For consistent stabilisation, see Hybrid Double CartPole below.

![Double CartPole Swing-Up](gifs/double_cartpole_swingup.gif)

---

### Hybrid Double CartPole (DP + DP)

Two-stage controller that **chains** two pre-trained DP policies — pure DP, no LQR or other framework involved.

| Phase | Policy | When active |
|---|---|---|
| **Swing-up** | `double_cartpole_swingup_cuda` (full angle range) | far from upright |
| **Balance** | `double_cartpole_cuda` (narrow ±0.3 rad range, fine bins) | near upright |

**Switch logic** (with hysteresis to prevent rapid toggling):

```
Enter balance:  |θ₁| < 0.28  AND  |θ₂| < 0.28  AND  |ω₁| < 3.5  AND  |ω₂| < 3.5
Return swing:   |θ₁| > 0.35  OR   |θ₂| > 0.35  OR   |ω₁| > 4.5  OR   |ω₂| > 4.5
```

The balance policy is trained on a much finer grid restricted to the upright neighbourhood, giving precision impossible to achieve with a single global policy. The swing-up policy handles the high-energy regime.

**Prerequisite:** both `double_cartpole_swingup_cuda_policy.npz` and `double_cartpole_cuda_policy.npz` must exist in `results/` before running this script — train them individually first.

![Hybrid Double CartPole](gifs/hybrid_double_cartpole.gif)

---

### Overhead Crane (Anti-Sway)

A trolley moves along a fixed rail carrying a suspended load. Goal: transport the load from one end of the rail to the other while minimising swing.

Dynamics follow the **Lagrangian formulation** with a 2×2 mass matrix inverted analytically at each timestep:

```
H = | M+m      m·L·cosθ |       det(H) = m·L²·(M + m·sin²θ) > 0
    | m·L·cosθ  m·L²    |
```

| State | Bounds |
|---|---|
| Trolley position | [-3, 3] m |
| Trolley velocity | [-4, 4] m/s |
| Rope angle θ | [-π/2·1.1, π/2·1.1] rad (~±99°) |
| Rope angular velocity θ̇ | [-4, 4] rad/s |

**Physical parameters:** M = 1 kg (trolley), m = 5 kg (load), L = 1.5 m (rope), g = 9.81 m/s²

**Actions:** {-30, -15, 0, +15, +30} N &nbsp;|&nbsp; **Grid:** 30⁴ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration4D`

**Reward:** `1 - 0.15·xn² - 0.15·ẋn² - 0.45·θn² - 0.25·θ̇n²`  (each term normalised by its grid bound)

![Overhead Crane](gifs/overhead_crane.gif)

---

## Project Structure

```
DynamicProgramming/
├── src/
│   └── cuda_policy_iteration.py   # Core engine: CudaPolicyIteration2D/4D/6D + CudaPIConfig
├── runners/
│   ├── cartpole_cuda.py                   # CartPole              (4D balance)
│   ├── cartpole_swingup_cuda.py           # CartPole              (4D swing-up)
│   ├── pendulum_cuda.py                   # Pendulum              (2D swing-up)
│   ├── mountain_car_cuda.py               # Mountain Car          (2D)
│   ├── continuous_mountain_car_cuda.py    # Continuous Mtn Car    (2D)
│   ├── double_pendulum_swingup_cuda.py    # Double Pendulum       (4D underactuated)
│   ├── double_cartpole_cuda.py            # Double CartPole       (6D balance)
│   ├── double_cartpole_swingup_cuda.py    # Double CartPole       (6D swing-up)
│   ├── hybrid_double_cartpole.py          # Hybrid DP + DP        (6D swing-up + balance)
│   └── overhead_crane_cuda.py             # Overhead Crane        (4D anti-sway)
├── utils/
│   └── barycentric.py             # Barycentric interpolation for policy inference
├── gifs/                          # Pre-recorded environment demos
├── results/                       # Saved policies (.npz) and plots
├── requirements.txt
├── Dockerfile                     # CUDA 12.x runtime + all deps (see Docker section)
├── docker-compose.yml             # Convenience wrapper for `docker compose run pi …`
├── .dockerignore
└── continuous_RL.pdf              # Theoretical foundation
```

Each runner subclasses the appropriate base class and implements a single method:

```python
def _dynamics_cuda_src(self) -> str:
    """Return a raw CUDA C string containing the step_dynamics() device function."""
```

The kernel is compiled at runtime via **NVRTC** — no pre-compilation step needed.

---

## Setup

### Quick path: Docker

If you want a one-shot deploy on a different machine — including older Ubuntu hosts where installing the right CUDA + Python combination by hand is painful — use the included Dockerfile.

**Host prerequisites:**
- NVIDIA GPU + recent driver (compatible with CUDA 12.x by default)
- Docker 20.10+ and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Build and run with `docker compose`:**

```bash
# One-time build (~10-15 min)
docker compose build

# List available runners
docker compose run --rm pi

# Run a specific runner
docker compose run --rm pi python3 runners/pendulum_cuda.py --random 5

# Headless render (no X server needed) — record a GIF directly to your host
docker compose run --rm pi xvfb-run -a \
    python3 runners/double_pendulum_swingup_cuda.py \
            --record gifs/pendulum_doble.gif --episodes 3 --no-plot --bins 50
```

The `docker-compose.yml` mounts `./results`, `./gifs`, and `./trials` from the host into the container, so policies and recordings persist between runs.

**Build for older CUDA versions (e.g. CUDA 11.x):**

Edit `docker-compose.yml`, set:
```yaml
args:
  CUDA_VERSION: 11.8.0
  CUPY_PACKAGE: cupy-cuda11x
```
…then `docker compose build`.

**Without compose** (single `docker run`):

```bash
docker build -t cuda-pi .

docker run --rm --gpus all \
    -v "$PWD/results:/app/results" \
    -v "$PWD/gifs:/app/gifs" \
    cuda-pi \
    python3 runners/cartpole_cuda.py --random 5
```

### Native install (alternative)

### 1. Prerequisites

- **NVIDIA GPU** with CUDA Toolkit installed
- **Python 3.10+**
- `nvidia-smi` available in PATH

Check your CUDA version:

```bash
nvidia-smi | grep "CUDA Version"
# or
nvcc --version
```

### 2. Create a Virtual Environment

```bash
python3 -m venv env
source env/bin/activate        # Linux / macOS
# env\Scripts\activate         # Windows
```

### 3. Install CuPy — Match Your CUDA Version

> **Critical:** CuPy must match the CUDA Toolkit installed on your machine. Installing the wrong version will fail silently or raise `cupy.cuda.compiler.CompileException`.

| CUDA Version | Package |
|---|---|
| 11.x | `cupy-cuda11x` |
| 12.x | `cupy-cuda12x` |
| 13.x | `cupy-cuda13x` |

Edit `requirements.txt` to match your version, then install:

```bash
# Example for CUDA 12.x (default in requirements.txt)
pip install cupy-cuda12x

# Example for CUDA 11.x
pip install cupy-cuda11x
```

Verify CuPy can see your GPU:

```bash
python3 -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

### 4. Install All Dependencies

```bash
pip install -r requirements.txt
```

For MP4 video recording (optional):

```bash
pip install "imageio[ffmpeg]"
```

### 5. Final Check

```bash
python3 -c "
import cupy, numba, numpy, gymnasium, matplotlib
print('cupy   :', cupy.__version__)
print('numba  :', numba.__version__)
print('numpy  :', numpy.__version__)
print('gym    :', gymnasium.__version__)
print('All OK')
"
```

---

## Running the Runners

All runners follow the same workflow:

```
cd DynamicProgramming/

# First run: trains and saves policy, then evaluates
python3 runners/<runner>.py

# Subsequent runs: loads saved policy, skips training
python3 runners/<runner>.py

# Force retraining
python3 runners/<runner>.py --retrain

# Render live + record to GIF
python3 runners/<runner>.py --render --record gifs/<runner>.gif --episodes 3

# Quick random baseline (no training needed)
python3 runners/<runner>.py --random 5
```

### Generate All GIFs

```bash
python3 runners/cartpole_cuda.py                --record gifs/cartpole.gif                --episodes 3 --no-plot
python3 runners/cartpole_swingup_cuda.py        --record gifs/cartpole_swingup.gif        --episodes 3 --no-plot
python3 runners/pendulum_cuda.py                --record gifs/pendulum.gif                --episodes 3 --no-plot
python3 runners/mountain_car_cuda.py            --record gifs/mountain_car.gif            --episodes 3 --no-plot
python3 runners/continuous_mountain_car_cuda.py --record gifs/continuous_mountain_car.gif --episodes 3 --no-plot
python3 runners/double_pendulum_swingup_cuda.py --record gifs/pendulum_doble.gif          --episodes 3 --no-plot --bins 50
python3 runners/double_cartpole_cuda.py         --record gifs/double_cartpole.gif         --episodes 3 --no-plot
python3 runners/double_cartpole_swingup_cuda.py --record gifs/double_cartpole_swingup.gif --episodes 3 --no-plot
python3 runners/hybrid_double_cartpole.py       --record gifs/hybrid_double_cartpole.gif  --episodes 3
python3 runners/overhead_crane_cuda.py          --record gifs/overhead_crane.gif          --episodes 3 --no-plot
```

### Overhead Crane: Custom Target

```bash
# Transport from x=+2.5 m to x=-2.5 m (default)
python3 runners/overhead_crane_cuda.py --retrain --target-x -2.5 --start-x 2.5

# Transport to centre
python3 runners/overhead_crane_cuda.py --retrain --target-x 0.0 --start-x 2.5
```

---

## CLI Reference

**All runners share the exact same set of arguments.** This makes batch scripts portable across environments — change only the runner name and the rest of the command line stays the same.

| Argument | Default | Description |
|---|---|---|
| `--render` | off | Open a live pygame window during evaluation |
| `--record PATH` | off | Save evaluation video to `.gif` or `.mp4` (MP4 needs `imageio[ffmpeg]`) |
| `--random [N]` | off | Run N episodes with a random policy as baseline, no training (default N=5) |
| `--episodes N` | **5** | Number of evaluation episodes |
| `--steps N` | **1000** | Max steps per evaluation episode |
| `--bins N` | per-runner | Bins per state dimension — trades memory for resolution |
| `--seed N` | **42** | Random seed for episode resets |
| `--no-plot` | off | Skip saving value function / policy slice plots |
| `--retrain` | off | Force retraining even if a saved policy exists |
| `--save-path PATH` | `results/<runner>_policy.npz` | Where to save / load the policy `.npz` |

**Runner-specific arguments:**

| Runner | Argument | Default | Description |
|---|---|---|---|
| `overhead_crane_cuda.py` | `--start-x X` | `2.5` | Initial trolley position (m) |
| `overhead_crane_cuda.py` | `--target-x X` | `-2.5` | Target trolley position (m) — baked into CUDA at compile time |

**Notes on the hybrid runner:** `hybrid_double_cartpole.py` accepts the same flags for CLI uniformity, but it never trains — it loads two pre-trained DP policies (`double_cartpole_swingup_cuda_policy.npz` + `double_cartpole_cuda_policy.npz`) and chains them. The flags `--bins`, `--no-plot`, `--retrain`, `--save-path`, and `--random` are accepted but ignored.

**Overhead Crane only:**

| Argument | Default | Description |
|---|---|---|
| `--target-x X` | `-2.5` | Target trolley position (m) — baked into CUDA at compile time |
| `--start-x X` | `2.5` | Initial trolley position for evaluation (m) |

### Memory Guide for `--bins`

| Problem | Dims | `--bins 20` | `--bins 30` | `--bins 40` |
|---|---|---|---|---|
| Pendulum, Mountain Car | 2D | < 1 MB | < 1 MB | < 1 MB |
| CartPole, Overhead Crane | 4D | ~25 MB | ~130 MB | ~408 MB |
| Double CartPole | 6D | ~2.1 GB | ~24 GB* | — |

> *6D grids scale very steeply. For Double CartPole, `--bins 12` (≈90 MB) or `--bins 15` (≈420 MB) are practical choices.

---

## Extending with a New Environment

1. Subclass `CudaPolicyIteration2D`, `4D`, or `6D` from `src/cuda_policy_iteration.py`
2. Implement `_dynamics_cuda_src()` returning a CUDA C string with a `step_dynamics()` device function
3. Define `BINS_SPACE`, `ACTION_SPACE`, and a `train()` / `evaluate()` wrapper
4. Optionally override `_terminal_fn()` to mark absorbing states and `_allocate_tensors_and_compile()` to set their values

```python
class MyEnvCuda(CudaPolicyIteration4D):
    def _dynamics_cuda_src(self) -> str:
        return r'''
        __device__ void step_dynamics(
            float s0, float s1, float s2, float s3, float action,
            float* ns0, float* ns1, float* ns2, float* ns3,
            float* reward, bool* terminated
        ) {
            // your Euler-step dynamics here
        }
        '''
```

The CUDA string is compiled once at the start of `run()` via NVRTC — no Makefile or `.cu` files needed.

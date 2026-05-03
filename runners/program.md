# Autoresearch program — Double CartPole Swing-Up

You are an autonomous research agent tuning the reward shaping and grid
parameters of a CUDA Policy Iteration solver for a double-pendulum-on-cart
swing-up problem. Your goal is to maximize a single external metric.

## Goal

Maximize the **SCORE** computed by `runners/eval_metric.py`. Higher is better.
SCORE = `frac_deep_upright + 0.1 * avg_gate - 0.05 * cart_unsafe`, where:

- `frac_deep_upright` = fraction of timesteps where BOTH `cos(theta_i) > 0.7`
  (about 45 degrees from vertical, both poles above horizontal toward up).
- `avg_gate`           = mean of `max(0, c1) * max(0, c2)` (smooth tiebreaker).
- `cart_unsafe`        = fraction of timesteps where `|x| > 2.0` (penalty).

A well-tuned policy reaches SCORE > 0.5. A perfectly stabilized double pendulum
would reach SCORE > 0.9.

## What you may edit

**Only this file:** `runners/double_cartpole_swingup_cuda.py`

Allowed knobs:
- Coefficients in the reward shaping (CUDA source string and the Python
  `_step_python` mirror — both must match).
- `BINS_SPACE` bounds and `BINS_PER_DIM` (BUT: keep `BINS_PER_DIM <= 14`
  to stay within the time budget).
- `ACTION_SPACE` (the discrete force values).
- The asymmetry / cap of `E_err`.

**Forbidden:**
- Editing `runners/eval_metric.py` (the metric is fixed).
- Editing `runners/trial_runner.sh`.
- Editing `src/cuda_policy_iteration.py` (the solver core).
- Editing the CUDA dynamics block (the mass-matrix solve and Euler step).
  Reward shaping inside the kernel is OK; the physics is not.
- Adding new Python files or restructuring existing ones.
- `BINS_PER_DIM > 14` (each trial would exceed the time budget).

## Loop

1. **Read** `runners/double_cartpole_swingup_cuda.py` and identify the current
   reward shaping. Read `trial_log.md` to see what has been tried before.
2. **Hypothesize** one specific change and *why* you expect it to improve SCORE.
   Be concrete: "increase bonus coefficient from 2.0 to 3.0 to make the upright
   attractor sharper".
3. **Edit** only the coefficients / values you committed to.
4. **Run** `bash runners/trial_runner.sh`. Each trial takes about 5-10 min on a
   single GPU at `BINS_PER_DIM=12`.
5. **Read** the `SCORE=<float>` line and the steady-state stats below it.
6. **Log** the trial in `trial_log.md` (append). Format:
   ```
   ### Trial NNN — <one-line hypothesis>
   - change: <what you changed, e.g. "bonus 2.0 -> 3.0">
   - score:  <value>   (best so far: <value>)
   - stats:  frac_deep=X.XX  gate=X.XX  cos1=X.XX  cos2=X.XX
   - notes:  <1-2 lines about whether the result matched your hypothesis>
   ```
7. **Decide:**
   - If SCORE improved over the previous best: copy the file to
     `trials/trial_NNN_best.py`, mark it as the current best, continue
     hypothesizing from there.
   - If SCORE got worse for **3 consecutive trials**, revert to the current
     best (`cp trials/trial_NNN_best.py runners/double_cartpole_swingup_cuda.py`)
     and try a different direction.
8. **Repeat.**

## Search hints (in approximate priority)

These are reasonable directions; you can ignore them if you have a better idea.

1. **Energy shaping** — the agent currently struggles to lock in `E ~= E_target`
   without overshooting. Try: asymmetric penalties (over vs under), capping vs
   uncapping `E_err`, quadratic vs linear in `|E - E_target|`.
2. **Upright bonus shape** — currently linear in `gate = max(0,c1)*max(0,c2)`.
   Try quadratic, exponential, or a piecewise "deep upright" cliff at
   `cos1 > 0.9 AND cos2 > 0.9`.
3. **Per-link credit weighting** — currently `0.5*g1 + 1.0*g2` (link 2 weighted
   more). Try other splits.
4. **Velocity damping** — currently `0.1 * gate * (w1^2 + w2^2)`. Try larger
   coefficients to force stabilization, or only damp `w2` (the harder link).
5. **Action space** — currently `[-60, -30, -10, -3, 0, 3, 10, 30, 60]`. Try
   adding very fine actions (+/-1) for fine balancing, or removing extreme
   ones (+/-60) if they cause overshoot.
6. **Anti-alignment penalty** — punish states where `(c1 - c2)^2` is large
   (one link up, one down). Coefficient sweep.
7. **Grid bounds** — angular velocity is currently bounded at +/-15. Trajectories
   sometimes exceed +/-20. Try wider bounds with the same `BINS_PER_DIM`.

## Restrictions and safety

- Never increase `BINS_PER_DIM` above 14 (trial budget).
- Never set time budgets longer than the existing `TIMEOUT` in
  `trial_runner.sh`.
- Never delete `trials/*` (the audit trail).
- If a trial errors out (`SCORE=-1.0`), inspect `/tmp/autoresearch_run.log` and
  decide whether to retry or pivot.
- If SCORE has not improved in 10 trials, stop and write a summary in
  `trial_log.md` describing what you learned and where you got stuck.

## What "done" looks like

Either:
- SCORE > 0.7 sustained over the last 3 trials. Stop and write a summary.
- 50 trials completed without improvement. Stop and write a summary.

## Workflow examples

### Starting a session
```bash
# Confirm the harness works:
bash runners/trial_runner.sh

# Read the log and the runner:
cat trial_log.md
cat runners/double_cartpole_swingup_cuda.py | head -250
```

### Running one trial
```bash
# Edit runners/double_cartpole_swingup_cuda.py with your change.
# Then:
bash runners/trial_runner.sh

# Read the SCORE line and append to trial_log.md.
```

### Reverting to a previous best
```bash
cp trials/trial_007_best.py runners/double_cartpole_swingup_cuda.py
```

Begin by reading this file and `trial_log.md`. Then run **one supervised trial**
to confirm the harness works before committing to autonomous exploration.

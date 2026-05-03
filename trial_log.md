# Autoresearch trial log

Append one block per trial. Format:

```
### Trial NNN — <one-line hypothesis>
- change: <what you changed>
- score:  <value>   (best so far: <value>)
- stats:  frac_deep=X.XX  gate=X.XX  cos1=X.XX  cos2=X.XX
- notes:  <1-2 lines about whether the result matched your hypothesis>
```

---

### Trial 000 — baseline (pre-autoresearch)
- change: starting point — no edit yet
- score:  TBD (run the harness once supervised before committing to autonomous mode)
- stats:  see steady-state output of `bash runners/trial_runner.sh`
- notes:  current shaping is asymmetric E_err (1.5x under, 1x over),
          upright bonus 2.0*gate (linear), vel_pen 0.1*gate*w^2,
          ACTION_SPACE 9 forces, BINS_PER_DIM=12 (configurable via env var).

#!/bin/bash
# runners/trial_runner.sh — DO NOT MODIFY.
#
# One full autoresearch trial:
#   1. Wipe cached policy.
#   2. Run the script (which trains + evaluates in one go).
#   3. Compute SCORE via eval_metric.py.
#
# Usage:
#   bash runners/trial_runner.sh
#   BINS=14 bash runners/trial_runner.sh
#
# Output: writes the run log under /tmp; the final stdout is the
# `SCORE=<float>` line plus the metric breakdown from eval_metric.py.

set -e
set -o pipefail

cd "$(dirname "$0")/.."

BINS="${BINS:-12}"
EPISODES="${EPISODES:-3}"
STEPS="${STEPS:-1000}"
TIMEOUT="${TIMEOUT:-1500}"   # seconds for the full train+eval pass

POLICY_PATH="results/double_cartpole_swingup_cuda_policy.npz"
TRAJ_PATH="results/last_trajectory.npz"
RUN_LOG="/tmp/autoresearch_run.log"

echo "=== TRIAL: bins=${BINS}, episodes=${EPISODES}, steps=${STEPS} ==="

# 1. Clean previous policy + trajectory.
rm -f "${POLICY_PATH}" "${TRAJ_PATH}"

# 2. Single pass: train (because policy file is gone) + evaluate.
echo "[trial] training + evaluating (timeout ${TIMEOUT}s) ..."
START=$(date +%s)
if ! timeout "${TIMEOUT}" python runners/double_cartpole_swingup_cuda.py \
        --bins     "${BINS}" \
        --episodes "${EPISODES}" \
        --steps    "${STEPS}" \
        --no-plot \
        > "${RUN_LOG}" 2>&1; then
    echo "ERROR: trial failed or timed out. Tail of log:"
    tail -n 40 "${RUN_LOG}"
    echo "SCORE=-1.0"
    exit 1
fi
END=$(date +%s)
echo "[trial] completed in $((END - START))s"

# 3. Compute the score from the dumped trajectory.
if [ ! -f "${TRAJ_PATH}" ]; then
    echo "ERROR: ${TRAJ_PATH} was not produced by evaluate."
    tail -n 40 "${RUN_LOG}"
    echo "SCORE=-1.0"
    exit 1
fi

python runners/eval_metric.py

# Append the steady-state diagnostic block from the run log for the agent
# to read (cos1/cos2/E_err/etc.). Safe even if grep misses.
echo "--- run log (steady-state stats) ---"
grep -E "Episode|final state|last [0-9]+ steps|reward parts|<E/E_tgt>" "${RUN_LOG}" || true

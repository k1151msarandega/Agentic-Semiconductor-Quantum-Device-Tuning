#!/usr/bin/env bash
set -u
# NOTE: benchmark_phase2.py requires a trained checkpoint at
# experiments/checkpoints/phase1/. Run train_phase1.py first.
# Use --skip-missing-checkpoints only for pure orchestration CI checks.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 2

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

declare -a NAMES=(
  "Agent unit tests"
  "Planning unit tests"
  "Phase012 integration tests"
  "Phase2 fast benchmark (skip missing checkpoints)"
)

declare -a CMDS=(
  "pytest tests/test_agent.py -v --tb=short"
  "pytest tests/test_planning.py -v --tb=short"
  "pytest tests/test_integration_phase012.py -v --tb=short"
  "python experiments/benchmark_phase2.py --fast --out /tmp/phase2_benchmark"
)

declare -a STATUSES=()
overall=0

echo "== Phase 2 Regression Triage =="
echo "Repository: $ROOT_DIR"
echo "Date (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo

for i in "${!CMDS[@]}"; do
  name="${NAMES[$i]}"
  cmd="${CMDS[$i]}"
  echo "--- [$((i + 1))/${#CMDS[@]}] $name"
  echo "+ $cmd"

  if eval "$cmd"; then
    STATUSES+=("PASS")
    echo "Result: PASS"
  else
    code=$?
    STATUSES+=("FAIL($code)")
    overall=1
    echo "Result: FAIL (exit $code)"
  fi

  echo
  sleep 1
done

echo "== Summary =="
for i in "${!STATUSES[@]}"; do
  printf "%2d. %-55s %s\n" "$((i + 1))" "${NAMES[$i]}" "${STATUSES[$i]}"
done

exit "$overall"

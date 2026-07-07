#!/bin/bash
# Chunked LIBERO eval: 15 trials per subprocess (sim hangs after ~18 env resets per process).
# Usage: chunked_eval.sh <label> [HYBRID=1 passed via env]
LABEL=$1
PY=${MK_PYTHON:-/network_volume/megakernels/venv/bin/python}
CKPT=${MK_CHECKPOINT:-/network_volume/megakernels/pi05_libero_pt}
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT=${MK_OUT:-/tmp/mk_chunks}_$LABEL
LOG=${MK_OUT:-/tmp/mk_chunks}_eval_${LABEL}.log
mkdir -p "$OUT"
: > "$LOG"
cd "$DIR" || exit 1
for TID in 0 1 2 3 4 5 6 7 8 9; do
  for OFF in 0 15 30 45; do
    N=15
    [ "$OFF" -eq 45 ] && N=5
    CH="$OUT/task${TID}_off${OFF}.json"
    if [ -s "$CH" ]; then echo "skip task $TID off $OFF (done)" >> "$LOG"; continue; fi
    echo "=== task $TID offset $OFF n $N ===" >> "$LOG"
    _FLASHVLA_SUBTASK_OUTPUT="$CH" PATH=/network_volume/megakernels/venv/bin:$PATH PYTHONPATH=/network_volume/megakernels/LIBERO \
      TORCH_EXTENSIONS_DIR=/root/extbuild \
      timeout 3000 "$PY" eval_libero_hybrid.py --checkpoint "$CKPT" \
        --task_suite libero_spatial --framework torch --seed 1 \
        --num_trials $N --trial_offset $OFF --_task_id $TID >> "$LOG" 2>&1
    echo "chunk exit: $?" >> "$LOG"
  done
done
"$PY" - "$OUT" >> "$LOG" 2>&1 << 'PYEOF'
import json, glob, sys, collections
per = collections.defaultdict(lambda: [0, 0])
for f in glob.glob(sys.argv[1] + "/task*_off*.json"):
    d = json.load(open(f))
    tid = int(f.split("task")[1].split("_")[0])
    s = d.get("successes", d.get("num_successes", 0))
    n = d.get("trials", d.get("num_trials", 0))
    per[tid][0] += s
    per[tid][1] += n
tot_s = sum(v[0] for v in per.values())
tot_n = sum(v[1] for v in per.values())
for t in sorted(per):
    print(f"Task {t}: {per[t][0]}/{per[t][1]}")
print(f"CHUNKED_OVERALL: {tot_s}/{tot_n} = {100.0*tot_s/max(tot_n,1):.1f}%")
PYEOF
echo "SUITE_DONE_$LABEL" >> "$LOG"

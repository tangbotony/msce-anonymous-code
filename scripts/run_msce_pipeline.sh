#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/run_msce_pipeline.sh <session_dir> <output_dir> [procedural|verifier_card]" >&2
  exit 2
fi

SESSION_DIR="$1"
OUT_DIR="$2"
MODE="${3:-procedural}"

mkdir -p "$OUT_DIR"

python -m msce.extract_memory \
  --session-dir "$SESSION_DIR" \
  --output-dir "$OUT_DIR" \
  --parallel-tasks "${MSCE_PARALLEL_TASKS:-4}" \
  --parallel-reflection "${MSCE_PARALLEL_REFLECTION:-4}"

python -m msce.induce_l2 \
  --l1-traces "$OUT_DIR/l1_traces.jsonl" \
  --task-summaries "$OUT_DIR/task_summaries.jsonl" \
  --output "$OUT_DIR/l2_policies.jsonl" \
  --sim-thresh "${MSCE_SIM_THRESH:-0.62}" \
  --min-cluster-size "${MSCE_MIN_CLUSTER_SIZE:-2}" \
  --max-clusters "${MSCE_MAX_CLUSTERS:-60}" \
  --parallel "${MSCE_PARALLEL_L2:-4}" \
  --min-abs-V "${MSCE_MIN_ABS_V:-0.10}"

python -m msce.abstract_l3 \
  --policies "$OUT_DIR/l2_policies.jsonl" \
  --l1-traces "$OUT_DIR/l1_traces.jsonl" \
  --output "$OUT_DIR/l3_topics.jsonl" \
  --parallel "${MSCE_PARALLEL_L3:-3}" \
  --min-policies-per-topic "${MSCE_MIN_POLICIES_PER_TOPIC:-2}"

python -m msce.crystallize_skill \
  --policies "$OUT_DIR/l2_policies.jsonl" \
  --topics "$OUT_DIR/l3_topics.jsonl" \
  --l1-traces "$OUT_DIR/l1_traces.jsonl" \
  --output "$OUT_DIR/skill_bank.jsonl" \
  --parallel "${MSCE_PARALLEL_SKILL:-3}" \
  --n-min "${MSCE_N_MIN:-2}" \
  --mode "$MODE"

python -m msce.embed_skills \
  --skill-bank "$OUT_DIR/skill_bank.jsonl" \
  --output "$OUT_DIR/skill_index.jsonl" \
  --emb-npy "$OUT_DIR/skill_embeddings.npy" \
  --emb-ids "$OUT_DIR/skill_ids.json"

echo "MSCE artifacts written to $OUT_DIR"

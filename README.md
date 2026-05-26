# MSCE: Memory-Skill Co-Evolution

This folder contains the open-source implementation of the algorithm described
in the paper *From Memory to Skills: Evidence-Grounded Governance for
Long-Horizon LLM Agents*.

It is the paper-aligned implementation, not the earlier prototype. The code
keeps the final MSCE pipeline:

- L1 grounded trace memory with reflection-weighted value backfilling
- L2 cross-episode policy induction with expected gain
- L3 environmental cognition abstraction
- evidence-grounded skill crystallization with anti-patterns and boundaries
- multi-route retrieval with applicability and value-calibrated gates
- optional self-evolution from failed evaluation episodes
- optional reasoning verifier-card mode for single-turn reasoning tasks

Private artifacts were intentionally removed. This repository does not include
training sessions, benchmark data, cached jobs, remote-machine scripts, model
endpoints, or API keys.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set provider configuration through environment variables:

```bash
cp .env.example .env
# Fill in .env, then:
set -a
source .env
set +a
```

## Pipeline

Run the full MSCE memory-to-skill build:

```bash
python -m msce.extract_memory \
  --session-dir runs/train_sessions/code \
  --output-dir runs/msce-code \
  --parallel-tasks 4 \
  --parallel-reflection 4

python -m msce.induce_l2 \
  --l1-traces runs/msce-code/l1_traces_v3.jsonl \
  --task-summaries runs/msce-code/task_summaries_v3.jsonl \
  --output runs/msce-code/l2_policies_v3.jsonl \
  --sim-thresh 0.62 \
  --min-cluster-size 2 \
  --min-abs-V 0.10

python -m msce.abstract_l3 \
  --policies runs/msce-code/l2_policies_v3.jsonl \
  --l1-traces runs/msce-code/l1_traces_v3.jsonl \
  --output runs/msce-code/l3_topics_v3.jsonl

python -m msce.crystallize_skill \
  --policies runs/msce-code/l2_policies_v3.jsonl \
  --topics runs/msce-code/l3_topics_v3.jsonl \
  --l1-traces runs/msce-code/l1_traces_v3.jsonl \
  --output runs/msce-code/skill_bank.jsonl \
  --n-min 2

python -m msce.embed_skills \
  --skill-bank runs/msce-code/skill_bank.jsonl \
  --output runs/msce-code/skill_index.jsonl \
  --emb-npy runs/msce-code/skill_embeddings.npy \
  --emb-ids runs/msce-code/skill_ids.json
```

For Mathematical Reasoning, use verifier-card crystallization:

```bash
python -m msce.crystallize_skill \
  --policies runs/msce-reasoning/l2_policies_v3.jsonl \
  --topics runs/msce-reasoning/l3_topics_v3.jsonl \
  --l1-traces runs/msce-reasoning/l1_traces_v3.jsonl \
  --output runs/msce-reasoning/skill_bank.jsonl \
  --mode verifier_card
```


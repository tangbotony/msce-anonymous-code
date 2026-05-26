# MSCE: Memory-Skill Co-Evolution

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
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
  --l1-traces runs/msce-code/l1_traces.jsonl \
  --task-summaries runs/msce-code/task_summaries.jsonl \
  --output runs/msce-code/l2_policies.jsonl \
  --sim-thresh 0.62 \
  --min-cluster-size 2 \
  --min-abs-V 0.10

python -m msce.abstract_l3 \
  --policies runs/msce-code/l2_policies.jsonl \
  --l1-traces runs/msce-code/l1_traces.jsonl \
  --output runs/msce-code/l3_topics.jsonl

python -m msce.crystallize_skill \
  --policies runs/msce-code/l2_policies.jsonl \
  --topics runs/msce-code/l3_topics.jsonl \
  --l1-traces runs/msce-code/l1_traces.jsonl \
  --output runs/msce-code/skill_bank.jsonl \
  --n-min 2

python -m msce.embed_skills \
  --skill-bank runs/msce-code/skill_bank.jsonl \
  --output runs/msce-code/skill_index.jsonl \
  --emb-npy runs/msce-code/skill_embeddings.npy \
  --emb-ids runs/msce-code/skill_ids.json
```

#!/usr/bin/env python3
"""MSCE — Step 3: topic-indexed L3 environmental cognition graph.

Implementation notes:
1. Group L2 policies into topics by intent/artifact tag co-occurrence.
2. Each topic node aggregates:
     - world_knowledge: {spatial_structure, behavior_rules, constraints}
     - tool_atlas: per-command_kind role + v_avg_in_topic + n_uses
     - common_pitfalls: error_kind → remedy mapping
     - related_skills (filled later by crystallize)
     - related_topics edges (intent tag overlap)
3. Each topic node is consumable directly by the retrieval layer.

Input:
    --policies l2_policies.jsonl
    --l1-traces l1_traces.jsonl
Output:
    --output l3_topics.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .clients import chat_completion
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import chat_completion


def call_llm(system: str, user: str, max_tokens: int = 1400, retries: int = 3,
             temperature: float = 0.1) -> str:
    try:
        return chat_completion(
            system=system,
            user=user,
            max_tokens=max_tokens,
            retries=retries,
            temperature=temperature,
            timeout=180,
        )
    except Exception:
        return ""


# ── Topic discovery: cluster policies by intent_tag co-occurrence ──


def _normalize_tag(t: str) -> str:
    return t.strip().lower().replace(" ", "-").replace("_", "-")[:40]


def discover_topics(policies: list[dict], min_policies: int = 2) -> list[dict]:
    """Return list of topic seeds: each has intent_tags + member_policy_ids."""
    # Step 1: aggregate normalised intent_tags
    tag_count = Counter()
    pol_tags = []
    for p in policies:
        tags = [_normalize_tag(t) for t in (p.get("intent_tags") or []) if t]
        tags = list(dict.fromkeys(tags))   # dedupe preserve order
        pol_tags.append(tags)
        for t in tags:
            tag_count[t] += 1

    # Step 2: greedy topic seeds — start from the most frequent tag, gather all
    # policies that have it; remove tags already covered.
    remaining_pol = set(range(len(policies)))
    topics = []
    for tag, cnt in tag_count.most_common():
        if cnt < min_policies:
            continue
        members = [i for i in remaining_pol if tag in pol_tags[i]]
        if len(members) < min_policies:
            continue
        # Collect all tags / artifact_tags across members for richer topic def
        all_intent = Counter()
        all_artifact = Counter()
        for i in members:
            for t in pol_tags[i]:
                all_intent[t] += 1
            for t in (policies[i].get("artifact_tags") or []):
                all_artifact[_normalize_tag(t)] += 1
        related_intent = [t for t, c in all_intent.most_common(6)]
        related_artifact = [t for t, c in all_artifact.most_common(6)]
        topics.append({
            "topic_id": f"topic_{len(topics):03d}_{tag.replace('-', '_')[:30]}",
            "seed_tag": tag,
            "related_intent_tags": related_intent,
            "related_artifact_tags": related_artifact,
            "member_policy_ids": [policies[i].get("policy_id") for i in members],
            "member_idx": members,
        })
        # Remove covered policies for next iteration (each policy can stay in
        # the strongest topic only, to keep graph sparse)
        for i in members:
            if i in remaining_pol:
                remaining_pol.remove(i)

    # Leftover policies (rare tags) -> orphan topic
    if remaining_pol:
        orphan_intent = Counter()
        orphan_artifact = Counter()
        for i in remaining_pol:
            for t in pol_tags[i]:
                orphan_intent[t] += 1
            for t in (policies[i].get("artifact_tags") or []):
                orphan_artifact[_normalize_tag(t)] += 1
        topics.append({
            "topic_id": "topic_orphan",
            "seed_tag": "misc",
            "related_intent_tags": [t for t, _ in orphan_intent.most_common(6)],
            "related_artifact_tags": [t for t, _ in orphan_artifact.most_common(6)],
            "member_policy_ids": [policies[i].get("policy_id") for i in remaining_pol],
            "member_idx": list(remaining_pol),
        })
    return topics


# ── Per-topic stats aggregation ────────────────────────────────────


def aggregate_topic_stats(topic: dict, policies: list[dict],
                          traces: list[dict]) -> dict:
    """Merge tool_effectiveness, error_frequency across member policies."""
    member_pol = [policies[i] for i in topic["member_idx"]]
    src_trace_ids = set()
    for p in member_pol:
        src_trace_ids.update(p.get("source_traces") or [])

    # tool_effectiveness aggregation (weighted by n_uses)
    tool_agg = defaultdict(lambda: {"v_sum_weighted": 0.0, "succ_sum_weighted": 0.0,
                                       "n_uses": 0, "errors": Counter()})
    for p in member_pol:
        te = p.get("tool_effectiveness") or {}
        for kind, st in te.items():
            n = st.get("n_uses", 0)
            tool_agg[kind]["v_sum_weighted"] += st.get("v_avg", 0) * n
            tool_agg[kind]["succ_sum_weighted"] += st.get("success_rate", 0) * n
            tool_agg[kind]["n_uses"] += n
            for err, c in (st.get("errors") or {}).items():
                tool_agg[kind]["errors"][err] += c

    tool_atlas = {}
    for kind, agg in tool_agg.items():
        n = max(agg["n_uses"], 1)
        v_avg = agg["v_sum_weighted"] / n
        succ = agg["succ_sum_weighted"] / n
        if v_avg >= 0.45 and succ >= 0.6:
            role = "primary"
        elif v_avg >= 0.30:
            role = "support"
        elif v_avg >= 0.15:
            role = "verify"
        else:
            role = "explore_only"
        tool_atlas[kind] = {
            "role": role,
            "v_avg_in_topic": round(v_avg, 3),
            "success_rate_in_topic": round(succ, 3),
            "n_uses": agg["n_uses"],
            "common_errors": dict(agg["errors"].most_common(3)),
        }

    # common pitfalls — from error_frequency aggregated, and trace fail_diagnoses
    err_freq = Counter()
    diagnoses = []
    for p in member_pol:
        for e, c in (p.get("error_frequency") or {}).items():
            err_freq[e] += c

    # also walk traces for explicit error_diagnosis examples (high-signal text)
    trace_index = {t["trace_id"]: t for t in traces}
    for tid in src_trace_ids:
        t = trace_index.get(tid)
        if not t:
            continue
        d = (t.get("reflection_v2") or {}).get("error_diagnosis") or ""
        if d and len(d) > 10:
            diagnoses.append(d[:200])

    return {
        "tool_atlas": tool_atlas,
        "error_frequency": dict(err_freq.most_common(8)),
        "diagnoses_sample": diagnoses[:8],
        "n_member_policies": len(member_pol),
        "n_source_traces": len(src_trace_ids),
    }


# ── LLM topic synthesis prompt ─────────────────────────────────────

TOPIC_SYS = """You abstract environmental cognition from cross-task policy evidence.

An environmental cognition model is **declarative** knowledge about how the environment IS: its topology, its causal/behavioural regularities, its taboos. It is **NOT** a recipe for what to do; that lives in the L2 procedural layer, generated by a separate prompt. Cross-contamination on either side dilutes both.

Input POLICIES: a list of L2 policies (with trigger / procedure / verification / boundary / support / gain), plus a short sample of the L1 traces that minted each. Every policy shares a compatible domain (matched by primary tag / tool).

Produce ONE environmental cognition model describing the **environment** those policies operate in. It must answer:

- Environment topology: what lives where, what is the shape of this environment? Pure facts of existence and structure.
  GOOD: "Alpine containers ship musl libc, no glibc"; "Node project repos group source under src/"; "macOS bundles BSD sed; Linux distros bundle GNU sed".
  BAD: "use apk add to install system libs"; "prefer Python scripts over sed on macOS".

- Inference rules: how does the environment causally respond to common stimuli? Phrase as cause-effect, NOT as guidance.
  GOOD: "loading a glibc-linked binary wheel inside Alpine raises a dynamic-link error"; "editing config.yaml does not propagate until the process restarts".
  BAD: "if pip install fails, install dev libs and retry"; "always restart the service after editing config".

- Constraints: what facts of the environment make some actions unsafe or invalid? State the FACT, not the avoidance behavior.
  GOOD: "node_modules/ is rewritten by npm install; manual edits are lost on the next sync"; "production database tables hold customer data; destructive DDL is irreversible".
  BAD: "don't edit node_modules/ directly"; "don't run DROP TABLE in production".

Do NOT, under any section:
- Use imperative or recommendation verbs (do / don't / should / use / prefer / avoid / try / install / run).
- Restate a single trace; the cognition model must generalise across policies.
- Include advice tied to a single user or session.

Return JSON:
{
  "title": "short noun phrase, e.g. 'Alpine python dependency model'",
  "domain_tags": ["tag1", "tag2"],
  "environment": [
    { "label": "...", "description": "...", "evidenceIds": ["po_...", "tr_..."] }
  ],
  "inference": [
    { "label": "...", "description": "...", "evidenceIds": [] }
  ],
  "constraints": [
    { "label": "...", "description": "...", "evidenceIds": [] }
  ],
  "body": "rendered markdown summary of the three sections",
  "confidence": number in [0, 1],
  "supersedes_cognition_ids": []
}"""


TOPIC_USER_TEMPLATE = """Input POLICIES:
seed_tag: {seed_tag}
related_intent_tags: {intent}
related_artifact_tags: {artifact}
member_policies: {n_pol}, source_traces: {n_tr}

tool_atlas:
{tool_atlas_md}

error_frequency:
{err_md}

representative failure diagnoses:
{diag_md}

member policy summaries:
{pol_md}

Return the environmental cognition JSON."""


def render_tool_atlas_md(atlas: dict) -> str:
    rows = []
    for kind, st in sorted(atlas.items(), key=lambda kv: -kv[1]["v_avg_in_topic"]):
        rows.append(f"  - {kind} [{st['role']}]: v_avg={st['v_avg_in_topic']:.2f}, "
                    f"succ={st['success_rate_in_topic']:.2f}, n={st['n_uses']}, "
                    f"errors={st['common_errors']}")
    return "\n".join(rows) or "(empty)"


def render_pol_md(member_pol: list[dict]) -> str:
    rows = []
    for p in member_pol[:6]:
        trig = p.get("trigger", {})
        if isinstance(trig, dict):
            trig_text = trig.get("text", "")[:200]
        else:
            trig_text = str(trig)[:200]
        proc = p.get("procedure", {})
        if isinstance(proc, dict):
            proc_text = proc.get("text", "")[:300]
        else:
            proc_text = str(proc)[:300]
        rows.append(
            f"  - {p.get('policy_id')}: trigger='{trig_text}' | "
            f"action='{p.get('action', proc_text)}' | gain={p.get('expected_gain', {})}"
        )
    return "\n".join(rows)


def synthesise_topic(topic: dict, policies: list[dict],
                      traces: list[dict]) -> dict | None:
    stats = aggregate_topic_stats(topic, policies, traces)
    member_pol = [policies[i] for i in topic["member_idx"]]

    user = TOPIC_USER_TEMPLATE.format(
        seed_tag=topic["seed_tag"],
        intent=topic["related_intent_tags"],
        artifact=topic["related_artifact_tags"],
        n_pol=stats["n_member_policies"],
        n_tr=stats["n_source_traces"],
        tool_atlas_md=render_tool_atlas_md(stats["tool_atlas"]),
        err_md=json.dumps(stats["error_frequency"], ensure_ascii=False),
        diag_md="\n".join(f"  - {d}" for d in stats["diagnoses_sample"]) or "(none)",
        pol_md=render_pol_md(member_pol),
    )
    raw = call_llm(TOPIC_SYS, user, max_tokens=1400)
    if not raw.strip():
        node = {
            "title": topic["seed_tag"],
            "domain_tags": [topic["seed_tag"]],
            "environment": [],
            "inference": [],
            "constraints": [],
            "body": "",
            "confidence": 0.0,
        }
    else:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            node = json.loads(raw)
        except Exception:
            lo, hi = raw.find("{"), raw.rfind("}")
            if lo < 0 or hi < 0:
                node = {"title": topic["seed_tag"], "environment": [],
                        "inference": [], "constraints": [], "body": "",
                        "confidence": 0.0}
            else:
                try:
                    node = json.loads(raw[lo:hi + 1])
                except Exception:
                    node = {"title": topic["seed_tag"], "environment": [],
                            "inference": [], "constraints": [], "body": "",
                            "confidence": 0.0}

    environment = node.get("environment", []) or []
    inference = node.get("inference", []) or []
    constraints = node.get("constraints", []) or []

    def _join_desc(items: list) -> str:
        out = []
        for item in items:
            if isinstance(item, dict):
                label = item.get("label", "")
                desc = item.get("description", "")
                out.append(f"{label}: {desc}".strip(": "))
            else:
                out.append(str(item))
        return "\n".join(x for x in out if x)

    out = {
        "topic_id": topic["topic_id"],
        "topic_name": node.get("title", topic["seed_tag"]),
        "seed_tag": topic["seed_tag"],
        "related_intent_tags": topic["related_intent_tags"],
        "related_artifact_tags": topic["related_artifact_tags"],
        "domain_tags": node.get("domain_tags", []),
        "environment": environment,
        "inference": inference,
        "constraints": constraints,
        "body": node.get("body", ""),
        "confidence": node.get("confidence", 0.0),
        "supersedes_cognition_ids": node.get("supersedes_cognition_ids", []),
        "world_knowledge": {
            "spatial_structure": _join_desc(environment),
            "behavior_rules": _join_desc(inference),
            "constraints": _join_desc(constraints),
        },
        "tool_atlas": stats["tool_atlas"],
        "common_pitfalls": [
            {
                "signal": x.get("label", "") if isinstance(x, dict) else str(x),
                "remedy": x.get("description", "") if isinstance(x, dict) else "",
            }
            for x in constraints[:6]
        ],
        "member_policy_ids": topic["member_policy_ids"],
        "n_member_policies": stats["n_member_policies"],
        "n_source_traces": stats["n_source_traces"],
        "related_topics": [],   # to be filled after all topics built
    }
    return out


# ── related_topics edges ──────────────────────────────────────────


def link_related_topics(topics: list[dict]) -> None:
    """For each pair, link if Jaccard(related_intent_tags) >= 0.34."""
    sets = [set(t["related_intent_tags"]) for t in topics]
    for i, ti in enumerate(topics):
        sims = []
        for j, tj in enumerate(topics):
            if i == j:
                continue
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union == 0:
                continue
            jac = inter / union
            if jac >= 0.34:
                sims.append((jac, tj["topic_id"]))
        sims.sort(key=lambda x: -x[0])
        ti["related_topics"] = [t for _, t in sims[:4]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--l1-traces", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--min-policies-per-topic", type=int, default=2)
    args = ap.parse_args()

    policies = [json.loads(l) for l in open(args.policies)]
    print(f"Loaded {len(policies)} L2 policies")
    traces = []
    with open(args.l1_traces) as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except Exception:
                pass
    print(f"Loaded {len(traces)} L1 traces")

    topics = discover_topics(policies, args.min_policies_per_topic)
    print(f"Discovered {len(topics)} topics")
    for t in topics:
        print(f"  - {t['topic_id']} ({t['seed_tag']}): {len(t['member_idx'])} policies")

    nodes = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(synthesise_topic, t, policies, traces): t
                for t in topics}
        for f in as_completed(futs):
            try:
                node = f.result()
            except Exception as e:
                print(f"  ERR: {e}", file=sys.stderr)
                continue
            if node:
                nodes.append(node)

    link_related_topics(nodes)

    with open(args.output, "w") as fo:
        for n in nodes:
            fo.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(nodes)} topic nodes to {args.output}")


if __name__ == "__main__":
    sys.exit(main())

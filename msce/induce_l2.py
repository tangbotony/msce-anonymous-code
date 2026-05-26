#!/usr/bin/env python3
"""MSCE — Step 2: cross-task L2 policy induction with causal evidence.

Implementation notes:
1. Cluster on command kind, error kind, task tags, and state semantics.
2. Per-cluster compute:
     - tool_effectiveness: v_avg/success_rate per command_kind
     - essential_steps (high-V, appear in most successful traces)
     - optional_steps  (mid-V, appear sometimes)
     - redundant_steps (low-V, low success_rate)
     - anti_pattern    (failed traces' common command_kind + error_kind)
3. expected_gain: V_avg_with vs V_avg_without (cross-cluster baseline).
4. Output L2 includes intent_tags / artifact_tags (carried from L1).

Input:
    --l1-traces  output/l1_traces.jsonl from extract_memory
    --task-summaries output/task_summaries.jsonl (for V_avg_without baseline)
Output:
    --output  l2_policies.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
try:
    from .clients import chat_completion, embedding_batch as provider_embedding_batch
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import chat_completion, embedding_batch as provider_embedding_batch

TAU_V = 0.5
GAIN_N0 = 5
GAIN_ANCHOR = 0.5


def embed_batch(texts: list[str], batch_size: int = 16) -> np.ndarray:
    return provider_embedding_batch(texts, batch_size=batch_size)


def call_llm(system: str, user: str, max_tokens: int = 1500, retries: int = 3,
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


# ── trace digest (for embedding) ───────────────────────────────────


def trace_digest(t: dict) -> str:
    tc = t.get("tool_call", {})
    ob = t.get("observation", {})
    rf = t.get("reflection_v2", {})
    return (
        f"intent: {' '.join(t.get('task_intent_tags', []))} | "
        f"artifact: {' '.join(t.get('task_artifact_tags', []))}\n"
        f"state: {t.get('state_summary','')[:300]}\n"
        f"tool: {tc.get('command_kind','')} {tc.get('command_text','')[:200]}\n"
        f"obs: exit={ob.get('exit_code')} err={ob.get('error_kind')} "
        f"files={ob.get('files_created')} sig={rf.get('key_signal','')[:120]}\n"
        f"refl: prog={rf.get('is_progress')} blk={rf.get('is_blocker')} "
        f"diag={rf.get('error_diagnosis','')[:200]}"
    )


# ── clustering ─────────────────────────────────────────────────────


def greedy_clusters(embs: np.ndarray, task_ids: list[str],
                    sim_thresh: float = 0.62,
                    min_size: int = 2, max_size: int = 12) -> list[list[int]]:
    n = len(embs)
    if n == 0:
        return []
    sim = embs @ embs.T
    assigned = [False] * n
    clusters = []
    seeds = list(range(n))
    np.random.shuffle(seeds)
    for seed in seeds:
        if assigned[seed]:
            continue
        neighbors = np.argsort(-sim[seed])[: max_size * 3]
        cluster = [seed]
        cluster_tasks = {task_ids[seed]}
        for j in neighbors:
            if j == seed or assigned[j]:
                continue
            if sim[seed, j] < sim_thresh:
                break
            cluster.append(int(j))
            cluster_tasks.add(task_ids[j])
            if len(cluster) >= max_size:
                break
        if len(cluster) >= min_size and len(cluster_tasks) >= 2:
            for i in cluster:
                assigned[i] = True
            clusters.append(cluster)
    return clusters


# ── per-cluster causal stats ───────────────────────────────────────


def per_cluster_stats(traces: list[dict]) -> dict:
    """Compute tool_effectiveness, error frequency, intent/artifact tags."""
    by_kind = defaultdict(list)
    by_err = Counter()
    intent_c = Counter()
    artifact_c = Counter()
    blocker_kinds = Counter()
    progress_kinds = Counter()
    for t in traces:
        tc = t.get("tool_call", {})
        ob = t.get("observation", {})
        rf = t.get("reflection_v2", {})
        k = tc.get("command_kind", "other")
        v = float(t.get("V", 0))
        exit_code = ob.get("exit_code")
        err = ob.get("error_kind")
        # success: exit_code == 0 or files_created not empty or progress=true
        succ = (exit_code == 0) or bool(ob.get("files_created")) or bool(rf.get("is_progress"))
        by_kind[k].append({"v": v, "succ": succ, "err": err})
        if err:
            by_err[err] += 1
        if rf.get("is_blocker"):
            blocker_kinds[k] += 1
        if rf.get("is_progress"):
            progress_kinds[k] += 1
        for it in t.get("task_intent_tags", []):
            intent_c[it] += 1
        for it in t.get("task_artifact_tags", []):
            artifact_c[it] += 1

    tool_eff = {}
    for k, lst in by_kind.items():
        n = len(lst)
        v_avg = sum(x["v"] for x in lst) / n if n else 0.0
        succ_rate = sum(1 for x in lst if x["succ"]) / n if n else 0.0
        tool_eff[k] = {
            "v_avg": round(v_avg, 3),
            "success_rate": round(succ_rate, 3),
            "n_uses": n,
            "errors": dict(Counter(x["err"] for x in lst if x["err"]).most_common(3)),
        }

    # Classify essential / optional / redundant by V quartiles
    sorted_kinds = sorted(tool_eff.items(), key=lambda kv: -kv[1]["v_avg"])
    essential, optional, redundant = [], [], []
    for k, st in sorted_kinds:
        if st["n_uses"] < 2:
            continue
        if st["v_avg"] >= 0.45 and st["success_rate"] >= 0.6 and progress_kinds.get(k, 0) >= 1:
            essential.append({"tool_kind": k, "v_avg": st["v_avg"],
                              "success_rate": st["success_rate"], "n_uses": st["n_uses"]})
        elif st["v_avg"] >= 0.20:
            optional.append({"tool_kind": k, "v_avg": st["v_avg"],
                             "success_rate": st["success_rate"], "n_uses": st["n_uses"]})
        else:
            redundant.append({"tool_kind": k, "v_avg": st["v_avg"],
                              "success_rate": st["success_rate"], "n_uses": st["n_uses"]})

    return {
        "tool_effectiveness": tool_eff,
        "error_frequency": dict(by_err.most_common(8)),
        "blocker_kinds": dict(blocker_kinds.most_common(5)),
        "progress_kinds": dict(progress_kinds.most_common(5)),
        "intent_tags_top": [t for t, _ in intent_c.most_common(6)],
        "artifact_tags_top": [t for t, _ in artifact_c.most_common(6)],
        "essential": essential,
        "optional": optional,
        "redundant": redundant,
    }


# ── LLM L2 induction prompt ────────────────────────────────────────

INDUCE_SYS = """You induce reusable **procedural policies** from agent experience.

A policy is a "how-to": "when you see condition X in the agent's state, do action Y, verify with Z, watch out for caveat W." It is **NOT** a description of the environment.

Input TRACES: a list of { state_summary, action, outcome, utility } records that all share a similar state signature.

Produce ONE policy describing the action pattern. The policy must:
- Name a TRIGGER recognizable from the agent's STATE: a condition the agent can detect at the moment of decision (an error code, a missing file, a request shape). NOT a fact about the environment in general.
- Prescribe an ACTION template: a parameterized step or short step sequence. Templates over single exact commands. NOT a single example.
- Note at least one CAVEAT or failure mode observed in the traces: a step-level pitfall, NOT a generic environment taboo.
- Generalize across the input traces, not restate one of them.

Boundaries what NOT to write:
This output is a **procedural policy**, not environmental cognition. Environmental cognition lives in a separate layer (L3) generated by a different prompt. Cross-contamination on either side dilutes both.

Do NOT write any of these; they belong to L3 (environmental cognition), not here:
- Topology facts: "Alpine containers ship musl libc"; "Python deps form a 3-layer stack"; "src/components/ holds React components".
- Environment behavioural rules in pure declarative form: "binary wheels are incompatible with musl"; "the service reads config only at startup".
- Environment taboos detached from a specific action choice: "this directory is read-only"; "production tables shouldn't be DROPped lightly".

If a trace tells you the environment looks a certain way, FOLD that fact INTO the trigger or caveat as a state-level CONDITION the agent can check, not as a standalone description.

Same fact, two framings:
If the underlying truth is "Alpine containers don't ship system dev libs by default":
Express here (procedural): "When pip install fails inside an Alpine container with a missing system library, run apk add <pkg>-dev then retry pip."
Do NOT express here (declarative that's L3's job): "Alpine container images ship only the pure-Python tier of the Python dependency stack."

Return JSON:
{
  "title": "short imperative title",
  "trigger": "state-level condition the agent can detect",
  "action": "templated step or step sequence",
  "rationale": "why this action works ON THESE TRACES (not why the environment behaves this way)",
  "caveats": ["step-level pitfall string", ...],
  "confidence": number in [0, 1],
  "support_trace_ids": ["tr_...", ...]
}"""


INDUCE_USER_TEMPLATE = """Input TRACES selected from {n_tasks} distinct tasks:
{traces_md}

Grounding statistics for reference:
- intent_tags: {intent_tags}
- artifact_tags: {artifact_tags}
- tool_effectiveness:
{tool_eff_md}
- essential_candidates: {ess_md}
- optional_candidates: {opt_md}
- redundant_candidates: {red_md}
- error_frequency: {error_freq}
- blocker_kinds: {blocker_kinds}

Return the L2 procedural policy JSON."""


def render_tool_eff_md(stats: dict) -> str:
    rows = []
    for k, st in sorted(stats["tool_effectiveness"].items(), key=lambda kv: -kv[1]["v_avg"]):
        rows.append(f"  - {k}: v_avg={st['v_avg']:.2f}, succ={st['success_rate']:.2f}, "
                    f"n={st['n_uses']}, errors={st['errors']}")
    return "\n".join(rows) if rows else "(empty)"


def render_step_md(items: list[dict]) -> str:
    return "\n".join(f"  - {it['tool_kind']} (v_avg={it['v_avg']:.2f}, "
                     f"succ={it['success_rate']:.2f}, n={it['n_uses']})"
                     for it in items) or "(none)"


def render_traces_md(traces: list[dict], n_show: int = 6) -> str:
    traces_sorted = sorted(traces, key=lambda t: -float(t.get("V", 0)))
    out = []
    for i, t in enumerate(traces_sorted[:n_show]):
        tc = t["tool_call"]
        ob = t["observation"]
        rf = t.get("reflection_v2", {})
        out.append(
            "{"
            f"\"trace_id\":\"{t['trace_id']}\", "
            f"\"task_id\":\"{t['task_id']}\", "
            f"\"state_summary\":{json.dumps(t.get('state_summary','')[:300], ensure_ascii=False)}, "
            f"\"action\":{json.dumps(tc.get('command_text','')[:260], ensure_ascii=False)}, "
            f"\"command_kind\":\"{tc.get('command_kind','')}\", "
            f"\"outcome\":{json.dumps((ob.get('raw_truncated') or '')[:260], ensure_ascii=False)}, "
            f"\"error_kind\":{json.dumps(ob.get('error_kind'))}, "
            f"\"utility\":{float(t.get('V', 0)):.3f}, "
            f"\"alpha\":{float(t.get('alpha', 0)):.3f}, "
            f"\"reflection\":{json.dumps((rf.get('what_happened') or rf.get('error_diagnosis') or '')[:200], ensure_ascii=False)}"
            "}"
        )
    return "\n".join(out)


def induce_l2_for_cluster(traces: list[dict]) -> dict | None:
    if len(traces) < 2:
        return None
    stats = per_cluster_stats(traces)
    user = INDUCE_USER_TEMPLATE.format(
        n_traces=len(traces),
        n_tasks=len({t["task_id"] for t in traces}),
        intent_tags=stats["intent_tags_top"],
        artifact_tags=stats["artifact_tags_top"],
        tool_eff_md=render_tool_eff_md(stats),
        ess_md=render_step_md(stats["essential"]),
        opt_md=render_step_md(stats["optional"]),
        red_md=render_step_md(stats["redundant"]),
        error_freq=stats["error_frequency"],
        blocker_kinds=stats["blocker_kinds"],
        n_show=min(8, len(traces)),
        traces_md=render_traces_md(traces, n_show=8),
    )
    raw = call_llm(INDUCE_SYS, user, max_tokens=1800)
    if not raw.strip():
        return None
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        d = json.loads(raw)
    except Exception:
        lo, hi = raw.find("{"), raw.rfind("}")
        if lo < 0 or hi < 0:
            return None
        try:
            d = json.loads(raw[lo:hi + 1])
        except Exception:
            return None
    if not isinstance(d, dict) or "trigger" not in d or "action" not in d:
        return None
    trigger_text = str(d.get("trigger", ""))[:600]
    action_text = str(d.get("action", ""))[:1200]
    caveats = [str(x)[:300] for x in (d.get("caveats") or [])][:8]
    # attach stats (ground-truth signals from data, not LLM hallucination)
    d["tool_effectiveness"] = stats["tool_effectiveness"]
    d["intent_tags"] = stats["intent_tags_top"]
    d["artifact_tags"] = stats["artifact_tags_top"]
    d["error_frequency"] = stats["error_frequency"]
    # Compatibility fields used by downstream construction and retrieval.
    d["trigger_text"] = trigger_text
    d["action_text"] = action_text
    d["procedure"] = {
        "text": action_text,
        "essential_steps": [
            {"tool_kind": e["tool_kind"], "purpose": "core",
             "must_succeed": True} for e in stats["essential"][:3]
        ],
        "optional_steps": [
            {"tool_kind": o["tool_kind"], "purpose": "support",
             "skippable_if": "context already covered"} for o in stats["optional"][:3]
        ],
        "redundant_steps": [
            {"tool_kind": r["tool_kind"], "reason": "low V_avg in this context"}
            for r in stats["redundant"][:3]
        ],
    }
    d["verification"] = {"text": "verify the expected outcome in the trace or task verifier"}
    d["anti_pattern"] = {"text": "; ".join(caveats), "fail_signals": list(stats["error_frequency"].keys())[:8]}
    d["scope"] = {"applies_to": trigger_text, "does_not_apply_to": "state does not match trigger or caveats"}
    support = [x for x in (d.get("support_trace_ids") or []) if x in {t["trace_id"] for t in traces}]
    d["source_traces"] = support or [t["trace_id"] for t in traces]
    d["source_tasks"] = sorted({t["task_id"] for t in traces})
    d["n_support"] = len(traces)
    d["V_avg"] = sum(float(t.get("V", 0)) for t in traces) / len(traces)
    d["V_pos_count"] = sum(1 for t in traces if float(t.get("V", 0)) > 0)
    return d


# ── expected_gain (cross-cluster baseline) ────────────────────────


def compute_expected_gain(policy: dict, all_traces: list[dict],
                          all_task_summaries: dict) -> dict:
    """Estimate paper-style policy gain from with/without trace sets."""
    intent = set(policy.get("intent_tags") or [])
    src_traces = set(policy.get("source_traces") or [])
    policy_cmd_kinds = set()
    for s in policy.get("procedure", {}).get("essential_steps", []) or []:
        if isinstance(s, dict) and s.get("tool_kind"):
            policy_cmd_kinds.add(s["tool_kind"])

    source_values = [
        float(t.get("V", 0))
        for t in all_traces
        if t.get("trace_id") in src_traces
    ]
    if len(source_values) >= 3:
        vals = np.asarray(source_values, dtype=np.float32)
        weights = np.exp((vals - float(vals.max())) / TAU_V)
        weights = weights / max(float(weights.sum()), 1e-8)
        v_with = float(np.sum(weights * vals))
        with_aggregate = "softmax_weighted"
    elif source_values:
        v_with = sum(source_values) / len(source_values)
        with_aggregate = "mean"
    else:
        v_with = float(policy.get("V_avg", 0.0))
        with_aggregate = "policy_v_avg_fallback"

    vs = []
    for t in all_traces:
        if t["trace_id"] in src_traces:
            continue
        t_intent = set(t.get("task_intent_tags") or [])
        if intent and (intent & t_intent):
            # same topic, but not using policy's essential tools
            kk = t.get("tool_call", {}).get("command_kind")
            if kk not in policy_cmd_kinds:
                vs.append(float(t.get("V", 0)))
    v_without_raw = sum(vs) / len(vs) if vs else 0.0
    v_without = (
        len(vs) * v_without_raw + GAIN_N0 * GAIN_ANCHOR
    ) / (len(vs) + GAIN_N0)
    n_pos = sum(1 for t in all_traces
                if t["trace_id"] in src_traces and float(t.get("V", 0)) > 0)
    n_neg = sum(1 for t in all_traces
                if t["trace_id"] in src_traces and float(t.get("V", 0)) <= 0)
    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "v_avg_with": round(v_with, 3),
        "v_avg_without": round(v_without, 3),
        "v_avg_without_raw": round(v_without_raw, 3),
        "gain": round(v_with - v_without, 3),
        "n_baseline_traces": len(vs),
        "with_aggregate": with_aggregate,
        "without_shrinkage": {"n0": GAIN_N0, "anchor": GAIN_ANCHOR},
        "tau_v": TAU_V,
    }


# ── main ─────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-traces", required=True)
    ap.add_argument("--task-summaries", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sim-thresh", type=float, default=0.62)
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--max-clusters", type=int, default=60)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--min-abs-V", type=float, default=0.10)
    args = ap.parse_args()

    # Load L1 traces (full schema)
    traces = []
    with open(args.l1_traces) as f:
        for line in f:
            try:
                t = json.loads(line)
            except Exception:
                continue
            if abs(float(t.get("V", 0))) < args.min_abs_V:
                continue
            traces.append(t)
    print(f"Loaded {len(traces)} traces (after |V|>={args.min_abs_V} filter)")
    if not traces:
        return 0

    # Embed via digest
    print("Embedding trace digests with bge-m3 ...")
    texts = [trace_digest(t) for t in traces]
    embs = embed_batch(texts)
    print(f"  embeddings: {embs.shape}")

    # Cluster
    task_ids = [t["task_id"] for t in traces]
    clusters = greedy_clusters(embs, task_ids, args.sim_thresh,
                                args.min_cluster_size)
    print(f"Found {len(clusters)} cross-task clusters")
    if not clusters:
        return 0
    # cap by V_avg
    scored = [(sum(float(traces[i].get("V", 0)) for i in cl) / len(cl), cl)
              for cl in clusters]
    scored.sort(key=lambda x: -x[0])
    clusters = [cl for _, cl in scored[: args.max_clusters]]
    print(f"  keeping top {len(clusters)} clusters by V_avg")

    # Load all traces (no V filter) for expected_gain baseline
    all_traces = []
    with open(args.l1_traces) as f:
        for line in f:
            try:
                all_traces.append(json.loads(line))
            except Exception:
                pass

    # Load task summaries
    task_summ = {}
    if args.task_summaries and Path(args.task_summaries).exists():
        for line in open(args.task_summaries):
            try:
                s = json.loads(line)
                task_summ[s["task_id"]] = s
            except Exception:
                pass

    print("Inducing L2 policies (LLM) ...")
    n_done, n_kept = 0, 0
    with open(args.output, "w") as fo, ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(induce_l2_for_cluster,
                          [traces[i] for i in cl]): ci
                for ci, cl in enumerate(clusters)}
        for f in as_completed(futs):
            ci = futs[f]
            try:
                pol = f.result()
            except Exception as e:
                print(f"  ci={ci} ERR: {e}", file=sys.stderr)
                continue
            n_done += 1
            if pol is None:
                continue
            pol["policy_id"] = f"L2_{ci:03d}"
            pol["expected_gain"] = compute_expected_gain(pol, all_traces, task_summ)
            fo.write(json.dumps(pol, ensure_ascii=False) + "\n")
            fo.flush()
            n_kept += 1
            if n_done % 5 == 0:
                print(f"  induced {n_done}/{len(clusters)} (kept {n_kept})")

    print(f"\nDone. Wrote {n_kept} L2 policies to {args.output}")


if __name__ == "__main__":
    sys.exit(main())

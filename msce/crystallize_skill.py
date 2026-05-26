#!/usr/bin/env python3
"""MSCE — Step 4: crystallize callable skills, anchored to topic nodes.

Each crystallized skill carries:
  - applicability_signature (intent_tags, artifact_tags, command_kinds)
  - trigger / procedure (incl. essential / optional / redundant) / verification
  - anti_pattern, scope
  - tool_effectiveness (causal)
  - expected_gain (V_with - V_without)
  - topic_id   (which L3 topic node it belongs to)
  - embedding_text (consolidated retrieval text)

Crystallization criteria:
  - n_support >= n_min (default 2)
  - V_avg > 0
  - V_pos_rate >= 0.5
  - expected_gain.gain > 0 (must beat no-skill baseline)
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .clients import chat_completion
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import chat_completion


def call_llm(system: str, user: str, max_tokens: int = 1500, retries: int = 3,
             temperature: float = 0.1, model: str | None = None) -> str:
    try:
        return chat_completion(
            system=system,
            user=user,
            max_tokens=max_tokens,
            retries=retries,
            temperature=temperature,
            timeout=180,
            model=model,
        )
    except Exception:
        return ""


# ── prompts ────────────────────────────────────────────────────────

SKILL_SYS = """You crystallize a skill an agent should be able to call.

Input:
- POLICY: the L2 policy being promoted (trigger / action / rationale / caveats).
- EVIDENCE: 3..10 successful traces that support the policy.
- EVIDENCE_TOOLS: the exhaustive list of tool/command names that actually appeared in the evidence traces' tool calls. This is the ground-truth whitelist your `tools` output MUST be a subset of this list.
- COUNTER_EXAMPLES (optional): traces with V < 0 from the same context; failures the policy is meant to prevent.
- REPAIR_HINTS (optional): a JSON block { preference: [...], antiPattern: [...] } attached to the policy by the decision-repair pipeline. These are concrete "prefer / avoid" lines synthesised from earlier failures + user feedback; treat them as authoritative seeds for `decision_guidance` below.
- NAMING_SPACE: a list of existing skill names to avoid colliding with.

Return JSON:
{
  "name": "snake_case_identifier, 32 chars, unique vs NAMING_SPACE",
  "display_title": "human title in user's language",
  "summary": "2-3 sentence description of what the skill does and when to use it",
  "parameters": [
    { "name": "...", "type": "string|number|boolean|enum", "required": true|false,
      "description": "...", "enum": ["..."] }
  ],
  "preconditions": ["bullet", ...],
  "steps": [
    { "title": "short", "body": "markdown-friendly paragraph describing the step" }
  ],
  "examples": [
    { "input": "...", "expected": "..." }
  ],
  "tools": ["tool_or_command_name", ...],
  "decision_guidance": {
    "preference": ["Prefer: ", ...],
    "anti_pattern": ["Avoid: ", ...]
  },
  "tags": ["optional string", ...]
}

Rules:
- `tools` MUST only contain names from EVIDENCE_TOOLS. Never invent tool names that are not in the whitelist. Include every tool the skill's procedure actually invokes; omit tools not referenced in your steps.
- Keep "steps" short (2-6 items).
- `summary` must be self-contained so the agent can decide whether to call this skill without reading the full SKILL.md.
- For `decision_guidance`: if REPAIR_HINTS is non-empty, fold each line in verbatim or lightly normalised; you MAY add 1-2 extra entries derived from contrasting EVIDENCE vs COUNTER_EXAMPLES. Don't invent guidance unsupported by the inputs.
- Each decision-guidance entry should be one short, actionable sentence (under 200 chars).
- Empty arrays are fine when there's nothing to say; never fabricate."""


SKILL_USER_TEMPLATE = """POLICY:
{policy_json}

EVIDENCE:
{evidence}

EVIDENCE_TOOLS:
{evidence_tools}

COUNTER_EXAMPLES:
{counter_examples}

REPAIR_HINTS:
{repair_hints}

NAMING_SPACE:
{naming_space}

Return the callable skill JSON."""


def render_evidence(traces: list[dict], n: int = 4) -> str:
    out = []
    for t in traces[:n]:
        tc = t["tool_call"]
        ob = t["observation"]
        rf = t.get("reflection_v2", {})
        out.append(
            f"[{t['trace_id']}] V={t['V']:.2f} α={t.get('alpha',0):.2f}\n"
            f"  tool={tc['command_kind']} cmd={tc['command_text'][:200]}\n"
            f"  obs.exit={ob.get('exit_code')} err={ob.get('error_kind')} files={ob.get('files_created')}\n"
            f"  refl: {rf.get('what_happened','')[:140]}"
        )
    return "\n".join(out) or "(none)"


def evidence_tool_names(traces: list[dict]) -> list[str]:
    names = set()
    for t in traces:
        tc = t.get("tool_call") or {}
        for key in ("command_kind", "name"):
            value = tc.get(key)
            if value:
                names.add(str(value))
    return sorted(names)


def render_policy_payload(policy: dict) -> dict:
    keys = (
        "title", "trigger", "action", "rationale", "caveats",
        "confidence", "support_trace_ids",
    )
    payload = {k: policy.get(k) for k in keys if k in policy}
    if "trigger" not in payload and policy.get("trigger_text"):
        payload["trigger"] = policy.get("trigger_text")
    if "action" not in payload and policy.get("action_text"):
        payload["action"] = policy.get("action_text")
    return payload


def parse_json_object(raw: str | dict) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        lo, hi = raw.find("{"), raw.rfind("}")
        if lo < 0 or hi < 0:
            return None
        try:
            return json.loads(raw[lo:hi + 1])
        except Exception:
            return None


def reliability_record(n_support: int, v_avg: float, v_pos_count: int) -> dict:
    eta = (v_pos_count + 1.0) / (n_support + 2.0)
    state = "active" if n_support >= 2 and eta >= 0.6 else "probationary"
    return {
        "n_support": n_support,
        "v_avg": round(v_avg, 3),
        "v_pos_rate": round(v_pos_count / max(n_support, 1), 3),
        "eta": round(eta, 3),
        "lifecycle_state": state,
    }


def crystallize_one(policy: dict, topic: dict, evidence: list[dict],
                    n_min: int = 2,
                    negative_candidates: list[dict] | None = None) -> dict | None:
    n_sup = policy.get("n_support", 0)
    v_avg = policy.get("V_avg", 0.0)
    v_pos = policy.get("V_pos_count", 0)
    gain = (policy.get("expected_gain") or {}).get("gain", 0.0)
    if n_sup < n_min or v_avg <= 0 or (v_pos / max(n_sup, 1)) < 0.5 or gain <= 0:
        return None

    evidence_tools = evidence_tool_names(evidence)
    if not evidence_tools:
        evidence_tools = sorted((policy.get("tool_effectiveness") or {}).keys())

    user = SKILL_USER_TEMPLATE.format(
        policy_json=json.dumps(render_policy_payload(policy), ensure_ascii=False, indent=2),
        evidence=render_evidence(evidence),
        evidence_tools=json.dumps(evidence_tools, ensure_ascii=False),
        counter_examples=render_evidence(negative_candidates or []),
        repair_hints=json.dumps(
            policy.get("repair_hints") or {"preference": [], "antiPattern": []},
            ensure_ascii=False,
        ),
        naming_space=json.dumps([], ensure_ascii=False),
    )
    raw = call_llm(SKILL_SYS, user, max_tokens=1500)
    d = parse_json_object(raw)
    if not isinstance(d, dict):
        return None

    required = ("name", "summary", "steps", "tools")
    if any(k not in d for k in required):
        return None
    tools = [str(t) for t in (d.get("tools") or []) if str(t)]
    if set(tools) - set(evidence_tools):
        return None
    d["tools"] = sorted(dict.fromkeys(tools))

    steps = d.get("steps") if isinstance(d.get("steps"), list) else []
    step_lines = []
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            title = str(step.get("title", f"Step {idx}")).strip()
            body = str(step.get("body", "")).strip()
            step_lines.append(f"{title}: {body}" if body else title)
        else:
            step_lines.append(str(step))
    procedure_text = "\n".join(step_lines)[:1200]

    intent_tags = list(topic.get("related_intent_tags", [])) or list(policy.get("intent_tags", []))
    artifact_tags = list(topic.get("related_artifact_tags", [])) or list(policy.get("artifact_tags", []))
    trigger_text = str(policy.get("trigger") or policy.get("trigger_text") or "")[:600]
    action_text = str(policy.get("action") or policy.get("action_text") or procedure_text)[:1200]
    anti_items = []
    guidance = d.get("decision_guidance") if isinstance(d.get("decision_guidance"), dict) else {}
    for item in guidance.get("anti_pattern") or []:
        anti_items.append(str(item))
    caveats = [str(x) for x in (policy.get("caveats") or [])]

    # Compatibility metadata for indexing/retrieval; the LLM-facing prompt and
    # required fields above follow the paper's Appendix E skill schema.
    d["trigger"] = {
        "text": trigger_text,
        "command_kinds": d["tools"],
        "error_kinds": list((policy.get("error_frequency") or {}).keys())[:8],
    }
    d["procedure"] = {
        "text": action_text or procedure_text,
        "essential_steps": [
            {"tool_kind": t, "purpose": "skill step", "must_succeed": True}
            for t in d["tools"]
        ],
        "optional_steps": [],
        "redundant_steps": [],
    }
    d["verification"] = {
        "text": "verify the task's expected outcome or generated artifact",
    }
    d["anti_pattern"] = {
        "text": "; ".join(anti_items or caveats)[:800],
        "fail_signals": list((policy.get("error_frequency") or {}).keys())[:8],
    }
    d["scope"] = {
        "applies_to": "; ".join(d.get("preconditions") or [])[:800],
        "does_not_apply_to": "preconditions do not hold",
    }
    d["applicability_signature"] = {
        "intent_tags": intent_tags,
        "artifact_tags": artifact_tags,
        "command_kinds": d["tools"],
    }
    d["topic_id"] = topic.get("topic_id")
    d["tool_effectiveness"] = policy.get("tool_effectiveness", {})
    d["expected_gain"] = policy.get("expected_gain", {})
    d["source_policy"] = policy.get("policy_id")
    d["source_traces"] = policy.get("source_traces", [])[:8]
    d["reliability"] = reliability_record(n_sup, v_avg, v_pos)
    # Ensure abstract + lexical_keywords for dense/sparse retrieval.
    if not d.get("abstract"):
        d["abstract"] = (
            f"{d.get('summary','')[:240]} Trigger: {trigger_text[:180]} "
            f"Action: {action_text[:180]}"
        )[:600]
    if not d.get("lexical_keywords"):
        kws = set()
        for t in intent_tags + artifact_tags + d["tools"] + list(d.get("tags") or []):
            if t:
                kws.add(str(t).lower())
        d["lexical_keywords"] = sorted(kws)[:20]
    # Used by dense retrieval.
    d["embedding_text"] = d["abstract"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--topics", required=True, help="l3_topics.jsonl")
    ap.add_argument("--l1-traces", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--n-min", type=int, default=2)
    args = ap.parse_args()

    policies = [json.loads(l) for l in open(args.policies)]
    print(f"Loaded {len(policies)} L2 policies")

    topics = [json.loads(l) for l in open(args.topics)]
    # policy_id -> topic
    pol_to_topic = {}
    for t in topics:
        for pid in t.get("member_policy_ids") or []:
            pol_to_topic[pid] = t
    print(f"Loaded {len(topics)} topic nodes")

    # Load L1 for evidence
    trace_index = {}
    with open(args.l1_traces) as f:
        for line in f:
            try:
                tt = json.loads(line)
                trace_index[tt["trace_id"]] = tt
            except Exception:
                pass

    candidates = []
    for p in policies:
        n_sup = p.get("n_support", 0)
        v_avg = p.get("V_avg", 0.0)
        v_pos = p.get("V_pos_count", 0)
        gain = (p.get("expected_gain") or {}).get("gain", 0.0)
        if n_sup >= args.n_min and v_avg > 0 and v_pos / max(n_sup, 1) >= 0.5 and gain > 0:
            candidates.append(p)
    print(f"After 4-condition filter (n>={args.n_min}, V>0, pos_rate>=0.5, gain>0): "
          f"{len(candidates)}/{len(policies)} candidates")

    with open(args.output, "w") as fo, ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {}
        for p in candidates:
            topic = pol_to_topic.get(p["policy_id"]) or {"topic_id": "topic_orphan",
                                                          "topic_name": "Misc",
                                                          "related_intent_tags": [],
                                                          "related_artifact_tags": [],
                                                          "tool_atlas": {},
                                                          "common_pitfalls": []}
            # Build positive_pool (high-V successful steps in this cluster)
            # and negative_pool (cluster failing steps + same-task failing steps).
            tids = p.get("source_traces") or []
            cluster_traces = [trace_index[t] for t in tids if t in trace_index]
            positive_pool = sorted(
                [t for t in cluster_traces if float(t.get("V", 0)) > 0],
                key=lambda x: -float(x.get("V", 0)),
            )[:8]
            # Failing/blocker steps from same cluster (if cluster has any)
            neg_pool = [
                t for t in cluster_traces
                if float(t.get("V", 0)) <= 0
                or (t.get("reflection_v2") or {}).get("is_blocker")
                or (t.get("observation") or {}).get("error_kind")
            ]
            # If cluster has no negatives, pull from same source tasks' other
            # steps (broader fail signal, still task-correlated).
            if not neg_pool:
                src_task_ids = {t.get("task_id") for t in cluster_traces}
                for tid, tt in trace_index.items():
                    if tt.get("task_id") in src_task_ids and tid not in tids:
                        rf = tt.get("reflection_v2") or {}
                        ob = tt.get("observation") or {}
                        if (float(tt.get("V", 0)) <= 0
                                or rf.get("is_blocker")
                                or ob.get("error_kind")):
                            neg_pool.append(tt)
                            if len(neg_pool) >= 6:
                                break
            futs[ex.submit(crystallize_one, p, topic,
                            positive_pool[:6], args.n_min, neg_pool[:6])] = p["policy_id"]
        i = 0
        prefix = "msce_skill"
        for f in as_completed(futs):
            pid = futs[f]
            try:
                skill = f.result()
            except Exception as e:
                print(f"  {pid} ERR: {e}", file=sys.stderr)
                continue
            if skill is None:
                continue
            i += 1
            skill["skill_id"] = f"{prefix}_{i:03d}"
            fo.write(json.dumps(skill, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 5 == 0:
                print(f"  crystallized {i}/{len(candidates)}")

    # Patch topics with related_skills so retrieval can route topic → skills.
    skills = [json.loads(l) for l in open(args.output)]
    topic_to_skills = defaultdict(list)
    for s in skills:
        topic_to_skills[s.get("topic_id")].append(s["skill_id"])
    # rewrite topic file
    out_topics = []
    for t in topics:
        t["related_skills"] = topic_to_skills.get(t["topic_id"], [])
        out_topics.append(t)
    with open(args.topics, "w") as ft:
        for t in out_topics:
            ft.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(skills)} skills to {args.output}")
    print(f"Patched {args.topics} with related_skills per topic.")


if __name__ == "__main__":
    sys.exit(main())

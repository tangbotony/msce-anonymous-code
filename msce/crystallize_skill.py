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
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .clients import DEFAULT_LLM_MODEL, chat_completion
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import DEFAULT_LLM_MODEL, chat_completion

# Stronger model for verifier card mode (where archetype mis-tag would be most harmful).
LLM_MODEL_BIG = os.environ.get("MSCE_LLM_MODEL_BIG", DEFAULT_LLM_MODEL)


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

SKILL_SYS = """你是一个 skill crystallizer。给你一条经过统计验证的 L2 policy + 关联的 topic L3 + 高价值 evidence traces，输出严格 JSON：

{
  "name": "<2-5 词英文 kebab-case 短名, 例: python3-fallback-on-cmdnotfound>",
  "summary": "<1 句话总结这个 skill 在 agent 工作流中做什么>",
  "abstract": "<80-150 字独立 abstract，专门给检索使用：包含 (a) 适用场景/触发条件; (b) 核心动作; (c) 验证信号; (d) 适用边界。要可读、自包含、不能依赖外部上下文>",
  "lexical_keywords": ["<5-12 个英文/中文关键词，用于 BM25 倒排索引，覆盖 tool/error/artifact/intent>"],
  "trigger": {
    "text": "<什么 state/error/子任务下应用本 skill>",
    "command_kinds": [...],
    "error_kinds":   [...]
  },
  "procedure": {
    "text": "<可执行步骤 1-5 条；只放 essential + 必要 optional>",
    "essential_steps": [
      {"tool_kind": "...", "purpose": "...", "must_succeed": true}
    ],
    "optional_steps": [
      {"tool_kind": "...", "purpose": "...", "skippable_if": "..."}
    ],
    "redundant_steps": [
      {"tool_kind": "...", "reason": "在该 topic 下 V_avg 低，是浪费"}
    ]
  },
  "verification": {"text": "<exit_code/产物/回读的具体信号>"},
  "anti_pattern": {
    "text": "<反例：什么情况不要用 / agent 容易做错什么>",
    "fail_signals": ["..."]
  },
  "scope": {"applies_to": "...", "does_not_apply_to": "..."}
}

要求：
- procedure 三段必须基于输入提供的统计 grounding，不要重新发明
- redundant_steps 必须给"为什么是浪费"（用 topic 内 v_avg 低做依据）
- anti_pattern.fail_signals 是真实 trace 中出现的错误关键词
- 只输出 JSON，无 markdown 包裹
"""


# ── Reasoning verifier-card mode (single-turn reasoning, no tool loop) ────────
#
# For single-turn reasoning tasks (e.g. competition math), procedural skills
# add noise: there is no environment to interact with, so "how to solve" prompts
# disrupt the model's direct reasoning. MSCE-R replaces solver skills with
# *verifier cards* — short cognitive controls that name the archetype,
# encode the most common pitfall, and supply a one-shot sanity check.
#
# Each card is ~80-150 tokens and contains only:
#   - archetype (one of fixed taxonomy, used for routing)
#   - check    (the invariant the answer MUST satisfy)
#   - trap     (the most common error in training failures)
#   - skip_if  (when NOT to apply)
#
# Crystallized with a stronger model (LLM_MODEL_BIG) to reduce archetype
# mis-classification risk.

VERIFIER_CARD_SYS = """你是一个数学推理 verifier card 蒸馏器。给你一条 L2 policy + 关联 topic L3 + 正例/反例 traces，
你要把它蒸馏成一张极短的 verifier card——**不是教模型怎么解题，而是教它怎么验答案、避陷阱**。

输出严格 JSON：

{
  "archetype": "<必须严格从下列固定 taxonomy 选一个，不可自创：
    probability-process    : 概率/马尔可夫链/吸收概率/期望步数
    counting-enumeration   : 计数/组合/排列方案数
    permutation-cycle      : 置换/cycle 长度/置换幂等
    geometry-metric        : 平面/坐标几何 求长度/面积/比值
    recurrence-expectation : 递推方程/期望递推/生成函数
    number-theory          : 数论/整除/同余/质因数
    algebra-equation       : 代数方程/不等式/化简求值
    general-math           : 其他/无法确定>",
  "name": "<2-5 词英文 kebab-case 短名，必须包含 archetype, 例: probability-process-verify>",
  "summary": "<1 句话描述这张 card 适用题型与核心验证操作>",
  "abstract": "<60-120 字独立 abstract 给 BM25/dense 检索用，必须包含 archetype 关键词 + 题型特征>",
  "lexical_keywords": ["<6-10 个英文/中文关键词；必须包含 archetype 字面 token 与题型特征词>"],
  "check": "<一句话表达答案必须满足的硬约束/不变量，能用一行算式或断言判定。示例：probability in [0,1]; permutation count <= n!; geometry length > 0>",
  "trap": "<一句话指出训练失败案例中最常见的 1 个具体错误（从 evidence 反例提炼）。示例：assumed independence between two marked elements without conditioning on cycle membership>",
  "skip_if": "<一句话说明何时不应该用这张 card；例如题型不匹配、需要构造而非验证、需要详细证明等>",
  "trigger": {
    "text": "<什么 state 下应用本 card（一句话）>",
    "command_kinds": ["final"],
    "error_kinds": []
  }
}

要求：
- archetype 必须严格从给定 taxonomy 选；如确实无法判定再用 general-math（最后兜底）。
- check / trap / skip_if 各一句话，简洁可执行，不写散文段落。
- trap 必须从 evidence 反例提炼，不要凭空捏造。如果 evidence 没有反例，写 "no observed failure pattern in training"。
- 整张 card 渲染时控制在 150 token 内（abstract+check+trap+skip_if 加起来约 80-120 字）。
- 严禁包含 "procedure"、"essential_steps"、"worked example" 这些指导解题的字段。
- 只输出 JSON，无 markdown 包裹。"""


VERIFIER_CARD_USER_TEMPLATE = """归属 topic: {topic_id} ({topic_name})
related_intent_tags: {intent}
related_artifact_tags: {artifact}

== L2 policy（仅用于理解上下文，不要复述其 procedure）==
trigger.text: {trig}
verification.text: {verif}
scope.applies_to: {scope_a}
scope.does_not_apply_to: {scope_n}
anti_pattern.text: {ap}
anti_pattern.fail_signals: {ap_sig}

== 关联 topic L3 ==
common_pitfalls: {pitfalls}

== 训练集 evidence（正例 + 反例）==
{evidence}

请把这条 policy 蒸馏成一张 verifier card JSON（严格按 archetype taxonomy）。"""


SKILL_USER_TEMPLATE = """归属 topic: {topic_id} ({topic_name})
related_intent_tags: {intent}
related_artifact_tags: {artifact}

== L2 policy ==
trigger.text: {trig}
procedure.text: {proc}
verification.text: {verif}
scope.applies_to: {scope_a}
scope.does_not_apply_to: {scope_n}
anti_pattern.text: {ap}
anti_pattern.fail_signals: {ap_sig}

essential 候选 (from L2): {ess}
optional 候选: {opt}
redundant 候选: {red}

tool_effectiveness (本 cluster 内): {te}
expected_gain (V_with vs V_without): {eg}

== 关联 topic L3 ==
tool_atlas (本 topic 内): {atlas}
common_pitfalls (本 topic 已知坑):
{pitfalls}

== high-V evidence traces ==
{evidence}

请输出 Skill JSON。"""


def render_atlas(atlas: dict) -> str:
    if not atlas:
        return "(none)"
    return ", ".join(f"{k}({v['role']},V={v['v_avg_in_topic']:.2f})"
                       for k, v in sorted(atlas.items(),
                                          key=lambda kv: -kv[1]['v_avg_in_topic']))


def render_pitfalls(pf: list) -> str:
    return "\n".join(f"  - signal=\"{p.get('signal','')}\" remedy=\"{p.get('remedy','')}\""
                     for p in (pf or [])[:6]) or "(none)"


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


# ── worked-example extraction (training-free; reuses reflection_v2) ─────────


def _trace_to_worked_example(t: dict, polarity: str) -> dict:
    """Render one L1 trace as a worked example block (polarity in/out only)."""
    tc = t.get("tool_call", {}) or {}
    ob = t.get("observation", {}) or {}
    rf = t.get("reflection_v2", {}) or {}
    return {
        "polarity": polarity,
        "trace_id": t.get("trace_id"),
        "task_id": t.get("task_id"),
        "state": (t.get("state_summary") or "")[:400],
        "action": {
            "tool_kind": tc.get("command_kind", ""),
            "command_text": (tc.get("command_text") or "")[:400],
        },
        "observation": {
            "exit_code": ob.get("exit_code"),
            "error_kind": ob.get("error_kind"),
            "files_created": ob.get("files_created", []) or [],
            "key_signal": (rf.get("key_signal") or "")[:160],
        },
        "outcome": (rf.get("what_happened") or "")[:240],
        "why": ((rf.get("error_diagnosis") or rf.get("next_action_hint") or ""))[:240],
        "V": round(float(t.get("V", 0.0)), 3),
        "alpha": round(float(t.get("alpha", 0.0)), 3),
    }


def pick_worked_examples(positive_pool: list[dict],
                          negative_pool: list[dict],
                          k_pos: int = 2,
                          k_neg: int = 2) -> tuple[list[dict], list[dict]]:
    """Pick top-V positives + bottom-V/failing negatives. Both pools are trace dicts."""
    pos = sorted(
        [t for t in positive_pool if float(t.get("V", 0)) > 0],
        key=lambda t: -float(t.get("V", 0)),
    )[:k_pos]
    neg = sorted(
        negative_pool,
        key=lambda t: float(t.get("V", 0)),
    )[:k_neg]
    return ([_trace_to_worked_example(t, "positive") for t in pos],
            [_trace_to_worked_example(t, "negative") for t in neg])


def render_verifier_evidence(positive: list[dict], negative: list[dict],
                              k_pos: int = 3, k_neg: int = 3) -> str:
    """Compact evidence rendering for verifier card mode (focus on outcomes, not commands)."""
    out = []
    pos = sorted([t for t in positive if float(t.get("V", 0)) > 0],
                 key=lambda t: -float(t.get("V", 0)))[:k_pos]
    neg = sorted(negative or [], key=lambda t: float(t.get("V", 0)))[:k_neg]
    for t in pos:
        rf = t.get("reflection_v2", {}) or {}
        out.append(f"[+ V={float(t.get('V',0)):.2f}] state={t.get('state_summary','')[:200]} | outcome={rf.get('what_happened','')[:140]}")
    for t in neg:
        rf = t.get("reflection_v2", {}) or {}
        ob = t.get("observation", {}) or {}
        out.append(f"[- V={float(t.get('V',0)):.2f}] state={t.get('state_summary','')[:200]} | err={ob.get('error_kind','')} | why={(rf.get('error_diagnosis') or rf.get('next_action_hint') or '')[:160]}")
    return "\n".join(out) or "(no evidence)"


def crystallize_verifier_card(policy: dict, topic: dict, positive: list[dict],
                              negative: list[dict],
                              n_min: int = 2) -> dict | None:
    """MSCE-R: distill policy + evidence into a short verifier card.

    Uses LLM_MODEL_BIG to reduce archetype mis-classification risk.
    Returns None if the policy fails the 4-condition gate (same as procedural mode)
    or if archetype is missing in the LLM output.
    """
    n_sup = policy.get("n_support", 0)
    v_avg = policy.get("V_avg", 0.0)
    v_pos = policy.get("V_pos_count", 0)
    gain = (policy.get("expected_gain") or {}).get("gain", 0.0)
    if n_sup < n_min or v_avg <= 0 or (v_pos / max(n_sup, 1)) < 0.5 or gain <= 0:
        return None

    trig = policy.get("trigger") or {}
    verif = policy.get("verification") or {}
    scope = policy.get("scope") or {}
    ap = policy.get("anti_pattern") or {}

    def _text(x):
        if isinstance(x, dict):
            return str(x.get("text", ""))[:600]
        return str(x)[:600]

    user = VERIFIER_CARD_USER_TEMPLATE.format(
        topic_id=topic.get("topic_id", ""),
        topic_name=topic.get("topic_name", ""),
        intent=topic.get("related_intent_tags", []),
        artifact=topic.get("related_artifact_tags", []),
        trig=_text(trig),
        verif=_text(verif),
        scope_a=str(scope.get("applies_to", ""))[:200],
        scope_n=str(scope.get("does_not_apply_to", ""))[:200],
        ap=_text(ap),
        ap_sig=ap.get("fail_signals", []) if isinstance(ap, dict) else [],
        pitfalls=render_pitfalls(topic.get("common_pitfalls", [])),
        evidence=render_verifier_evidence(positive, negative),
    )
    raw = call_llm(VERIFIER_CARD_SYS, user, max_tokens=900,
                   model=LLM_MODEL_BIG)
    if not raw or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, str):
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
    elif isinstance(raw, dict):
        d = raw
    else:
        return None

    # Required: archetype + check
    if "archetype" not in d or "check" not in d:
        return None
    # Archetype validation
    VALID_ARCHETYPES = {
        "probability-process", "counting-enumeration", "permutation-cycle",
        "geometry-metric", "recurrence-expectation", "number-theory",
        "algebra-equation", "general-math",
    }
    if d["archetype"] not in VALID_ARCHETYPES:
        d["archetype"] = "general-math"

    # Reasoning verifier cards target only "final" command kind.
    d.setdefault("trigger", {"text": "single-turn reasoning final answer",
                              "command_kinds": ["final"], "error_kinds": []})

    # Carry training-set stats (used by V-calibrated gate at eval time).
    d["expected_gain"] = policy.get("expected_gain", {})
    d["source_policy"] = policy.get("policy_id")
    d["source_traces"] = policy.get("source_traces", [])[:8]
    d["topic_id"] = topic.get("topic_id")
    d["reliability"] = {
        "n_support": n_sup,
        "v_avg": round(v_avg, 3),
        "v_pos_rate": round(v_pos / max(n_sup, 1), 3),
    }
    # Applicability signature: use archetype itself as the primary intent tag,
    # so retrieval can match on archetype (after TaskProfiler also tags archetype).
    d["applicability_signature"] = {
        "intent_tags": [d["archetype"]],
        "artifact_tags": ["final-answer"],
        "command_kinds": ["final"],
    }
    if not d.get("lexical_keywords"):
        d["lexical_keywords"] = [d["archetype"], "verify", "reasoning"]
    if not d.get("abstract"):
        d["abstract"] = (f"[{d['archetype']}] check={d.get('check','')} "
                          f"trap={d.get('trap','')}")[:400]
    d["embedding_text"] = d["abstract"]
    # Mode marker so eval can use the verifier renderer.
    d["card_kind"] = "verifier"
    return d


def crystallize_one(policy: dict, topic: dict, evidence: list[dict],
                    n_min: int = 2,
                    negative_candidates: list[dict] | None = None) -> dict | None:
    n_sup = policy.get("n_support", 0)
    v_avg = policy.get("V_avg", 0.0)
    v_pos = policy.get("V_pos_count", 0)
    gain = (policy.get("expected_gain") or {}).get("gain", 0.0)
    if n_sup < n_min or v_avg <= 0 or (v_pos / max(n_sup, 1)) < 0.5 or gain <= 0:
        return None

    trig = policy.get("trigger") or {}
    proc = policy.get("procedure") or {}
    verif = policy.get("verification") or {}
    scope = policy.get("scope") or {}
    ap = policy.get("anti_pattern") or {}

    def _text(x):
        if isinstance(x, dict):
            return str(x.get("text", ""))[:600]
        return str(x)[:600]

    user = SKILL_USER_TEMPLATE.format(
        topic_id=topic.get("topic_id", ""),
        topic_name=topic.get("topic_name", ""),
        intent=topic.get("related_intent_tags", []),
        artifact=topic.get("related_artifact_tags", []),
        trig=_text(trig),
        proc=_text(proc),
        verif=_text(verif),
        scope_a=str(scope.get("applies_to", ""))[:200],
        scope_n=str(scope.get("does_not_apply_to", ""))[:200],
        ap=_text(ap),
        ap_sig=ap.get("fail_signals", []) if isinstance(ap, dict) else [],
        ess=proc.get("essential_steps", []) if isinstance(proc, dict) else [],
        opt=proc.get("optional_steps", []) if isinstance(proc, dict) else [],
        red=proc.get("redundant_steps", []) if isinstance(proc, dict) else [],
        te=policy.get("tool_effectiveness", {}),
        eg=policy.get("expected_gain", {}),
        atlas=render_atlas(topic.get("tool_atlas", {})),
        pitfalls=render_pitfalls(topic.get("common_pitfalls", [])),
        evidence=render_evidence(evidence),
    )
    raw = call_llm(SKILL_SYS, user, max_tokens=1500)
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
    if "trigger" not in d or "procedure" not in d:
        return None

    # Attach NAT-style worked examples (✅ pos / ❌ neg), zero extra LLM calls
    pos_ex, neg_ex = pick_worked_examples(
        positive_pool=evidence,
        negative_pool=negative_candidates or [],
        k_pos=2, k_neg=2,
    )
    d["positive_examples"] = pos_ex
    d["negative_examples"] = neg_ex

    # Compute applicability_signature + command_kinds
    cmd_kinds = set()
    for s in (d.get("procedure", {}).get("essential_steps") or []):
        if isinstance(s, dict) and s.get("tool_kind"):
            cmd_kinds.add(s["tool_kind"])
    if not cmd_kinds:
        # fall back to top primary atlas
        for k, st in (topic.get("tool_atlas") or {}).items():
            if st.get("role") == "primary":
                cmd_kinds.add(k)

    intent_tags = list(topic.get("related_intent_tags", [])) or list(policy.get("intent_tags", []))
    artifact_tags = list(topic.get("related_artifact_tags", [])) or list(policy.get("artifact_tags", []))
    d["applicability_signature"] = {
        "intent_tags": intent_tags,
        "artifact_tags": artifact_tags,
        "command_kinds": sorted(cmd_kinds),
    }
    d["topic_id"] = topic.get("topic_id")
    d["tool_effectiveness"] = policy.get("tool_effectiveness", {})
    d["expected_gain"] = policy.get("expected_gain", {})
    d["source_policy"] = policy.get("policy_id")
    d["source_traces"] = policy.get("source_traces", [])[:8]
    d["reliability"] = {
        "n_support": n_sup,
        "v_avg": round(v_avg, 3),
        "v_pos_rate": round(v_pos / max(n_sup, 1), 3),
    }
    # Ensure abstract + lexical_keywords (LLM may forget)
    if not d.get("abstract"):
        def _trig_text(x):
            return x.get("text", "") if isinstance(x, dict) else str(x)
        d["abstract"] = (
            f"{d.get('summary','')[:120]} Trigger: {_trig_text(d.get('trigger'))[:200]} "
            f"Procedure: {_trig_text(d.get('procedure'))[:200]}"
        )[:600]
    if not d.get("lexical_keywords"):
        kws = set()
        for t in intent_tags + artifact_tags + sorted(cmd_kinds):
            if t:
                kws.add(str(t).lower())
        if isinstance(d.get("trigger"), dict):
            for k in d["trigger"].get("error_kinds", []) or []:
                kws.add(str(k).lower())
        d["lexical_keywords"] = sorted(kws)[:20]
    # Used by dense retrieval (kept for backward compat with v2 eval)
    d["embedding_text"] = d["abstract"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--topics", required=True, help="l3_topics_v3.jsonl")
    ap.add_argument("--l1-traces", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--mode", choices=["procedural", "verifier_card"],
                    default="procedural",
                    help="procedural=default solver skill; verifier_card=MSCE-R Reasoning mode")
    args = ap.parse_args()

    policies = [json.loads(l) for l in open(args.policies)]
    print(f"Loaded {len(policies)} L2 policies")
    pol_index = {p["policy_id"]: p for p in policies}

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

    skills_out = []
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
            if args.mode == "verifier_card":
                futs[ex.submit(crystallize_verifier_card, p, topic,
                                positive_pool[:6], neg_pool[:6], args.n_min)] = p["policy_id"]
            else:
                futs[ex.submit(crystallize_one, p, topic,
                                positive_pool[:6], args.n_min, neg_pool[:6])] = p["policy_id"]
        i = 0
        prefix = "msce_v5" if args.mode == "verifier_card" else "msce_v3"
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
            skills_out.append(skill["skill_id"])
            if i % 5 == 0:
                print(f"  crystallized {i}/{len(candidates)}")

    # Patch topics with related_skills (so eval can route topic → skills)
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

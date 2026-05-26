#!/usr/bin/env python3
"""MSCE self-evolution loop — Read-Execute-Reflect-Write.

Given a finished eval job, this module:

  1. READ failed tasks + the skills that were injected for them.
  2. EXECUTE LLM diagnosis: "Did the injected skill help? If not, what is the
     diagnosis (wrong trigger / wrong procedure / missing anti-pattern / etc.)
     and what mutation should we apply?".
  3. REFLECT-WRITE: apply mutations to the skill bank, producing a new
     skill_bank_v3evo.jsonl. Also append new exploratory skills for tasks
     that had no matching skill (SkillX-style coverage expansion).

Input:
    --skill-bank   existing skill_bank.jsonl (v3 / v3.1)
    --topics       l3_topics_v3.jsonl
    --job          finished eval job dir (has v3_inject_log.jsonl + result.json per task)
    --task-prompts task_summaries_v3.jsonl from extract_memory_v3 (optional, for richer context)
Output:
    --output       new skill_bank_v3evo.jsonl (mutated bank)
    --log          self_evolve_log.jsonl (one record per mutation)
"""
from __future__ import annotations
import argparse, json, os, sys, time, glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .clients import chat_completion_json
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import chat_completion_json


def call_llm(system: str, user: str, max_tokens: int = 1200,
             retries: int = 3, temperature: float = 0.0):
    try:
        return chat_completion_json(
            system=system,
            user=user,
            max_tokens=max_tokens,
            retries=retries,
            temperature=temperature,
            timeout=120,
        )
    except Exception:
        return {}


DIAGNOSE_SYS = """你是 skill 演化诊断师。给你一条 skill 和一个失败的 test task（含 prompt、injected skill names、agent 输出/失败信号），判断本 skill 在该 task 里如何失效，并给出明确的修补建议。

输出严格 JSON：

{
  "diagnosis": "<failed-reason 中文，1-2 句>",
  "mutation_type": "wrong_trigger | wrong_procedure | missing_anti_pattern | missing_essential_step | inject_should_skip | other",
  "patch": {
    "trigger_addition":          "<可选: 给 trigger 新增/纠正的描述>",
    "procedure_amendment":       "<可选: procedure 要增加/删除的步骤>",
    "new_anti_pattern":          "<可选: 要写入 anti_pattern 的新条目>",
    "new_essential_step":        "<可选: 要加入 essential_steps 的工具步骤>",
    "scope_does_not_apply_to":   "<可选: 要追加到 scope.does_not_apply_to 的描述>"
  },
  "confidence": <float 0~1>
}

要求：
- 优先采用最小修补：只填该 mutation_type 对应的 patch 字段，其他留空字符串
- patch 必须可被复用到其他相似 task；不要写"这个 task 特定"的修补
- inject_should_skip 表示本 skill 永远不该用在该 task 类型 → 此时只填 scope_does_not_apply_to
- 只输出 JSON
"""


def diagnose_one(skill: dict, fail_record: dict) -> dict:
    user = (
        f"== Failed Task ==\n"
        f"task_id: {fail_record.get('task_id')}\n"
        f"intent_tags: {fail_record.get('intent_tags')}\n"
        f"artifact_tags: {fail_record.get('artifact_tags')}\n"
        f"task_prompt_excerpt: {fail_record.get('task_prompt_excerpt', '')[:1500]}\n"
        f"agent_final_answer: {fail_record.get('agent_final_answer', '')[:800]}\n"
        f"verifier_signal: {fail_record.get('verifier_signal', '')[:400]}\n\n"
        f"== Injected Skill ==\n"
        f"name: {skill.get('name')}\n"
        f"abstract: {skill.get('abstract','')[:400]}\n"
        f"trigger.text: {(skill.get('trigger') or {}).get('text','')[:400]}\n"
        f"procedure.text: {(skill.get('procedure') or {}).get('text','')[:400]}\n"
        f"anti_pattern.text: {(skill.get('anti_pattern') or {}).get('text','')[:300]}\n"
        f"scope.applies_to: {(skill.get('scope') or {}).get('applies_to','')[:200]}\n"
        f"scope.does_not_apply_to: {(skill.get('scope') or {}).get('does_not_apply_to','')[:200]}\n"
    )
    res = call_llm(DIAGNOSE_SYS, user, max_tokens=800)
    if not isinstance(res, dict):
        return {"diagnosis": "(parse failed)", "mutation_type": "other",
                "patch": {}, "confidence": 0.0}
    return res


# ── apply mutation to a skill in-place ────────────────────────────


def _append_text(orig: str, addition: str) -> str:
    orig = (orig or "").strip()
    addition = (addition or "").strip()
    if not addition:
        return orig
    if addition in orig:
        return orig
    sep = "\n" if orig else ""
    return (orig + sep + addition).strip()


def apply_mutation(skill: dict, patch: dict, mut_type: str) -> bool:
    """Mutate skill in-place. Returns True if any field changed."""
    changed = False
    patch = patch or {}

    trig_add = patch.get("trigger_addition", "").strip()
    if trig_add and mut_type in ("wrong_trigger", "other"):
        trig = skill.setdefault("trigger", {})
        if isinstance(trig, dict):
            trig["text"] = _append_text(trig.get("text", ""), trig_add)
            changed = True

    proc_amend = patch.get("procedure_amendment", "").strip()
    if proc_amend and mut_type in ("wrong_procedure", "missing_essential_step", "other"):
        proc = skill.setdefault("procedure", {})
        if isinstance(proc, dict):
            proc["text"] = _append_text(proc.get("text", ""), proc_amend)
            changed = True

    new_ap = patch.get("new_anti_pattern", "").strip()
    if new_ap and mut_type in ("missing_anti_pattern", "other"):
        ap = skill.setdefault("anti_pattern", {})
        if isinstance(ap, dict):
            ap["text"] = _append_text(ap.get("text", ""), new_ap)
            changed = True

    new_ess = patch.get("new_essential_step", "").strip()
    if new_ess and mut_type in ("missing_essential_step", "other"):
        proc = skill.setdefault("procedure", {})
        if isinstance(proc, dict):
            ess = proc.setdefault("essential_steps", [])
            if not isinstance(ess, list):
                ess = []
                proc["essential_steps"] = ess
            ess.append({"tool_kind": "auto", "purpose": new_ess,
                         "must_succeed": True, "from": "self_evolve"})
            changed = True

    scope_dna = patch.get("scope_does_not_apply_to", "").strip()
    if scope_dna and mut_type in ("inject_should_skip", "other"):
        scope = skill.setdefault("scope", {})
        if isinstance(scope, dict):
            scope["does_not_apply_to"] = _append_text(
                scope.get("does_not_apply_to", ""), scope_dna)
            changed = True

    if changed:
        h = skill.setdefault("evolve_history", [])
        h.append({
            "ts": int(time.time()),
            "mutation_type": mut_type,
            "diagnosis_excerpt": patch.get("_diagnosis", "")[:240],
            "patch_keys": [k for k, v in patch.items() if v and not k.startswith("_")],
        })
        # bump expected_gain.n_neg to reflect observed failure
        eg = skill.setdefault("expected_gain", {})
        eg["n_neg_evolve"] = int(eg.get("n_neg_evolve", 0)) + 1
    return changed


# ── failure case extraction ───────────────────────────────────────


def extract_failed_cases(job_dir: Path, inject_log: list[dict],
                          max_cases: int = 60) -> list[dict]:
    """Return list of (task_id, intent_tags, artifact_tags, chosen_skill_ids,
    task_prompt_excerpt, agent_final_answer, verifier_signal)."""
    by_tid = {r["task_id"]: r for r in inject_log}
    cases = []
    for d in sorted(job_dir.glob("*__trial_*")):
        rf = d / "result.json"
        if not rf.exists():
            continue
        try:
            r = json.load(open(rf))
        except Exception:
            continue
        vr = r.get("verifier_result", {})
        if float(vr.get("reward", 0)) > 0:
            continue
        # extract task prompt excerpt + agent final answer from session.jsonl
        task_prompt = ""
        agent_final = ""
        try:
            with open(d / "session.jsonl") as f:
                events = [json.loads(l) for l in f]
            for e in events[:6]:
                m = e.get("message", {})
                if e.get("type") == "message" and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, list):
                        txt = "\n".join(cc.get("text", "") for cc in c
                                         if isinstance(cc, dict) and cc.get("type") == "text")
                    else:
                        txt = str(c)
                    task_prompt = txt[:3000]
                    break
            for e in reversed(events):
                m = e.get("message", {})
                if e.get("type") == "message" and m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, list):
                        txt = "\n".join(cc.get("text", "") for cc in c
                                         if isinstance(cc, dict) and cc.get("type") == "text")
                    else:
                        txt = str(c)
                    agent_final = txt[:2000]
                    break
        except Exception:
            pass

        # extract task_id (strip trial suffix)
        tid = d.name.split("__")[0]
        rec = by_tid.get(tid, {})
        cases.append({
            "task_id": tid,
            "intent_tags": rec.get("intent_tags", []),
            "artifact_tags": rec.get("artifact_tags", []),
            "chosen_skills": rec.get("chosen_skills", []),
            "task_prompt_excerpt": task_prompt,
            "agent_final_answer": agent_final,
            "verifier_signal": json.dumps(vr, ensure_ascii=False)[:600],
        })
        if len(cases) >= max_cases:
            break
    return cases


# ── main ──────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-bank", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--job", required=True, help="finished eval job dir")
    ap.add_argument("--output", required=True, help="mutated skill bank")
    ap.add_argument("--log", default=None, help="per-mutation log (jsonl)")
    ap.add_argument("--max-cases", type=int, default=60)
    ap.add_argument("--parallel", type=int, default=3)
    args = ap.parse_args()

    skills = [json.loads(l) for l in open(args.skill_bank)]
    skill_index = {s["skill_id"]: s for s in skills}
    print(f"Loaded {len(skills)} skills from {args.skill_bank}")

    job_dir = Path(args.job)
    inject_log_path = job_dir / "v3_inject_log.jsonl"
    if not inject_log_path.exists():
        print(f"ERROR: {inject_log_path} not found", file=sys.stderr)
        sys.exit(2)
    inject_log = [json.loads(l) for l in open(inject_log_path)]
    print(f"Loaded inject log: {len(inject_log)} rows")

    failed_cases = extract_failed_cases(job_dir, inject_log, args.max_cases)
    print(f"Failed cases (with injected skill): {len(failed_cases)}")

    # Build (skill, case) pairs to diagnose. For each failed case, diagnose
    # each injected skill independently. Cap total LLM calls.
    pairs = []
    for c in failed_cases:
        for sid in c.get("chosen_skills", []):
            s = skill_index.get(sid)
            if s:
                pairs.append((s, c))
    print(f"Diagnose pairs: {len(pairs)}")

    log_path = args.log or (str(Path(args.output).with_suffix("")) + "_evolve_log.jsonl")
    mutations = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex, \
            open(log_path, "w") as flog:
        futs = {ex.submit(diagnose_one, s, c): (s, c) for s, c in pairs}
        for f in as_completed(futs):
            s, c = futs[f]
            try:
                res = f.result()
            except Exception as e:
                continue
            mut_type = res.get("mutation_type", "other")
            patch = res.get("patch") or {}
            patch["_diagnosis"] = res.get("diagnosis", "")
            conf = float(res.get("confidence", 0) or 0)
            if conf < 0.35:
                continue
            changed = apply_mutation(s, patch, mut_type)
            if changed:
                mutations += 1
                flog.write(json.dumps({
                    "skill_id": s["skill_id"],
                    "task_id": c["task_id"],
                    "mutation_type": mut_type,
                    "confidence": conf,
                    "diagnosis": res.get("diagnosis", "")[:400],
                    "patch_keys": [k for k, v in patch.items()
                                    if v and not k.startswith("_")],
                }, ensure_ascii=False) + "\n")
            if mutations % 10 == 0 and mutations:
                print(f"  mutations={mutations}")

    # Write mutated bank
    with open(args.output, "w") as fo:
        for s in skills:
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(skills)} skills (with {mutations} mutations) to {args.output}")
    print(f"Mutation log: {log_path}")


if __name__ == "__main__":
    sys.exit(main())

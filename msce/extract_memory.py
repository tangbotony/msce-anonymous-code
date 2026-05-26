#!/usr/bin/env python3
"""MSCE — Step 1: structured L1 trace extraction with posthoc reflection.

Implementation notes:
1. Parse tool_call into structured {name, command_kind, command_text, args_raw}.
2. Parse observation into structured {raw, exit_code, error_kind, files_created, key_signal}.
3. Generate posthoc reflection_v2 via LLM when original sessions have sparse self-reflection.
4. Score alpha based on reflection_v2 and local context.
5. Estimate terminal feedback from the full verifier_result and task trace summary.

Input:
    --session-dir   train job dir with <task>__trial_N/session.jsonl + result.json
Output:
    --output-dir/
        l1_traces.jsonl      one record per step, with structured fields
        task_summaries.jsonl one record per task: passed, r_human, n_steps, intent_tags, artifact_tags
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .clients import chat_completion_json
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import chat_completion_json

GAMMA = 0.9

def call_llm(system: str, user: str, max_tokens: int = 600, retries: int = 3,
             temperature: float = 0.0) -> dict | str:
    """Return parsed JSON dict or raw text on failure to parse."""
    try:
        return chat_completion_json(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            timeout=90,
        )
    except Exception:
        return {}
    return {}


# ── Tool call kind classifier (regex-based, fast, no LLM) ───────────

_CMD_KIND_RULES = [
    ("python_eval", re.compile(r"^(python3?)(\s+-c|\s+-m|\s+<<|\s+/|\s+\S+\.py)", re.I)),
    ("pip", re.compile(r"^(pip3?|uv pip|conda)\s+(install|uninstall|list|show)", re.I)),
    ("pytest", re.compile(r"^(pytest|python3? -m pytest)", re.I)),
    ("git", re.compile(r"^git\s+", re.I)),
    ("docker", re.compile(r"^docker\s+", re.I)),
    ("bash_ls", re.compile(r"^(ls|tree|find)\b", re.I)),
    ("bash_cat", re.compile(r"^(cat|head|tail|less)\b", re.I)),
    ("bash_mkdir", re.compile(r"^(mkdir|touch)\b", re.I)),
    ("bash_cp", re.compile(r"^(cp|mv|rsync|scp)\b", re.I)),
    ("bash_rm", re.compile(r"^rm\b", re.I)),
    ("bash_grep", re.compile(r"^(grep|rg|ag)\b", re.I)),
    ("bash_sed_awk", re.compile(r"^(sed|awk)\b", re.I)),
    ("bash_curl", re.compile(r"^(curl|wget|httpie?)\b", re.I)),
    ("bash_zip", re.compile(r"^(zip|unzip|tar)\b", re.I)),
    ("node", re.compile(r"^(node|npm|npx|yarn|pnpm)\s+", re.I)),
]

_ERROR_RULES = [
    ("CommandNotFound", re.compile(r"command not found|No such file or directory.*\b(bash|sh)\b", re.I)),
    ("FileNotFoundError", re.compile(r"FileNotFoundError|No such file or directory", re.I)),
    ("PermissionDenied", re.compile(r"Permission denied|access denied", re.I)),
    ("ImportError", re.compile(r"(ImportError|ModuleNotFoundError):", re.I)),
    ("SyntaxError", re.compile(r"SyntaxError", re.I)),
    ("TypeError", re.compile(r"TypeError", re.I)),
    ("KeyError", re.compile(r"KeyError", re.I)),
    ("IndexError", re.compile(r"IndexError", re.I)),
    ("ValueError", re.compile(r"ValueError", re.I)),
    ("AttributeError", re.compile(r"AttributeError", re.I)),
    ("HTTPError", re.compile(r"HTTPError|status.*40\d|50\d", re.I)),
    ("Timeout", re.compile(r"timed out|TimeoutError|signal 9", re.I)),
    ("Traceback", re.compile(r"Traceback \(most recent call last\)", re.I)),
]

_FILE_CREATE_RULES = [
    # things that often print "Wrote ..." or "Saved to ..."
    re.compile(r"(?:Wrote|Saved|Created|Output|Generated)[\s:]+([^\s]+\.(?:docx|xlsx|pdf|png|jpg|csv|json|md|sol|sh|zip|tar|gz))",
               re.I),
    re.compile(r"^([^\s]+\.(?:docx|xlsx|pdf|png|csv|zip))\s*$", re.M),
]


def classify_command_kind(command_text: str) -> str:
    cmd = (command_text or "").strip().lstrip("$").strip()
    for kind, rx in _CMD_KIND_RULES:
        if rx.search(cmd[:120]):
            return kind
    if not cmd:
        return "empty"
    return "other"


def classify_error_kind(obs: str) -> str | None:
    if not obs:
        return None
    text = obs[-3000:]  # only inspect tail (more recent error)
    for name, rx in _ERROR_RULES:
        if rx.search(text):
            return name
    return None


def detect_files_created(obs: str) -> list[str]:
    if not obs:
        return []
    found = []
    for rx in _FILE_CREATE_RULES:
        for m in rx.findall(obs):
            if isinstance(m, tuple):
                m = m[0]
            if m and m not in found and len(found) < 8:
                found.append(m)
    return found


def detect_exit_code(obs: str) -> int | None:
    # OpenClaw exec often prints '[exit_code: N]' or 'returncode: N'
    if not obs:
        return None
    m = re.search(r"(?:exit_code|returncode|exit\s+code)\s*[:=]\s*(-?\d+)", obs, re.I)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # heuristic: presence of Traceback / "Error" without exit_code → 1
    if re.search(r"Traceback \(most recent call last\)", obs):
        return 1
    return None


# ── Session.jsonl parser → structured steps ────────────────────────


def parse_session(session_file: Path) -> list[dict]:
    """Parse OpenClaw session.jsonl into structured step records.

    Each step:
      {state_summary, tool_call: {name, command_text, args_raw, command_kind},
       observation: {raw, exit_code, error_kind, files_created, key_signal},
       assistant_text: str (optional, pre-tool text from assistant)}
    """
    try:
        events = []
        with open(session_file) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []

    steps = []
    state_summary = ""
    pending = None  # action awaiting observation

    for e in events:
        if e.get("type") != "message":
            continue
        m = e.get("message", {})
        role = m.get("role")
        c = m.get("content")
        text_parts = []
        tool_calls = []
        tool_results = []
        if isinstance(c, list):
            for item in c:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text":
                    text_parts.append(item.get("text", ""))
                elif t == "toolCall":
                    tool_calls.append(item)
                elif t == "toolResult":
                    tool_results.append(item)
        elif isinstance(c, str):
            text_parts.append(c)

        text = "\n".join(text_parts).strip()

        if role == "user" and pending is None:
            # initial query becomes the seed state
            state_summary = text[:1200]
        elif role == "assistant":
            assistant_text = text[:1500]
            for tc in tool_calls:
                args = tc.get("arguments") or tc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                command_text = ""
                if isinstance(args, dict):
                    command_text = args.get("command") or args.get("cmd") or args.get("script") or ""
                    if not command_text:
                        # fall back to first string arg
                        for v in args.values():
                            if isinstance(v, str) and v:
                                command_text = v
                                break
                command_kind = classify_command_kind(command_text)
                pending = {
                    "state_summary": state_summary[:1200],
                    "tool_call": {
                        "name": tc.get("name") or "exec",
                        "command_kind": command_kind,
                        "command_text": (command_text or "")[:1500],
                        "args_raw": json.dumps(args, ensure_ascii=False)[:1500],
                    },
                    "assistant_text": assistant_text,
                }
            if not tool_calls and assistant_text:
                # final answer w/o tool call
                if pending is not None:
                    steps.append({**pending, "observation": _obs_record(""), "assistant_text_final": assistant_text})
                    pending = None
                else:
                    steps.append({
                        "state_summary": state_summary[:1200],
                        "tool_call": {"name": "final_answer", "command_kind": "final",
                                       "command_text": "", "args_raw": ""},
                        "observation": _obs_record(""),
                        "assistant_text_final": assistant_text,
                    })
        elif role in ("tool", "toolResult"):
            obs_raw = text
            if pending is not None:
                steps.append({**pending, "observation": _obs_record(obs_raw)})
                state_summary = obs_raw[-1200:]
                pending = None

    if pending is not None:
        steps.append({**pending, "observation": _obs_record("")})

    return steps


def _obs_record(raw: str) -> dict:
    return {
        "raw_truncated": (raw or "")[:1500],
        "exit_code": detect_exit_code(raw or ""),
        "error_kind": classify_error_kind(raw or ""),
        "files_created": detect_files_created(raw or ""),
        "key_signal": "",   # filled by posthoc reflection
    }


# ── Posthoc reflection LLM ──────────────────────────────────────────

POSTHOC_SYS = """你是一个 agent step 反思器。给你 1 步执行的 state/工具调用/观测，输出 JSON 反思：

{
  "what_happened": "<本步发生了什么，1 句话>",
  "is_progress": true|false,
  "is_blocker": true|false,
  "error_diagnosis": "<如失败，根因；成功则空字符串>",
  "next_action_hint": "<对下一步的具体建议；如该步成功则空>",
  "key_signal": "<最有价值的一个观察信号，例如 'docx 已生成', 'python 命令不存在'>"
}

要求：
- is_progress：本步在任务向前推进上的贡献（产生关键观察、生成文件、修复错误、定位 bug 等）
- is_blocker：本步是阻塞性失败，需要换路（例如 CommandNotFound / Permission / ImportError）
- error_diagnosis：诊断根因，不要复述错误文本
- 只输出 JSON
"""


def posthoc_reflect(step: dict) -> dict:
    tc = step.get("tool_call", {})
    ob = step.get("observation", {})
    if tc.get("command_kind") == "final":
        return {
            "what_happened": "Agent gave final answer.",
            "is_progress": True,
            "is_blocker": False,
            "error_diagnosis": "",
            "next_action_hint": "",
            "key_signal": "final_answer",
        }
    state = step.get("state_summary", "")[:400]
    cmd = tc.get("command_text", "")[:400]
    cmd_kind = tc.get("command_kind", "")
    obs_raw = ob.get("raw_truncated", "")[:600]
    exit_code = ob.get("exit_code")
    error_kind = ob.get("error_kind")
    files_created = ob.get("files_created")

    user = (
        f"state: {state}\n"
        f"tool: {tc.get('name')} | command_kind: {cmd_kind}\n"
        f"command: {cmd}\n"
        f"exit_code: {exit_code} | error_kind: {error_kind} | files_created: {files_created}\n"
        f"observation:\n{obs_raw}"
    )
    res = call_llm(POSTHOC_SYS, user, max_tokens=400, temperature=0.0)
    if not isinstance(res, dict):
        # parse failed
        return {
            "what_happened": "(unparsed)",
            "is_progress": exit_code == 0 or files_created,
            "is_blocker": error_kind is not None,
            "error_diagnosis": error_kind or "",
            "next_action_hint": "",
            "key_signal": "",
        }
    # Coerce
    return {
        "what_happened": str(res.get("what_happened", ""))[:300],
        "is_progress": bool(res.get("is_progress", False)),
        "is_blocker": bool(res.get("is_blocker", False)),
        "error_diagnosis": str(res.get("error_diagnosis", ""))[:300],
        "next_action_hint": str(res.get("next_action_hint", ""))[:300],
        "key_signal": str(res.get("key_signal", ""))[:200],
    }


# ── R_human task-level scoring ─────────────────────────────────────

R_HUMAN_SYS = """你是任务级评分员，给一次 agent 完整 trace 打分。

输出 JSON: {"r_human": <float -1 to +1>, "intent_tags": ["..."], "artifact_tags": ["..."]}

rubric:
- 目标达成度 (-1 to +1): 任务核心需求是否被满足 (verifier passed/failed 是主信号)
- 过程质量 (-0.5 to +0.5): 是否走弯路、是否触发关键错误后未修复
- 用户满意信号 (-0.5 to +0.5): 终态 artifact 是否完整

intent_tags: 3-6 个，描述任务意图（如 office-doc-gen, blockchain-dapp, data-analysis-xlsx, retail-broker, salescon, web-research, code-impl-algorithm, info-retrieval-multi-hop, sysadmin-config 等）

artifact_tags: 3-6 个，描述期望产物（如 docx, xlsx, pdf, zip, code_repo, search_answer, sql_query, advisory_doc, markdown_report 等）

只输出 JSON。"""


def score_r_human(task_id: str, passed: bool, verifier_details: dict,
                   task_prompt: str, last_assistant: str,
                   steps_summary: str) -> dict:
    user = (
        f"task_id: {task_id}\n"
        f"verifier_passed: {passed}\n"
        f"verifier_details: {json.dumps(verifier_details, ensure_ascii=False)[:1000]}\n"
        f"---\n"
        f"task_prompt: {task_prompt[:2500]}\n"
        f"---\n"
        f"last_assistant: {last_assistant[:1500]}\n"
        f"---\n"
        f"trace_summary:\n{steps_summary[:3000]}"
    )
    res = call_llm(R_HUMAN_SYS, user, max_tokens=400, temperature=0.0)
    fallback_r = 0.7 if passed else -0.5
    if not isinstance(res, dict):
        return {"r_human": fallback_r, "intent_tags": [], "artifact_tags": []}
    r = res.get("r_human")
    if r is None:
        r = fallback_r
    return {
        "r_human": max(-1.0, min(1.0, float(r))),
        "intent_tags": [str(x)[:60] for x in (res.get("intent_tags") or [])][:8],
        "artifact_tags": [str(x)[:30] for x in (res.get("artifact_tags") or [])][:8],
    }


# ── alpha scoring (using reflection_v2) ─────────────────────────────


ALPHA_SYS = """你是 step-level 反思质量评分员。给你一步执行的工具/观测/posthoc reflection，输出 JSON：

{"alpha": <float 0 to 1>}

alpha:
- 0.7~0.9: 该步识别了关键信息（定位根因/发现正确路径/生成关键 artifact）
- 0.4~0.6: 正常推进（执行成功但只是常规步骤）
- 0.1~0.3: 探索性或低信息步骤（grep/ls 没发现什么）
- 0.0:    重复无效或纯噪音

只输出 JSON。"""


def score_alpha(step: dict) -> float:
    refl = step.get("reflection_v2", {})
    ob = step.get("observation", {})
    if step.get("tool_call", {}).get("command_kind") == "final":
        return 0.8  # final answer step
    is_blocker = refl.get("is_blocker", False)
    is_progress = refl.get("is_progress", False)
    error_kind = ob.get("error_kind")
    # Rule-based prior
    if is_progress and not is_blocker:
        prior = 0.6
    elif is_blocker:
        prior = 0.3  # blocker steps still informative (avoid this branch)
    elif error_kind:
        prior = 0.2
    else:
        prior = 0.4
    # LLM refine
    user = (
        f"command_kind: {step['tool_call'].get('command_kind')}\n"
        f"exit_code: {ob.get('exit_code')} | error_kind: {error_kind}\n"
        f"files_created: {ob.get('files_created')}\n"
        f"key_signal: {refl.get('key_signal')}\n"
        f"diagnosis: {refl.get('error_diagnosis')}\n"
        f"what_happened: {refl.get('what_happened')}"
    )
    res = call_llm(ALPHA_SYS, user, max_tokens=40, temperature=0.0)
    if isinstance(res, dict):
        a = res.get("alpha")
        if a is not None:
            try:
                return max(0.0, min(1.0, float(a)))
            except Exception:
                pass
    return prior


# ── Value backfill ─────────────────────────────────────────────────


def backfill_values(steps: list[dict], r_human: float, parallel: int = 4) -> list[dict]:
    T = len(steps)
    if T == 0:
        return []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        alphas = list(ex.map(score_alpha, steps))
    V = [0.0] * T
    V[T - 1] = r_human
    for t in range(T - 2, -1, -1):
        a_t = alphas[t]
        V[t] = a_t * r_human + (1 - a_t) * GAMMA * V[t + 1]
    for i, s in enumerate(steps):
        s["alpha"] = alphas[i]
        s["V"] = V[i]
    return steps


# ── Task-level processing ─────────────────────────────────────────


def load_result(task_dir: Path) -> dict:
    f = task_dir / "result.json"
    if not f.exists():
        return {"passed": False, "verifier_details": {}}
    try:
        d = json.load(open(f))
        vr = d.get("verifier_result", {})
        return {
            "passed": float(vr.get("reward", 0)) > 0,
            "reward": float(vr.get("reward", 0)),
            "verifier_details": vr,
        }
    except Exception:
        return {"passed": False, "verifier_details": {}}


def process_task(task_dir: Path, reflection_parallel: int = 4) -> dict | None:
    sess = task_dir / "session.jsonl"
    if not sess.exists():
        return None
    res = load_result(task_dir)
    steps = parse_session(sess)
    if not steps:
        return None

    # 1) posthoc reflection in parallel
    with ThreadPoolExecutor(max_workers=reflection_parallel) as ex:
        refls = list(ex.map(posthoc_reflect, steps))
    for s, r in zip(steps, refls):
        s["reflection_v2"] = r

    # Extract task prompt and last assistant
    task_prompt = steps[0]["state_summary"] if steps else ""
    last_assistant = ""
    for s in reversed(steps):
        if s.get("assistant_text_final"):
            last_assistant = s["assistant_text_final"]
            break

    # build short trace summary for R_human scoring
    lines = []
    n_show = min(6, len(steps))
    head = steps[: max(1, n_show // 2)]
    tail = steps[-(n_show - len(head)):] if n_show > len(head) else []
    for s in head + tail:
        lines.append(f"  [{s['tool_call']['command_kind']}] cmd={s['tool_call']['command_text'][:120]} "
                     f"→ exit={s['observation']['exit_code']} err={s['observation']['error_kind']} "
                     f"sig={s['reflection_v2'].get('key_signal','')[:80]}")
    steps_summary = "\n".join(lines)

    # 2) R_human + intent/artifact tags
    rh = score_r_human(task_dir.name, res["passed"], res["verifier_details"],
                        task_prompt, last_assistant, steps_summary)
    r_human = rh["r_human"]

    # 3) Backfill V values via reflection-weighted backprop
    steps = backfill_values(steps, r_human, parallel=reflection_parallel)

    return {
        "task_id": task_dir.name,
        "passed": res["passed"],
        "reward": res.get("reward", 0.0),
        "r_human": r_human,
        "intent_tags": rh["intent_tags"],
        "artifact_tags": rh["artifact_tags"],
        "n_steps": len(steps),
        "verifier_details": res["verifier_details"],
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--parallel-tasks", type=int, default=4)
    ap.add_argument("--parallel-reflection", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=0)
    args = ap.parse_args()

    sd = Path(args.session_dir)
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted([d for d in sd.iterdir() if d.is_dir() and (d / "session.jsonl").exists()])
    if args.max_tasks > 0:
        task_dirs = task_dirs[: args.max_tasks]
    print(f"Found {len(task_dirs)} train tasks with session.jsonl")

    l1p = od / "l1_traces.jsonl"
    sp = od / "task_summaries.jsonl"
    done = 0
    with l1p.open("w") as fl, sp.open("w") as fs, \
            ThreadPoolExecutor(max_workers=args.parallel_tasks) as pool:
        futs = {pool.submit(process_task, td, args.parallel_reflection): td for td in task_dirs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  ERR on {futs[fut].name}: {e}", file=sys.stderr)
                continue
            if rec is None:
                continue
            fs.write(json.dumps({
                "task_id": rec["task_id"],
                "passed": rec["passed"],
                "reward": rec["reward"],
                "r_human": rec["r_human"],
                "intent_tags": rec["intent_tags"],
                "artifact_tags": rec["artifact_tags"],
                "n_steps": rec["n_steps"],
            }, ensure_ascii=False) + "\n")
            for i, s in enumerate(rec["steps"]):
                fl.write(json.dumps({
                    "trace_id": f"{rec['task_id']}__step{i}",
                    "task_id": rec["task_id"],
                    "step_idx": i,
                    "state_summary": s["state_summary"],
                    "tool_call": s["tool_call"],
                    "observation": s["observation"],
                    "reflection_v2": s.get("reflection_v2", {}),
                    "verifier_signal": {
                        "task_passed": rec["passed"],
                        "task_reward": rec["reward"],
                    },
                    "alpha": s.get("alpha", 0.3),
                    "V": s.get("V", 0.0),
                    "r_human": rec["r_human"],
                    "task_intent_tags": rec["intent_tags"],
                    "task_artifact_tags": rec["artifact_tags"],
                }, ensure_ascii=False) + "\n")
            fl.flush()
            fs.flush()
            done += 1
            if done % 5 == 0:
                print(f"  processed {done}/{len(task_dirs)}")

    print(f"\nDone. {done} tasks. Output:")
    print(f"  L1 traces: {l1p}")
    print(f"  Task summaries: {sp}")


if __name__ == "__main__":
    sys.exit(main())

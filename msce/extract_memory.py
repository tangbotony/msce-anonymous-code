#!/usr/bin/env python3
"""MSCE — Step 1: structured L1 trace extraction with posthoc reflection.

Implementation notes:
1. Parse tool_call into structured {name, command_kind, command_text, args_raw}.
2. Parse observation into structured {raw, exit_code, error_kind, files_created, key_signal}.
3. Construct a compact posthoc reflection_v2 when original sessions have sparse self-reflection.
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

_SENSITIVE_REDACTIONS = [
    (re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1<REDACTED_TOKEN>"),
    (re.compile(r"(?i)\b(sk-[A-Za-z0-9_-]{12,})\b"), "<REDACTED_API_KEY>"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,;]+"), r"\1=<REDACTED_SECRET>"),
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "<REDACTED_EMAIL>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<REDACTED_IP>"),
    (re.compile(r"https?://[^\s'\"<>]+"), "<REDACTED_URL>"),
    (re.compile(r"(?:/Users|/home|/root)/[^\s'\"<>]+"), "<REDACTED_PATH>"),
]


def redact_sensitive(text: str) -> str:
    out = text or ""
    for rx, repl in _SENSITIVE_REDACTIONS:
        out = rx.sub(repl, out)
    return out


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

        text = redact_sensitive("\n".join(text_parts).strip())

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
                command_text = redact_sensitive(command_text or "")
                args_raw = redact_sensitive(json.dumps(args, ensure_ascii=False))
                pending = {
                    "state_summary": redact_sensitive(state_summary)[:1200],
                    "tool_call": {
                        "name": tc.get("name") or "exec",
                        "command_kind": command_kind,
                        "command_text": command_text[:1500],
                        "args_raw": args_raw[:1500],
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
                        "state_summary": redact_sensitive(state_summary)[:1200],
                        "tool_call": {"name": "final_answer", "command_kind": "final",
                                       "command_text": "", "args_raw": ""},
                        "observation": _obs_record(""),
                        "assistant_text_final": assistant_text,
                    })
        elif role in ("tool", "toolResult"):
            obs_raw = redact_sensitive(text)
            if pending is not None:
                steps.append({**pending, "observation": _obs_record(obs_raw)})
                state_summary = obs_raw[-1200:]
                pending = None

    if pending is not None:
        steps.append({**pending, "observation": _obs_record("")})

    return steps


def _obs_record(raw: str) -> dict:
    safe_raw = redact_sensitive(raw or "")
    return {
        "raw_truncated": safe_raw[:1500],
        "exit_code": detect_exit_code(raw or ""),
        "error_kind": classify_error_kind(raw or ""),
        "files_created": [redact_sensitive(x)[:200] for x in detect_files_created(raw or "")],
        "key_signal": "",   # filled by posthoc reflection
    }


# ── Posthoc reflection summary ──────────────────────────────────────


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
    cmd_kind = tc.get("command_kind", "")
    exit_code = ob.get("exit_code")
    error_kind = ob.get("error_kind")
    files_created = ob.get("files_created") or []
    is_progress = bool(exit_code == 0 or files_created)
    is_blocker = bool(error_kind and not is_progress)
    if files_created:
        what = f"{cmd_kind} produced artifact(s): {', '.join(files_created[:3])}."
        key_signal = "artifact_created"
    elif exit_code == 0:
        what = f"{cmd_kind} completed successfully."
        key_signal = "successful_tool_call"
    elif error_kind:
        what = f"{cmd_kind} failed with {error_kind}."
        key_signal = error_kind
    else:
        what = f"{cmd_kind} produced no explicit success or failure signal."
        key_signal = "unclear_outcome"
    return {
        "what_happened": what[:300],
        "is_progress": is_progress,
        "is_blocker": is_blocker,
        "error_diagnosis": (error_kind or "")[:300],
        "next_action_hint": "inspect the error signal and choose an alternate tool or dependency path" if is_blocker else "",
        "key_signal": key_signal[:200],
    }


# ── Reward quantification ──────────────────────────────────────────

R_HUMAN_SYS = """You are a strict grader of AI-agent task execution.

You receive:
- TASK_SUMMARY the FULL conversation arc for this task:
  * USER_ASKS_AND_AGENT_REPLIES lists every user turn paired with the agent's corresponding reply, in chronological order. One "task" frequently spans multiple user turns as the user refines / follows up / pivots topics within the same session.
  * MOST_RECENT_USER_ASK and MOST_RECENT_AGENT_REPLY call out the final exchange explicitly; that is usually the truest signal of whether the agent is actually tracking where the user is now.
- FEEDBACK the user's own messages AFTER the task attempt finished. May be short ("ok thanks"), explicit ("try again with X"), or structured ("resolved, but too slow"). Frequently empty.

Grade the agent on THREE INDEPENDENT AXES, each in [-1, 1]:

1. "goal_achievement" did the agent address what the user ACTUALLY asked?
   +1.0 every user ask across the exchange was addressed correctly.
   +0.3 the last ask was addressed well; earlier asks had minor gaps.
   0.0 unclear if the user's ask was met.
   -0.3 missed a significant portion of what was asked.
   -1.0 fundamentally wrong answer / caused damage.

   CRITICAL RULE: do NOT anchor on the first user turn. Judge each user ask on its own merits, weighted toward the most recent exchange.

2. "process_quality"
   +1.0 clean, minimal, correct reasoning across all turns.
   0.0 reasonable but not great.
   -1.0 lots of thrashing, wrong tools, noisy output.

3. "user_satisfaction" (from FEEDBACK text tone + trailing user asks)
   +1.0 thanks / happy / "" / accepts and closes out.
   +0.3 moves on neutrally to next ask or new topic.
   0.0 no emotional signal either way.
   -0.3 asks for correction ("no, do X instead" / "").
   -1.0 hard-stops, expresses frustration.

Rules:
- If FEEDBACK is empty, infer satisfaction CONSERVATIVELY from the last exchange's tone. A follow-up question is usually 0 (neutral continuation), NOT negative. Never invent anger.
- Base scores ONLY on what TASK_SUMMARY actually describes; do not assume facts not shown.
- You are grading the HOST AGENT described in HOST_AGENT_CONTEXT, not yourself. Do NOT use your own model identity, provider, policies, or capabilities.
- Produce one short justification.

Return JSON, EXACTLY this shape (no extra keys, no commentary):
{
  "goal_achievement": number in [-1, 1],
  "process_quality": number in [-1, 1],
  "user_satisfaction": number in [-1, 1],
  "label": "success" | "partial" | "failure" | "unknown",
  "reason": "one-sentence justification"
}"""


def infer_task_tags(task_prompt: str, steps_summary: str) -> tuple[list[str], list[str]]:
    text = f"{task_prompt}\n{steps_summary}".lower()
    intent, artifact = set(), set()
    rules = [
        ("code-impl-algorithm", ("python", "pytest", "leetcode", "function", "algorithm")),
        ("software-engineering", ("repo", "bug", "test", "patch", "github", "swe")),
        ("info-retrieval", ("search", "find", "research", "browse", "web")),
        ("math-reasoning", ("prove", "calculate", "probability", "equation", "geometry")),
        ("office-doc-gen", ("docx", "xlsx", "pptx", "pdf", "spreadsheet", "document")),
        ("knowledge-work", ("report", "memo", "analysis", "summary", "slides")),
    ]
    for tag, needles in rules:
        if any(n in text for n in needles):
            intent.add(tag)
    artifact_rules = [
        ("code", (".py", "code", "repo", "patch")),
        ("docx", ("docx", "word document")),
        ("xlsx", ("xlsx", "spreadsheet", "excel")),
        ("pdf", ("pdf",)),
        ("markdown-report", ("markdown", "report", "summary")),
        ("search-answer", ("search", "answer", "research")),
    ]
    for tag, needles in artifact_rules:
        if any(n in text for n in needles):
            artifact.add(tag)
    return sorted(intent)[:6], sorted(artifact)[:6]


def score_r_human(task_id: str, passed: bool, verifier_details: dict,
                   task_prompt: str, last_assistant: str,
                   steps_summary: str) -> dict:
    verifier_json = redact_sensitive(json.dumps(verifier_details, ensure_ascii=False))[:1000]
    task_prompt = redact_sensitive(task_prompt)
    last_assistant = redact_sensitive(last_assistant)
    steps_summary = redact_sensitive(steps_summary)
    user = (
        "HOST_AGENT_CONTEXT: provider-neutral LLM agent runtime.\n"
        "TASK_SUMMARY:\n"
        f"- task_id: {task_id}\n"
        f"- verifier_passed: {passed}\n"
        f"- verifier_details: {verifier_json}\n"
        f"- USER_ASKS_AND_AGENT_REPLIES:\n{task_prompt[:2500]}\n"
        f"- MOST_RECENT_USER_ASK:\n{task_prompt[-1200:]}\n"
        f"- MOST_RECENT_AGENT_REPLY:\n{last_assistant[:1500]}\n"
        f"- COMPACT_TRACE_SUMMARY:\n{steps_summary[:3000]}\n"
        "FEEDBACK:\n"
        f"{verifier_json}"
    )
    res = call_llm(R_HUMAN_SYS, user, max_tokens=400, temperature=0.0)
    fallback_r = 0.7 if passed else -0.5
    intent_tags, artifact_tags = infer_task_tags(task_prompt, steps_summary)
    if not isinstance(res, dict):
        return {"r_human": fallback_r, "intent_tags": intent_tags,
                "artifact_tags": artifact_tags}
    try:
        g = float(res.get("goal_achievement", 0.0))
        p = float(res.get("process_quality", 0.0))
        u = float(res.get("user_satisfaction", 0.0))
        r = 0.45 * g + 0.30 * p + 0.25 * u
    except Exception:
        r = fallback_r
    return {
        "r_human": max(-1.0, min(1.0, float(r))),
        "intent_tags": intent_tags,
        "artifact_tags": artifact_tags,
        "reward_axes": {
            "goal_achievement": res.get("goal_achievement"),
            "process_quality": res.get("process_quality"),
            "user_satisfaction": res.get("user_satisfaction"),
            "label": res.get("label"),
            "reason": res.get("reason"),
        },
    }


# ── alpha scoring (using reflection_v2) ─────────────────────────────


ALPHA_SYS = """You are a strict reviewer of agent self-reflections.

You see the FULL context of one agent step:
- STATE what the agent saw before acting (user prompt, prior observation)
- THINKING the LLM's own native chain-of-thought for this step, if any. Empty when the model didn't emit thinking this turn.
- ACTION what the agent produced (assistant text output)
- TOOL_CALLS tools the agent invoked this step, with inputs and outputs (or errors). Tool usage + outcomes are part of the action chain and carry their own signal about what the agent did.
- OUTCOME the final observable result of the step (last tool outcome or "(assistant-only step)" for pure text turns)
- REFLECTION the text being graded: the agent's first-person explanation of WHY it acted this way and WHAT it learned.

Score the REFLECTION on four axes, combined into ONE number [0, 1]:
1. faithfulness: does the reflection match what ACTUALLY happened across THINKING + ACTION + TOOL_CALLS + OUTCOME?
2. causal insight: does it identify why the action / tool choice worked or failed?
3. transferability: does it surface a lesson useful on a similar future task?
4. concreteness: are the details specific rather than generic platitudes?

Rules:
- THINKING and TOOL_CALLS are first-class evidence for grading.
- TOOL_CALLS that errored are strong signal: the reflection should name the error and what it implied.
- An empty / purely-tautological reflection = 0, usable = false.
- alpha >= 0.4 AND reflection non-tautological usable = true; else false.

Return JSON:
{
  "alpha": 0.0-1.0,
  "usable": true | false,
  "reason": "one-sentence justification"
}"""


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
        f"STATE: {step.get('state_summary', '')[:600]}\n"
        "THINKING:\n"
        "ACTION:\n"
        f"TOOL_CALLS: {json.dumps(step.get('tool_call', {}), ensure_ascii=False)[:800]}\n"
        f"OUTCOME: exit_code={ob.get('exit_code')} error_kind={error_kind} "
        f"files_created={ob.get('files_created')} raw={ob.get('raw_truncated', '')[:600]}\n"
        "REFLECTION:\n"
        f"{json.dumps(refl, ensure_ascii=False)[:1000]}"
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

#!/usr/bin/env python3
"""MSCE eval entrypoint.

Uses EvoAgentBench domain/agent/runner plumbing and supplies the MSCE
retrieval + prompt-rendering layer (TaskProfiler + SkillRetriever + RRF +
utility/value gates).

CLI:
    python eval_with_skills_v3.py \
        --domain knowledge_work \
        --skill-bank      runs/kw/skill_bank.jsonl \
        --topics          runs/kw/l3_topics.jsonl \
        --emb-npy         runs/kw/skill_embeddings.npy \
        --emb-ids         runs/kw/skill_ids.json \
        --profile-cache   runs/kw/task_profiles.jsonl \
        --split test --parallel 4 --top-k 3 --min-gain 0.05 \
        --job msce-kw-<ts>
"""
from __future__ import annotations
import argparse, json, os, sys, uuid
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# retrieval lib (sibling)
try:
    from .retrieval import SkillRetriever, TaskProfiler   # noqa: E402
except ImportError:  # pragma: no cover - allows direct script execution
    from retrieval import SkillRetriever, TaskProfiler   # noqa: E402

# Hook into the benchmark runner/domain/agent plumbing.
_BENCH_ROOT = Path(os.environ.get("EVOAGENTBENCH_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(_BENCH_ROOT / "src"))

def patch_domain_build_prompt(domain) -> None:
    """Append `task['skill_text']` to benchmark prompts when needed."""
    if getattr(domain, "name", "") == "reasoning":
        return
    original = domain.build_prompt

    def build_with_msce(task, env_info):
        prompt = original(task, env_info)
        text = task.get("skill_text") or ""
        if not text:
            return prompt
        return f"{prompt}\n\n{text}"

    domain.build_prompt = build_with_msce


def load_env_file(project_root: Path) -> None:
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _resolve_tasks(domain, args) -> list[dict]:
    task_args = argparse.Namespace(
        task=args.task,
        split=args.split,
        trials=1,
        parallel=1,
        max_retries=0,
        live=False,
        disk_budget=None,
    )
    return domain.load_tasks(task_args)


# ── Prompt rendering ───────────────────────────────────────────────


HEADER = (
    "## Reference Skills + Topic Context (Candidates — May Not Apply)\n\n"
    "Below is a small set of candidate skills retrieved from a training-derived "
    "skill bank, filtered by topic + keyword + semantic relevance, then gated by "
    "historical expected_gain (vs no-skill baseline on same intent). They are NOT "
    "mandatory. **If a skill's trigger does not fit this task, skip it.** Pay "
    "attention to ⚠️ Anti-patterns — they are common failure modes in this topic. "
    "Your own judgment of the task takes precedence.\n\n"
)


# ── Domain-adaptive prompt augmentation (v3.2) ─────────────────────
# Different domains need different in-context strategies. The MSCE
# bank is one source of memory; here we add a *domain operating discipline*
# that is grounded in the failure modes we observed in v3 evals.

DOMAIN_INSTRUCTIONS = {
    "information_retrieval": (
        "## Multi-Hop Retrieval Discipline (information_retrieval)\n\n"
        "This task is a multi-hop entity-grounded retrieval problem. Empirically, "
        "shallow searches lose 70%+ of cases. Therefore:\n"
        "1. **Decompose the question** into entity / relation / constraint. "
        "Write the decomposition before searching.\n"
        "2. **Plan hops**: hop-1 find the seed entity; hop-2 resolve the relation; "
        "hop-3 cross-verify via a different phrasing or source.\n"
        "3. **Perform at least 4–6 distinct retrieval attempts** (different "
        "queries, not the same query retried) before committing to an answer, "
        "unless TWO independent sources already confirm the same answer.\n"
        "4. **On empty / 403 / unsupported_freshness**: do NOT retry the same "
        "query. Switch provider or remove unsupported filters, or rephrase with "
        "an alias / abbreviation / year variant.\n"
        "5. **Stopping rule**: only commit to an answer when (a) two "
        "independent sources confirm it, OR (b) you have exhausted ≥5 search "
        "strategies. Otherwise say you cannot verify.\n"
        "6. Final answer MUST cite the URL(s) that verified it."
    ),

    "reasoning": (
        "## Reasoning Discipline (reasoning)\n\n"
        "This task is single-turn reasoning, not multi-step tool use. Tool "
        "skills do not help here — what helps is disciplined thinking. So:\n"
        "1. **Restate the question** in your own words and identify exactly "
        "what type of answer is required (number / option letter / set / yes-no).\n"
        "2. **List ALL hypotheses / cases / variables** before solving. Do not "
        "jump to one path.\n"
        "3. **For each hypothesis, search for ONE concrete counter-example** "
        "before accepting it.\n"
        "4. **Common traps to watch**: confusing correlation with causation; "
        "ignoring boundary conditions (n=0, empty set, negative values); "
        "off-by-one; conflating necessary vs sufficient; missing a case in "
        "case-analysis.\n"
        "5. **Output format**: end with EXACTLY the format requested (e.g. "
        "single letter A/B/C/D, a number with units, a single boolean). No "
        "extra prose around the final answer."
    ),

    "software_engineering": (
        "## SE Discipline (software_engineering)\n\n"
        "This is repository-level patch generation. So:\n"
        "1. **Read the failing test FIRST** to understand the contract.\n"
        "2. **Locate the file with `grep / find`**, do NOT guess paths.\n"
        "3. **Minimal diff**: change only what is needed to make the test pass; "
        "do not refactor unrelated code.\n"
        "4. **Run the specific failing test BEFORE finalizing** to verify the "
        "patch."
    ),

    "code_implementation": (
        "## Code Implementation Discipline (code_implementation)\n\n"
        "This is competitive programming. Tool skills rarely help — algorithm "
        "selection does. So:\n"
        "1. **Read constraints first** (N, time limit, value bounds) to "
        "narrow down algorithm class (O(N), O(N log N), O(N²), DP, graph, "
        "greedy, math).\n"
        "2. **Sketch the algorithm in 3-5 lines** before coding.\n"
        "3. **Handle edge cases explicitly**: empty input, single element, "
        "max bounds, all-same, negatives if applicable.\n"
        "4. **Run on the sample inputs** before submitting."
    ),

    "knowledge_work": "",  # KW already wins with skill bank, no extra discipline
}


def render_domain_instruction(domain: str) -> str:
    text = DOMAIN_INSTRUCTIONS.get(domain, "")
    return text.strip()


def render_topic_context(topic: dict) -> str:
    wk = topic.get("world_knowledge", {}) or {}
    atlas = topic.get("tool_atlas", {}) or {}
    pitfalls = topic.get("common_pitfalls", []) or []
    md = [f"### Topic: {topic.get('topic_name','?')}  ({topic.get('topic_id','')})"]
    if wk.get("spatial_structure"):
        md.append(f"**Spatial structure:** {wk['spatial_structure']}")
    if wk.get("behavior_rules"):
        md.append(f"**Behavior rules:** {wk['behavior_rules']}")
    if wk.get("constraints"):
        md.append(f"**Constraints:** {wk['constraints']}")
    if atlas:
        rows = []
        for k, st in sorted(atlas.items(),
                             key=lambda kv: -kv[1].get("v_avg_in_topic", 0))[:6]:
            rows.append(f"  - `{k}` [{st.get('role','?')}] V={st.get('v_avg_in_topic',0):.2f} "
                        f"(success={st.get('success_rate_in_topic',0):.2f}, n={st.get('n_uses',0)})")
        md.append("**Tool atlas (this topic):**\n" + "\n".join(rows))
    if pitfalls:
        md.append("**Common pitfalls:**")
        for p in pitfalls[:5]:
            md.append(f"  - signal: `{p.get('signal','')}` → remedy: {p.get('remedy','')}")
    return "\n".join(md)


def _txt(x):
    if isinstance(x, dict):
        return x.get("text", "")
    return str(x or "")


def render_verifier_card(s: dict) -> str:
    """MSCE-R: render a compact verifier card (~80-150 tokens).

    Used only for skills with `card_kind == "verifier"`. The card teaches the
    model what to check after solving, not how to solve.
    """
    arch = s.get("archetype", "general-math")
    md = [f"### Verifier Card [{arch}]   *(gain=+{(s.get('expected_gain') or {}).get('gain',0):.2f}, "
          f"V_with={(s.get('expected_gain') or {}).get('v_avg_with',0):.2f} vs "
          f"V_without={(s.get('expected_gain') or {}).get('v_avg_without',0):.2f})*"]
    if s.get("summary"):
        md.append(f"**Summary:** {s['summary']}")
    if s.get("check"):
        md.append(f"**Must-Check (sanity invariant):** {s['check']}")
    if s.get("trap"):
        md.append(f"**Common Trap (from training failures):** {s['trap']}")
    if s.get("skip_if"):
        md.append(f"**Skip if:** {s['skip_if']}")
    return "\n".join(md)


def render_skill_compact(s: dict) -> str:
    """Minimal skill rendering for cost-aware injection (~150-200 tokens vs
    600+ of the full renderer). Keeps only: name, trigger, 1-line procedure,
    anti-pattern. Drops abstract, topic field, V/gain metadata, essential/
    optional/redundant step lists, verification, fail_signals, and ALL
    worked examples. Use --compact-skill at eval time."""
    md = [f"### Skill: {s.get('name','?')}"]
    trig = _txt(s.get("trigger"))
    if trig:
        md.append(f"**Trigger:** {trig}")
    proc = s.get("procedure", {}) or {}
    proc_text = ""
    if isinstance(proc, dict):
        proc_text = (proc.get("text") or "").strip()
    elif isinstance(proc, str):
        proc_text = proc.strip()
    if proc_text:
        # Keep procedure to one short line for cost.
        if len(proc_text) > 320:
            proc_text = proc_text[:317] + "..."
        md.append(f"**Procedure:** {proc_text}")
    ap = s.get("anti_pattern") or {}
    if isinstance(ap, dict) and ap.get("text"):
        ap_text = ap["text"].strip()
        if len(ap_text) > 200:
            ap_text = ap_text[:197] + "..."
        md.append(f"**Avoid:** {ap_text}")
    return "\n".join(md)




def render_skill_watchout(s: dict) -> str:
    """Cost-aware SE/Code rendering: no procedure, only trigger + trap/check.

    Procedure-style skills often increase tool turns in SWE/Code tasks. This
    renderer converts a retrieved skill into a small selector/verifier note so
    the agent keeps its normal shortest path while avoiding known traps.
    """
    md = [f"### Watchout: {s.get('name','?')}"]
    trig = _txt(s.get("trigger"))
    if trig:
        if len(trig) > 220:
            trig = trig[:217] + "..."
        md.append(f"**Use only if:** {trig}")
    ap = s.get("anti_pattern") or {}
    if isinstance(ap, dict) and ap.get("text"):
        ap_text = ap["text"].strip()
        if len(ap_text) > 260:
            ap_text = ap_text[:257] + "..."
        md.append(f"**Avoid:** {ap_text}")
    ver = _txt(s.get("verification"))
    if ver:
        if len(ver) > 180:
            ver = ver[:177] + "..."
        md.append(f"**Quick check:** {ver}")
    return "\n".join(md)


def source_step_avg(s: dict) -> float:
    """Average training step index from source trace ids; lower means cheaper.

    Trace ids look like `task__trial_1__step18`. If unavailable, return 999.
    """
    import re
    vals = []
    for tid in s.get("source_traces") or []:
        m = re.search(r"__step(\d+)$", str(tid))
        if m:
            vals.append(int(m.group(1)))
    return sum(vals) / len(vals) if vals else 999.0


def _extract_file_hints(s: dict, limit: int = 4) -> list[str]:
    """Extract likely source-file hints from training commands/states.

    This is intentionally heuristic and training-only. It turns procedural
    traces into low-intervention localization hints for SWE tasks.
    """
    import collections
    import re
    counter = collections.Counter()
    examples = (s.get("positive_examples") or []) + (s.get("negative_examples") or [])
    for ex in examples:
        action = ex.get("action") or {}
        text = " ".join([
            str(action.get("command_text") or ""),
            str(ex.get("state") or ""),
            str(ex.get("outcome") or ""),
        ])
        for m in re.finditer(r"(?<![-\\w/])([A-Za-z0-9_./-]+\\.py)", text):
            path = m.group(1).strip("'\"`.,:;()[]{}")
            if path.startswith("/tmp/") or path.startswith(str(Path.home()) + "/"):
                continue
            if path.count("/") > 5:
                continue
            counter[path] += 1
    return [p for p, _ in counter.most_common(limit)]


def render_localization_card(s: dict) -> str:
    """SE-focused card: localization + trap + minimal execution discipline.

    Unlike full skills, this does not prescribe a multi-step procedure. The goal
    is to reduce repository exploration turns while preserving the useful memory.
    """
    md = [f"### Localization Card: {s.get('name','?')}"]
    files = _extract_file_hints(s)
    if files:
        md.append("**Likely files from training:** " + ", ".join(f"`{p}`" for p in files[:4]))
    trig = _txt(s.get("trigger"))
    if trig:
        if len(trig) > 180:
            trig = trig[:177] + "..."
        md.append(f"**Use only if:** {trig}")
    summary = str(s.get("summary") or s.get("abstract") or "").strip()
    if summary:
        if len(summary) > 220:
            summary = summary[:217] + "..."
        md.append(f"**Bug pattern:** {summary}")
    ap = s.get("anti_pattern") or {}
    if isinstance(ap, dict) and ap.get("text"):
        ap_text = ap["text"].strip()
        if len(ap_text) > 220:
            ap_text = ap_text[:217] + "..."
        md.append(f"**Avoid:** {ap_text}")
    md.append(
        "**Execution rule:** inspect at most 1-2 likely files, make a minimal patch, "
        "run one targeted verification, then stop. Do not broaden search unless the "
        "first localization is clearly wrong."
    )
    return "\n".join(md)


def render_algorithm_selector_card(s: dict) -> str:
    """Code-focused card: algorithm selector, not a procedure."""
    md = [f"### Algorithm Selector: {s.get('name','?')}"]
    kws = s.get("lexical_keywords") or []
    if kws:
        md.append("**Signals:** " + ", ".join(f"`{k}`" for k in kws[:8]))
    trig = _txt(s.get("trigger"))
    if trig:
        if len(trig) > 180:
            trig = trig[:177] + "..."
        md.append(f"**Use only if:** {trig}")
    summary = str(s.get("summary") or s.get("abstract") or "").strip()
    if summary:
        if len(summary) > 220:
            summary = summary[:217] + "..."
        md.append(f"**Pattern:** {summary}")
    ap = s.get("anti_pattern") or {}
    if isinstance(ap, dict) and ap.get("text"):
        ap_text = ap["text"].strip()
        if len(ap_text) > 180:
            ap_text = ap_text[:177] + "..."
        md.append(f"**Trap:** {ap_text}")
    md.append(
        "**Execution rule:** use this only to choose the algorithm/invariant. "
        "Do not add exploration steps because of this card; solve directly."
    )
    return "\n".join(md)

def render_skill(s: dict, no_worked_examples: bool = False) -> str:
    eg = s.get("expected_gain") or {}
    note = (f"gain=+{eg.get('gain',0):.2f}, n_pos={eg.get('n_pos',0)}/n_neg={eg.get('n_neg',0)}, "
            f"V_with={eg.get('v_avg_with',0):.2f} vs V_without={eg.get('v_avg_without',0):.2f}")
    md = [f"### Skill: {s.get('name','?')}   *({note})*",
          f"*skill_id: {s.get('skill_id','?')} | topic: {s.get('topic_id','')}*"]
    if s.get("abstract"):
        md.append(f"**Abstract:** {s['abstract']}")
    md.append(f"**Trigger:** {_txt(s.get('trigger'))}")
    proc = s.get("procedure", {}) or {}
    if isinstance(proc, dict):
        if proc.get("text"):
            md.append(f"**Procedure:** {proc['text']}")
        if proc.get("essential_steps"):
            md.append("**Essential steps:**")
            for e in proc["essential_steps"][:5]:
                md.append(f"  - `{e.get('tool_kind','?')}` — {e.get('purpose','')}"
                          + (" (must succeed)" if e.get("must_succeed") else ""))
        if proc.get("optional_steps"):
            md.append("**Optional steps:**")
            for o in proc["optional_steps"][:4]:
                hint = f" (skip if {o['skippable_if']})" if o.get("skippable_if") else ""
                md.append(f"  - `{o.get('tool_kind','?')}` — {o.get('purpose','')}{hint}")
        if proc.get("redundant_steps"):
            md.append("**Avoid (low-V exploration in this topic):**")
            for r in proc["redundant_steps"][:3]:
                md.append(f"  - `{r.get('tool_kind','?')}` — {r.get('reason','')}")
    if s.get("verification"):
        md.append(f"**Verify:** {_txt(s.get('verification'))}")
    ap = s.get("anti_pattern") or {}
    if isinstance(ap, dict):
        if ap.get("text"):
            md.append(f"**⚠️ Anti-pattern:** {ap['text']}")
        if ap.get("fail_signals"):
            md.append(f"  - fail_signals: {ap['fail_signals']}")
    # NAT-style worked examples (✅ / ❌)
    # Suppressed when no_worked_examples=True (e.g. single-step reasoning)
    pos_ex = [] if no_worked_examples else (s.get("positive_examples") or [])
    neg_ex = [] if no_worked_examples else (s.get("negative_examples") or [])
    if pos_ex or neg_ex:
        md.append("")
        md.append("**Worked examples (real grounded traces from training):**")
        for ex in pos_ex:
            ac = ex.get("action") or {}
            ob = ex.get("observation") or {}
            md.append(
                f"  ✅ correctly — `{ac.get('tool_kind','')}`: "
                f"`{(ac.get('command_text') or '')[:160]}`"
            )
            if ex.get("state"):
                md.append(f"      state: {ex['state'][:200]}")
            md.append(
                f"      → exit={ob.get('exit_code')} "
                f"files={ob.get('files_created')} key={ob.get('key_signal','')[:80]}"
            )
            if ex.get("outcome"):
                md.append(f"      outcome: {ex['outcome'][:160]}")
        for ex in neg_ex:
            ac = ex.get("action") or {}
            ob = ex.get("observation") or {}
            md.append(
                f"  ❌ incorrectly — `{ac.get('tool_kind','')}`: "
                f"`{(ac.get('command_text') or '')[:160]}`"
            )
            if ex.get("state"):
                md.append(f"      state: {ex['state'][:200]}")
            md.append(
                f"      → exit={ob.get('exit_code')} err={ob.get('error_kind')}"
            )
            if ex.get("why"):
                md.append(f"      why it failed: {ex['why'][:200]}")
    return "\n".join(md)


VERIFIER_HEADER = (
    "## Reasoning Verifier Card (sanity check, not solving guidance)\n\n"
    "The card below names the problem archetype, a hard invariant the answer "
    "must satisfy, and the most common pitfall observed in training failures. "
    "**Solve the problem with your own reasoning first.** Then briefly check "
    "the answer against the invariant. Use the trap only as a watch-out, "
    "not as a hint about the solution.\n\n"
)


def build_injection(chosen_skills: list[dict], topics: list[dict],
                    domain: str = "",
                    no_worked_examples: bool = False,
                    compact_skill: bool = False,
                    no_topic_context: bool = False,
                    watchout_only: bool = False,
                    card_mode: str = "") -> str:
    """Inject domain discipline (always) + topic context + skills (if any).

    If ALL chosen skills are verifier cards, use the verifier renderer and
    suppress topic/world-model blocks (the card is self-contained).
    """
    parts = []
    domain_instr = render_domain_instruction(domain)
    if domain_instr:
        parts.append(domain_instr)
    if chosen_skills:
        is_all_verifier = all(s.get("card_kind") == "verifier" for s in chosen_skills)
        if is_all_verifier:
            parts.append(VERIFIER_HEADER)
            for s in chosen_skills:
                parts.append(render_verifier_card(s))
        else:
            # Compact mode skips the long HEADER, topic context, and rich
            # skill rendering — keeps only a short instruction + compact card.
            if card_mode == "localization":
                parts.append(
                    "## Cost-Aware Localization Cards (minimal patch guidance)\n"
                    "Use these as localization/trap hints only. They should reduce search turns, "
                    "not add a procedure to follow.\n"
                )
                for s in chosen_skills:
                    parts.append(render_localization_card(s))
            elif card_mode == "selector":
                parts.append(
                    "## Algorithm Selector Cards (choose pattern, then solve directly)\n"
                    "Use these as algorithm/trap hints only. Do not add tool calls because of them.\n"
                )
                for s in chosen_skills:
                    parts.append(render_algorithm_selector_card(s))
            elif watchout_only:
                parts.append(
                    "## Cost-Aware Skill Watchouts (do not add extra exploration)\n"
                    "Use these only as routing/trap checks. Do NOT follow them as a procedure.\n"
                )
                for s in chosen_skills:
                    parts.append(render_skill_watchout(s))
            elif compact_skill:
                parts.append(
                    "## Reference Skills (candidates; apply only if trigger fits)\n"
                )
                for s in chosen_skills:
                    parts.append(render_skill_compact(s))
            else:
                parts.append(HEADER)
                if not no_topic_context:
                    for t in topics[:2]:
                        parts.append(render_topic_context(t))
                for s in chosen_skills:
                    parts.append(render_skill(s, no_worked_examples=no_worked_examples))
    if not parts:
        return ""
    return "\n\n".join(parts)


# ── Main (benchmark plumbing + MSCE retrieval) ─────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="MSCE eval")
    ap.add_argument("--config", default=str(_BENCH_ROOT / "config.yaml"))
    ap.add_argument("--domain", default=None)
    ap.add_argument("--skill-bank", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--emb-npy", required=True)
    ap.add_argument("--emb-ids", required=True)
    ap.add_argument("--profile-cache", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--job", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--per-route-topn", type=int, default=10)
    ap.add_argument("--disable-topic", action="store_true")
    ap.add_argument("--disable-bm25", action="store_true")
    ap.add_argument("--disable-dense", action="store_true")
    ap.add_argument("--disable-domain-instruction", action="store_true",
                    help="Skip domain-adaptive prompt augmentation (v3 baseline mode)")
    # V-calibrated gate: filters skills by training-set V_with / V_without thresholds.
    # Rationale: V_without too high → skill adds marginal value (no generalization needed);
    # V_without too low → skill overfits specific training examples (memorization);
    # V_with too low → skill itself has poor training-time quality.
    # All thresholds are derived from training-set statistics only (no test-set leakage).
    ap.add_argument("--vgate-with-min", type=float, default=0.0,
                    help="Min V_with for skill to pass V-calibrated gate (0=disabled)")
    ap.add_argument("--vgate-without-min", type=float, default=0.0,
                    help="Min V_without for skill to pass V-calibrated gate")
    ap.add_argument("--vgate-without-max", type=float, default=1.0,
                    help="Max V_without for skill to pass V-calibrated gate")
    ap.add_argument("--no-worked-examples", action="store_true",
                    help="Strip worked examples from skill injection (single-step domains)")
    # Cost-aware injection options (v4.1 / Plan A).
    ap.add_argument("--compact-skill", action="store_true",
                    help="Use a ~150-200 token compact skill renderer (drops topic context, abstract, V metadata, examples). Keeps name + trigger + 1-line procedure + anti-pattern.")
    ap.add_argument("--no-topic-context", action="store_true",
                    help="In full-render mode only, drop the L3 topic context block.")
    ap.add_argument("--high-conf-top1", action="store_true",
                    help="When the top-ranked skill has V_with > 0.95, inject only that single skill (overrides --top-k).")
    ap.add_argument("--turn-budget", type=int, default=0,
                    help="If >0, prepend a turn-budget hint to every task asking the agent to solve within N turns.")
    ap.add_argument("--watchout-only", action="store_true",
                    help="Render retrieved skills as short watchout/check cards, dropping procedure text.")
    ap.add_argument("--card-mode", choices=["", "watchout", "localization", "selector"],
                    default="", help="Special low-intervention rendering mode.")
    ap.add_argument("--max-source-step-avg", type=float, default=0.0,
                    help="If >0, drop skills whose average source trace step exceeds this value (cost proxy).")
    ap.add_argument("--cost-lambda-step", type=float, default=0.0,
                    help="If >0, require gain - lambda*(avg_source_step/10) > --min-cost-utility.")
    ap.add_argument("--min-cost-utility", type=float, default=0.0,
                    help="Minimum utility for --cost-lambda-step gate.")
    args = ap.parse_args()

    load_env_file(_BENCH_ROOT)
    from config import get_agent, get_domain, load_config, get_config   # type: ignore
    from runner import run_all   # type: ignore

    load_config(args.config)
    cfg = get_config()
    domain_name = args.domain or cfg["domain"]["name"]
    domain = get_domain(domain_name)
    agent = get_agent(cfg["agent"]["name"])

    # Build v3 retriever
    print(f"[msce] loading skill bank from {args.skill_bank}")
    retriever = SkillRetriever.load(args.skill_bank, args.topics,
                                     args.emb_npy, args.emb_ids)
    print(f"[msce] loaded {len(retriever.skills)} skills, "
          f"{len(retriever.topics)} topics")
    profiler = TaskProfiler(cache_path=args.profile_cache)

    # Resolve tasks via benchmark domain plumbing
    tasks = _resolve_tasks(domain, args)
    print(f"[msce] evaluating {len(tasks)} task(s) on domain={domain_name}")

    # Profile + retrieve per task → attach skill_text
    n_inject = 0
    debug = []
    for t in tasks:
        task_id = (t.get("name") or t.get("task_id") or t.get("id") or "unknown")
        prompt = None
        for key in ("problem", "problem_statement", "query", "question",
                    "prompt", "description"):
            v = t.get(key)
            if isinstance(v, str) and v.strip():
                prompt = v
                break
        if prompt is None:
            prompt = json.dumps(t, ensure_ascii=False)[:4000]
        profile = profiler.profile(str(task_id), prompt)
        chosen, dbg = retriever.retrieve(
            profile,
            top_k=args.top_k,
            min_gain=args.min_gain,
            per_route_topn=args.per_route_topn,
            enable_topic=not args.disable_topic,
            enable_bm25=not args.disable_bm25,
            enable_dense=not args.disable_dense,
        )
        # V-calibrated gate: filter by training-set V statistics only (no test leakage).
        # V_without in (vgate_without_min, vgate_without_max) ensures the skill provides
        # genuine generalizable benefit — not memorization and not marginal noise.
        if args.vgate_with_min > 0:
            n_before = len(chosen)
            chosen = [
                s for s in chosen
                if (s.get("expected_gain", {}).get("v_avg_with", 0) >= args.vgate_with_min
                    and args.vgate_without_min
                    <= s.get("expected_gain", {}).get("v_avg_without", 0)
                    <= args.vgate_without_max)
            ]
            n_after = len(chosen)
            if n_before != n_after:
                print(f"  [vgate] task {task_id}: {n_before}→{n_after} skills after V-calibrated gate")

        # Cost-calibrated gate: use training source step index as a proxy for
        # how late/expensive the skill tends to appear. This is training-only
        # metadata, no test leakage.
        if args.max_source_step_avg and args.max_source_step_avg > 0 and chosen:
            before = len(chosen)
            chosen = [s for s in chosen if source_step_avg(s) <= args.max_source_step_avg]
            if before != len(chosen):
                print(f"  [step-gate] task {task_id}: {before}→{len(chosen)} skills after source-step gate")

        if args.cost_lambda_step and args.cost_lambda_step > 0 and chosen:
            before = len(chosen)
            kept = []
            for s in chosen:
                eg = s.get("expected_gain", {}) or {}
                gain = eg.get("gain", 0) or (eg.get("v_avg_with", 0) - eg.get("v_avg_without", 0))
                utility = gain - args.cost_lambda_step * (source_step_avg(s) / 10.0)
                if utility >= args.min_cost_utility:
                    kept.append(s)
            chosen = kept
            if before != len(chosen):
                print(f"  [cost-gate] task {task_id}: {before}→{len(chosen)} skills after cost utility gate")

        # Optional top-1 trim when the leading skill is very high-confidence.
        if args.high_conf_top1 and chosen:
            top = chosen[0]
            vw = top.get("expected_gain", {}).get("v_avg_with", 0)
            if vw >= 0.95:
                chosen = chosen[:1]

        topics = retriever.topic_context_for(chosen)
        domain_for_inj = "" if args.disable_domain_instruction else domain_name
        injection = build_injection(chosen, topics, domain=domain_for_inj,
                                    no_worked_examples=args.no_worked_examples,
                                    compact_skill=args.compact_skill,
                                    no_topic_context=args.no_topic_context,
                                    watchout_only=args.watchout_only,
                                    card_mode=args.card_mode)
        # Optional cost-aware turn budget hint, attached to every task.
        budget_hint = ""
        if args.turn_budget and args.turn_budget > 0:
            budget_hint = (
                "## Cost Budget\n"
                f"Solve this task in **at most {args.turn_budget} tool-call turns**. "
                "Each unnecessary turn is costly. If you can answer with no tool calls, do so. "
                "Avoid exploration that is not needed for the final answer.\n"
            )
        if injection and budget_hint:
            t["skill_text"] = budget_hint + "\n" + injection
            n_inject += 1
        elif injection:
            t["skill_text"] = injection
            n_inject += 1
        elif budget_hint:
            t["skill_text"] = budget_hint
        debug.append({
            "task_id": str(task_id),
            "intent_tags": profile.intent_tags,
            "artifact_tags": profile.artifact_tags,
            "chosen_skills": [s["skill_id"] for s in chosen],
            "fused_top": [sid for sid, _ in dbg["fused_top"][:5]],
            "n_dropped": len(dbg["dropped"]),
        })
    print(f"[msce] injected on {n_inject}/{len(tasks)} tasks "
          f"(others run baseline)")

    patch_domain_build_prompt(domain)

    # Job dir + debug dump
    job_name = args.job or f"msce-{domain_name}-{datetime.now().strftime('%m%d_%H%M')}-{uuid.uuid4().hex[:4]}"
    job_dir = Path(cfg["job_dir"]) / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "v3_inject_log.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in debug)
    )
    (job_dir / "v3_config.json").write_text(json.dumps({
        "skill_bank": args.skill_bank,
        "skills": len(retriever.skills),
        "topics": len(retriever.topics),
        "top_k": args.top_k, "min_gain": args.min_gain,
        "per_route_topn": args.per_route_topn,
        "disable_topic": args.disable_topic,
        "disable_bm25": args.disable_bm25,
        "disable_dense": args.disable_dense,
        "vgate_with_min": args.vgate_with_min,
        "vgate_without_min": args.vgate_without_min,
        "vgate_without_max": args.vgate_without_max,
        "no_worked_examples": args.no_worked_examples,
        "compact_skill": args.compact_skill,
        "no_topic_context": args.no_topic_context,
        "high_conf_top1": args.high_conf_top1,
        "turn_budget": args.turn_budget,
        "watchout_only": args.watchout_only,
        "card_mode": args.card_mode,
        "max_source_step_avg": args.max_source_step_avg,
        "cost_lambda_step": args.cost_lambda_step,
        "min_cost_utility": args.min_cost_utility,
    }, ensure_ascii=False, indent=2))

    run_args = argparse.Namespace(
        task=",".join(str(t["name"]) for t in tasks),
        split=None,
        trials=cfg.get("trials", 1),
        parallel=args.parallel,
        max_retries=cfg.get("max_retries", 0),
        live=args.live,
        disk_budget=None,
    )
    run_all(tasks, domain, agent, job_dir, run_args)


if __name__ == "__main__":
    main()

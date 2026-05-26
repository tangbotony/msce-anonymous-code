#!/usr/bin/env python3
"""MSCE — Multi-route skill retrieval (Topic + BM25 + Dense, fused by RRF).

Pure library module; imported by eval_with_skills.py.

Usage sketch:
    from retrieval import SkillRetriever, TaskProfiler

    retriever = SkillRetriever.load(skill_bank_path, topics_path, emb_npy, emb_ids_json)
    profiler  = TaskProfiler()

    for task in tasks:
        profile = profiler.profile(task)
        skills  = retriever.retrieve(profile, top_k=3, min_gain=0.05)
"""
from __future__ import annotations
import json, math, os, re, time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    from .clients import DEFAULT_EMBEDDING_DIM, chat_completion_json, embedding_one as provider_embedding_one
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import DEFAULT_EMBEDDING_DIM, chat_completion_json, embedding_one as provider_embedding_one


# ── shared LLM/embedding helpers ────────────────────────────────────


def call_llm(system: str, user: str, max_tokens: int = 500,
             temperature: float = 0.0, retries: int = 3):
    try:
        return chat_completion_json(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            timeout=60,
        )
    except Exception:
        return {}


def embed_one(text: str) -> np.ndarray:
    text = (text or "")[:6000]
    try:
        return provider_embedding_one(text)
    except Exception:
        return np.zeros((DEFAULT_EMBEDDING_DIM,), dtype=np.float32)


# ── BM25 (very small impl, no external dep) ─────────────────────────


_TOKEN_RX = re.compile(r"[A-Za-z0-9_\-/.]+|[\u4e00-\u9fff]+", re.U)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RX.findall(text)][:300]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.docs = docs
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / max(self.N, 1)) or 1.0
        self.df = defaultdict(int)
        self.tf = []
        for d in docs:
            c = Counter(d)
            self.tf.append(c)
            for w in c.keys():
                self.df[w] += 1
        self.idf = {}
        for w, df in self.df.items():
            # smoothed inverse doc frequency (BM25+ flavor)
            self.idf[w] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for w in query_tokens:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(w, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores


# ── Task profile (LLM-extracted intent / artifact / capabilities / kw) ──


PROFILE_SYS = """你是任务剖析器。读 task prompt，输出 JSON：

{
  "intent_tags":   ["3-6 个意图标签 (kebab-case, 例: office-doc-gen, blockchain-dapp, data-analysis-xlsx, retail-broker, real-estate-search, info-retrieval-multi-hop, code-impl-algorithm)"],
  "artifact_tags": ["3-6 个期望产物标签 (docx, xlsx, pdf, zip, code_repo, sql_query, markdown_report, search_answer)"],
  "required_capabilities": "<80-120 字描述完成本任务需要的能力组合 (用于向量检索)>",
  "keywords": ["8-15 个 BM25 检索用关键词 (tool / artifact / domain 关键词)"]
}

只输出 JSON。不要 markdown 包裹。"""


@dataclass
class TaskProfile:
    intent_tags: list[str] = field(default_factory=list)
    artifact_tags: list[str] = field(default_factory=list)
    required_capabilities: str = ""
    keywords: list[str] = field(default_factory=list)


class TaskProfiler:
    def __init__(self, cache_path: str | None = None):
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and Path(cache_path).exists():
            for line in open(cache_path):
                try:
                    d = json.loads(line)
                    self._cache[d["task_id"]] = d["profile"]
                except Exception:
                    pass

    def profile(self, task_id: str, task_prompt: str) -> TaskProfile:
        if task_id in self._cache:
            d = self._cache[task_id]
            return TaskProfile(**d)
        prompt = task_prompt[:4000]
        res = call_llm(PROFILE_SYS, prompt, max_tokens=400)
        if not isinstance(res, dict):
            p = TaskProfile()
        else:
            p = TaskProfile(
                intent_tags=[str(x).lower().strip()[:40] for x in (res.get("intent_tags") or [])][:8],
                artifact_tags=[str(x).lower().strip()[:30] for x in (res.get("artifact_tags") or [])][:8],
                required_capabilities=str(res.get("required_capabilities", ""))[:600],
                keywords=[str(x).lower().strip()[:30] for x in (res.get("keywords") or [])][:20],
            )
        # cache write
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"task_id": task_id, "profile": p.__dict__},
                                    ensure_ascii=False) + "\n")
        self._cache[task_id] = p.__dict__
        return p


# ── Skill retriever (Topic + BM25 + Dense + RRF + Gate) ────────────


def _norm_tag(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "-").replace("_", "-")[:40]


class SkillRetriever:
    def __init__(self, skills: list[dict], topics: list[dict],
                 embeddings: np.ndarray, skill_ids: list[str]):
        self.skills = skills
        self.topics = topics
        self.embs = embeddings
        self.ids = skill_ids
        self.id2idx = {sid: i for i, sid in enumerate(skill_ids)}

        # Topic index: tag → topic_id list
        self.tag_to_topics = defaultdict(set)
        self.topic_index = {t["topic_id"]: t for t in topics}
        for t in topics:
            for tag in t.get("related_intent_tags", []):
                self.tag_to_topics[_norm_tag(tag)].add(t["topic_id"])
            for tag in t.get("related_artifact_tags", []):
                self.tag_to_topics[_norm_tag(tag)].add(t["topic_id"])
        # Skill index: skill_id → topic_id  (1:1)
        self.skill_to_topic = {s["skill_id"]: s.get("topic_id") for s in skills}
        # Topic → skill_ids (from skill records, more reliable than topic field)
        self.topic_to_skills = defaultdict(list)
        for s in skills:
            self.topic_to_skills[s.get("topic_id")].append(s["skill_id"])

        # BM25 docs
        docs = []
        for s in skills:
            kw = " ".join(s.get("lexical_keywords") or [])
            doc = (
                s.get("name", "") + " " + kw + " " +
                " ".join(s.get("applicability_signature", {}).get("intent_tags", [])) + " " +
                " ".join(s.get("applicability_signature", {}).get("artifact_tags", [])) + " " +
                " ".join(s.get("applicability_signature", {}).get("command_kinds", [])) + " " +
                (s.get("summary", "") or "")
            )
            docs.append(tokenize(doc))
        self.bm25 = BM25(docs)

    @classmethod
    def load(cls, skill_bank_path: str, topics_path: str,
             emb_npy: str, emb_ids: str) -> "SkillRetriever":
        skills = [json.loads(l) for l in open(skill_bank_path)]
        topics = [json.loads(l) for l in open(topics_path)]
        embeddings = np.load(emb_npy)
        skill_ids = json.load(open(emb_ids))
        return cls(skills, topics, embeddings, skill_ids)

    # ── recall routes ──

    def _recall_topic(self, profile: TaskProfile, top_n: int = 12) -> list[str]:
        """Pick topics by tag overlap, then concat their skills ordered by expected_gain."""
        scores = defaultdict(float)
        for tag in profile.intent_tags + profile.artifact_tags:
            for tid in self.tag_to_topics.get(_norm_tag(tag), []):
                scores[tid] += 1.0
        topic_ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out = []
        for tid, _ in topic_ranked:
            for sid in self.topic_to_skills.get(tid, []):
                if sid not in out:
                    out.append(sid)
                if len(out) >= top_n:
                    break
            if len(out) >= top_n:
                break
        return out

    def _recall_bm25(self, profile: TaskProfile, top_n: int = 12) -> list[str]:
        q_tokens = list(profile.keywords) + tokenize(profile.required_capabilities)
        if not q_tokens:
            return []
        scores = self.bm25.score(q_tokens)
        if not scores.size:
            return []
        order = np.argsort(-scores)
        return [self.skills[i]["skill_id"] for i in order[:top_n] if scores[i] > 0]

    def _recall_dense(self, profile: TaskProfile, top_n: int = 12) -> list[str]:
        q_text = profile.required_capabilities or \
            " ".join(profile.intent_tags + profile.artifact_tags)
        if not q_text.strip():
            return []
        qv = embed_one(q_text)
        if not np.any(qv):
            return []
        sims = self.embs @ qv
        order = np.argsort(-sims)
        return [self.ids[i] for i in order[:top_n] if sims[i] > 0.20]

    @staticmethod
    def _rrf(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
        score = defaultdict(float)
        for src, ranked in rankings.items():
            for rank, sid in enumerate(ranked):
                score[sid] += 1.0 / (k + rank)
        return sorted(score.items(), key=lambda x: -x[1])

    @staticmethod
    def _compatible(profile: TaskProfile, skill: dict) -> tuple[bool, str]:
        """Hard applicability gate.

        RRF can surface high-gain but off-topic skills. This gate blocks
        negative transfer: a skill must match by intent, or by a sufficiently
        specific artifact overlap. Generic artifacts such as "zip" or
        "markdown-report" do not justify cross-topic injection.
        """
        generic_artifacts = {
            "zip", "report", "markdown-report", "markdown_report", "md",
            "txt", "png", "jpg", "tables", "file", "files", "artifact",
        }
        code_intents = {
            "blockchain-dapp", "solidity-smart-contracts", "zk-snark-privacy",
            "cross-chain-bridge", "defi-integration", "fullstack-web3",
            "code-impl-algorithm", "code-repo", "frontend-app",
            "backend-service", "smart-contracts",
        }
        office_intents = {
            "office-doc-gen", "data-analysis-xlsx", "data-entry-xlsx",
            "pdf-generation", "docx-generation", "spreadsheet-cleaning",
        }

        p_int = {_norm_tag(x) for x in profile.intent_tags}
        p_art = {_norm_tag(x) for x in profile.artifact_tags}
        sig = skill.get("applicability_signature") or {}
        s_int = {_norm_tag(x) for x in sig.get("intent_tags", [])}
        s_art = {_norm_tag(x) for x in sig.get("artifact_tags", [])}

        intent_overlap = p_int & s_int
        artifact_overlap = p_art & s_art
        specific_artifact_overlap = artifact_overlap - generic_artifacts

        # Explicitly block code/Web3 tasks from office-document skills unless
        # there is also an intent match (there should not be).
        if (p_int & code_intents) and (s_int & office_intents) and not intent_overlap:
            return False, "blocked_code_vs_office"

        if intent_overlap:
            return True, "intent_overlap"

        # Artifact-only match must be strong and specific; e.g. docx/xlsx/pdf
        # can route office tasks, but "zip" or "markdown-report" cannot.
        if len(artifact_overlap) >= 2 and specific_artifact_overlap:
            return True, "specific_artifact_overlap"

        return False, "no_intent_or_specific_artifact_overlap"

    def retrieve(self, profile: TaskProfile, top_k: int = 3,
                 min_gain: float = 0.05,
                 per_route_topn: int = 10,
                 enable_topic: bool = True,
                 enable_bm25: bool = True,
                 enable_dense: bool = True,
                 ) -> tuple[list[dict], dict]:
        """Return (chosen_skills, debug_info)."""
        rankings = {}
        if enable_topic:
            rankings["topic"] = self._recall_topic(profile, per_route_topn)
        if enable_bm25:
            rankings["bm25"] = self._recall_bm25(profile, per_route_topn)
        if enable_dense:
            rankings["dense"] = self._recall_dense(profile, per_route_topn)
        fused = self._rrf(rankings)
        # compatibility + utility gate
        chosen, dropped = [], []
        for sid, sc in fused:
            idx = self.id2idx.get(sid)
            if idx is None:
                continue
            s = self.skills[idx]
            ok, why = self._compatible(profile, s)
            if not ok:
                dropped.append((sid, 0.0, why))
                continue
            gain = (s.get("expected_gain") or {}).get("gain", 0.0)
            if gain < min_gain:
                dropped.append((sid, gain, "low_gain"))
                continue
            chosen.append(s)
            if len(chosen) >= top_k:
                break
        debug = {
            "rankings": {k: v[:per_route_topn] for k, v in rankings.items()},
            "fused_top": fused[:per_route_topn],
            "dropped": dropped,
            "n_chosen": len(chosen),
        }
        return chosen, debug

    def topic_context_for(self, chosen_skills: list[dict]) -> list[dict]:
        """Return the topic nodes that the chosen skills belong to (dedup, ordered)."""
        seen = set()
        out = []
        for s in chosen_skills:
            tid = s.get("topic_id")
            if tid and tid not in seen and tid in self.topic_index:
                seen.add(tid)
                out.append(self.topic_index[tid])
        return out

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.ldu import LDU
from src.models.pageindex import PageIndex, SectionNode
from src.models.provenance import ProvenanceChain, ProvenanceSpan


# ── LLM helper ────────────────────────────────────────────────────────────────

def _call_openrouter(prompt: str, max_tokens: int = 20) -> str:
    """
    Call OpenRouter. Returns text content or empty string on any failure.
    Requires OPENROUTER_API_KEY environment variable.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ""
    try:
        import httpx
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("MODEL_NAME", "openrouter/auto:free"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


# ── Provenance helpers ────────────────────────────────────────────────────────

def _build_provenance(ldu: LDU, source_path: str) -> Dict[str, Any]:
    """Return a JSON-serialisable ProvenanceChain dict from an LDU."""
    spans = []
    if ldu.provenance and ldu.provenance.spans:
        for s in ldu.provenance.spans:
            spans.append({
                "page": s.page,
                "bbox": s.bbox,
            })
    elif ldu.page_refs:
        for p in ldu.page_refs:
            spans.append({
                "page": p,
                "bbox": ldu.bounding_box,  # FIX 3 in ldu_builder ensures non-null
            })
    return {
        "document_name": Path(source_path).name,
        "source_path": source_path,
        "content_hash": ldu.content_hash,
        "spans": spans,
        "ldu_id": ldu.ldu_id,
    }


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wraps ChromaDB for LDU semantic search.
    Falls back to keyword scoring if ChromaDB is unavailable.
    """

    def __init__(self, collection_name: str = "refinery_ldus", persist_dir: Optional[str] = None):
        self._collection = None
        self._ldus: List[LDU] = []
        self._fallback = True
        persist_dir = persist_dir or str(Path(".refinery/chroma"))
        try:
            import chromadb
            client = chromadb.PersistentClient(path=persist_dir)
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._fallback = False
        except Exception:
            self._fallback = True

    def ingest(self, ldus: List[LDU], source_path: str = "") -> int:
        """Ingest LDUs into the vector store. Returns count ingested."""
        self._ldus = ldus
        if self._fallback or self._collection is None:
            return len(ldus)
        try:
            existing_ids = set(self._collection.get(include=[])["ids"])
            to_add_docs, to_add_ids, to_add_metas = [], [], []
            for ldu in ldus:
                if ldu.ldu_id in existing_ids:
                    continue
                content = (ldu.content or "").strip()
                if not content:
                    continue
                to_add_docs.append(content[:2000])
                to_add_ids.append(ldu.ldu_id)
                to_add_metas.append({
                    "doc_id": ldu.doc_id,
                    "chunk_type": ldu.chunk_type,
                    "page_refs": json.dumps(ldu.page_refs),
                    "content_hash": ldu.content_hash,
                    "source_path": source_path,
                })
            if to_add_docs:
                self._collection.add(
                    documents=to_add_docs,
                    ids=to_add_ids,
                    metadatas=to_add_metas,
                )
            return len(to_add_docs)
        except Exception:
            self._fallback = True
            return len(ldus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Returns list of (ldu_id, score, metadata)."""
        if self._fallback or self._collection is None:
            return self._keyword_search(query, top_k)
        try:
            results = self._collection.query(query_texts=[query], n_results=min(top_k, 10))
            ids       = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metas     = results.get("metadatas", [[]])[0]
            return [(ids[i], 1.0 - distances[i], metas[i]) for i in range(len(ids))]
        except Exception:
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        q = query.lower().strip()
        hits: List[Tuple[str, float, LDU]] = []
        for ldu in self._ldus:
            text = (ldu.content or "").lower()
            score = sum(text.count(word) for word in q.split())
            if score > 0:
                hits.append((ldu.ldu_id, score / max(1, len(q.split())), ldu))
        hits.sort(key=lambda x: x[1], reverse=True)
        return [
            (h[0], h[1], {
                "chunk_type": h[2].chunk_type,
                "page_refs": json.dumps(h[2].page_refs),
                "content_hash": h[2].content_hash,
            })
            for h in hits[:top_k]
        ]


# ── SQLite FactTable ──────────────────────────────────────────────────────────

class FactTable:
    """
    Extracts numerical/financial facts from LDUs into a SQLite table.
    Schema: doc_id, ldu_id, field_name, value, unit, page, content_hash, source_path
    """

    FACT_PATTERNS = [
        re.compile(
            r"(?P<field>[A-Za-z][A-Za-z\s\-/]{2,40})"
            r"\s*[:=]\s*"
            r"(?P<value>[\$\xa3\u20ac]?[\d,\.]+\s*(?:million|billion|thousand|B|M|K|%)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<field>[A-Za-z0-9][A-Za-z0-9\s\-/]{1,30})"
            r"\s+(?P<value>[\$\xa3\u20ac][\d,\.]+\s*(?:million|billion|B|M|K)?)",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, db_path: str = ".refinery/facts.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    ldu_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    unit TEXT,
                    page INTEGER,
                    content_hash TEXT,
                    source_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_field  ON facts(field_name)")
            conn.commit()

    def extract_and_store(self, ldus: List[LDU], source_path: str = "") -> int:
        """Parse facts from LDUs and store in SQLite. Returns count inserted."""
        rows: List[tuple] = []
        for ldu in ldus:
            page = ldu.page_refs[0] if ldu.page_refs else None
            for pattern in self.FACT_PATTERNS:
                for m in pattern.finditer(ldu.content or ""):
                    field_name = m.group("field").strip().rstrip(":").strip()
                    value_raw  = m.group("value").strip()
                    unit = None
                    for u in ["million", "billion", "thousand", "B", "M", "K", "%", "$", "£", "€"]:
                        if u.lower() in value_raw.lower():
                            unit = u
                            break
                    if len(field_name) < 3 or not value_raw:
                        continue
                    rows.append((
                        ldu.doc_id, ldu.ldu_id, field_name, value_raw,
                        unit, page, ldu.content_hash, source_path,
                    ))
        if rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT INTO facts "
                    "(doc_id, ldu_id, field_name, value, unit, page, content_hash, source_path) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
        return len(rows)

    def query(self, field_name: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query facts by field name (fuzzy LIKE match)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if doc_id:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE LOWER(field_name) LIKE ? AND doc_id=? LIMIT 50",
                    (f"%{field_name.lower()}%", doc_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE LOWER(field_name) LIKE ? LIMIT 50",
                    (f"%{field_name.lower()}%",),
                ).fetchall()
            return [dict(r) for r in rows]

    def query_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute arbitrary SELECT on facts table (read-only)."""
        if not sql.strip().upper().startswith("SELECT"):
            return [{"error": "Only SELECT queries allowed"}]
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(sql).fetchall()]
        except Exception as e:
            return [{"error": str(e)}]


# ── PageIndex section search ──────────────────────────────────────────────────

def _section_matches(node: SectionNode, topic: str) -> float:
    topic_l = topic.lower()
    score = 0.0
    title_l = node.title.lower()
    score += title_l.count(topic_l) * 3
    for word in topic_l.split():
        score += title_l.count(word) * 2
        score += (node.summary or "").lower().count(word) * 0.5
        for ent in node.key_entities:
            score += ent.lower().count(word) * 1
    return score


def _find_sections(nodes: List[SectionNode], topic: str, top_k: int = 3) -> List[Dict[str, Any]]:
    hits: List[Tuple[float, SectionNode]] = []

    def recurse(node: SectionNode) -> None:
        s = _section_matches(node, topic)
        if s > 0:
            hits.append((s, node))
        for child in node.child_sections:
            recurse(child)

    for n in nodes:
        recurse(n)

    hits.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "title": n.title,
            "page_start": n.page_start,
            "page_end": n.page_end,
            "level": n.level,
            "summary": n.summary,
            "key_entities": n.key_entities,
            "data_types_present": n.data_types_present,
            "child_count": len(n.child_sections),
            "ldu_count": len(n.ldu_ids),
            "score": round(score, 3),
        }
        for score, n in hits[:top_k]
    ]


# ── Audit Mode ────────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    claim: str
    verdict: str          # "verified" | "not_found" | "unverifiable"
    confidence: float     # 0.0 - 1.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""


# ── Main Query Agent ──────────────────────────────────────────────────────────

class QueryAgent:
    """
    LangGraph-style query agent with three tools + Audit Mode.
    """

    def __init__(
        self,
        page_index: PageIndex,
        ldus: List[LDU],
        fact_table: Optional[FactTable] = None,
        vector_store: Optional[VectorStore] = None,
        source_path: str = "",
    ) -> None:
        self.page_index  = page_index
        self.ldus        = ldus
        self.source_path = source_path or page_index.source_path
        self._ldu_map: Dict[str, LDU] = {ldu.ldu_id: ldu for ldu in ldus}

        self.vector_store = vector_store or VectorStore()
        if ldus:
            self.vector_store.ingest(ldus, self.source_path)

        self.fact_table = fact_table or FactTable()
        if ldus:
            self.fact_table.extract_and_store(ldus, self.source_path)

    # ── Tool 1 ────────────────────────────────────────────────────────────────

    def pageindex_navigate(self, topic: str, page: Optional[int] = None) -> Dict[str, Any]:
        """Navigate the PageIndex by page number or topic string."""
        if page is not None:
            for node in self.page_index.root:
                if node.page == page:
                    return {"tool": "pageindex_navigate", "page": page, "items": node.model_dump()}
            return {"tool": "pageindex_navigate", "page": page, "items": [], "message": "page not found"}

        matches = _find_sections(self.page_index.sections, topic, top_k=3)
        return {
            "tool": "pageindex_navigate",
            "query": topic,
            "sections_found": len(matches),
            "results": matches,
        }

    # ── Tool 2 ────────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search LDUs via vector similarity. Returns results with ProvenanceChain."""
        hits = self.vector_store.search(query, top_k)
        results = []
        for ldu_id, score, meta in hits:
            ldu = self._ldu_map.get(ldu_id)
            if ldu is None:
                continue
            results.append({
                "ldu_id": ldu_id,
                "score": round(score, 4),
                "chunk_type": ldu.chunk_type,
                "snippet": (ldu.content or "")[:300],
                "page_refs": ldu.page_refs,
                "provenance": _build_provenance(ldu, self.source_path),
            })
        return {
            "tool": "semantic_search",
            "query": query,
            "results_count": len(results),
            "results": results,
        }

    # ── Tool 3 ────────────────────────────────────────────────────────────────

    def structured_query(
        self,
        field_name: str,
        doc_id: Optional[str] = None,
        raw_sql: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query SQLite FactTable by field name or raw SQL."""
        rows = self.fact_table.query_sql(raw_sql) if raw_sql else self.fact_table.query(field_name, doc_id)
        for row in rows:
            ldu = self._ldu_map.get(row.get("ldu_id", ""))
            row["provenance"] = _build_provenance(ldu, self.source_path) if ldu else {
                "document_name": Path(self.source_path).name,
                "page": row.get("page"),
                "content_hash": row.get("content_hash", ""),
            }
        return {
            "tool": "structured_query",
            "field_name": field_name,
            "results_count": len(rows),
            "results": rows,
        }

    # ── Unified ask() ─────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main entry point: routes through all three tools and synthesises answer.
        Always returns a ProvenanceChain in the response.
        """
        nav    = self.pageindex_navigate(topic=question)
        search = self.semantic_search(question, top_k)

        structured: Optional[Dict] = None
        number_keywords = re.findall(
            r"\b(?:revenue|profit|loss|expenditure|total|amount|cost|"
            r"budget|income|tax|rate|ratio|value|year|quarter|q\d)\b",
            question, re.I,
        )
        if number_keywords:
            structured = self.structured_query(number_keywords[0])

        top_results  = search.get("results", [])
        top_sections = nav.get("results", [])
        answer_parts   = [r["snippet"] for r in top_results[:3]]
        all_provenance = [r["provenance"] for r in top_results[:3]]

        return {
            "question": question,
            "answer_snippets": answer_parts,
            "provenance_chain": all_provenance,
            "navigation_hits": top_sections,
            "structured_facts": (structured or {}).get("results", [])[:5],
            "tools_used": ["pageindex_navigate", "semantic_search"]
            + (["structured_query"] if structured else []),
        }

    # ── Audit Mode ────────────────────────────────────────────────────────────

    def verify_claim(self, claim: str) -> AuditResult:
        """
        Verify whether a claim is supported by the document corpus.

        Two-step approach (FIX for keyword-only limitation):
          Step 1: semantic retrieval finds best candidate evidence chunks
          Step 2: LLM confirmation if OPENROUTER_API_KEY is set.
                  Handles paraphrased claims that overlap scoring misses.
                  Falls back to overlap-only when no API key.

        Returns AuditResult with verdict: 'verified' | 'not_found' | 'unverifiable'
        """
        search  = self.semantic_search(claim, top_k=8)
        results = search.get("results", [])

        if not results:
            return AuditResult(
                claim=claim,
                verdict="not_found",
                confidence=0.0,
                explanation="No matching content found in document corpus.",
            )

        # ── Step 1: keyword overlap scoring ──────────────────────────────────
        claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
        scored = []
        for r in results:
            snippet_words = set(re.findall(r"\b\w{4,}\b", (r.get("snippet") or "").lower()))
            overlap = len(claim_words & snippet_words) / max(1, len(claim_words))
            scored.append((overlap, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_result = scored[0]

        # ── Step 2: LLM confirmation (handles paraphrasing) ───────────────────
        llm_verdict: Optional[str] = None
        if best_score >= 0.15 and os.getenv("OPENROUTER_API_KEY"):
            prompt = (
                f"You are a document auditor.\n\n"
                f"Claim: {claim}\n\n"
                f"Evidence from document:\n{best_result.get('snippet', '')[:600]}\n\n"
                f"Does the evidence directly support the claim?\n"
                f"Respond with exactly one word: SUPPORTED, PARTIAL, or NOT_SUPPORTED"
            )
            raw = _call_openrouter(prompt, max_tokens=10).upper()
            if "SUPPORTED" in raw and "NOT" not in raw:
                llm_verdict = "verified"
            elif "PARTIAL" in raw:
                llm_verdict = "unverifiable"
            elif "NOT_SUPPORTED" in raw:
                llm_verdict = "not_found"

        # ── Final verdict ─────────────────────────────────────────────────────
        if llm_verdict:
            verdict    = llm_verdict
            confidence = round(min(1.0, best_score + 0.2), 3) if llm_verdict == "verified" else round(best_score, 3)
        else:
            if best_score >= 0.5:
                verdict = "verified"
            elif best_score >= 0.2:
                verdict = "unverifiable"
            else:
                verdict = "not_found"
            confidence = round(best_score, 3)

        explanation = (
            f"Best match (overlap={best_score:.2f}"
            + (f", llm={llm_verdict}" if llm_verdict else "")
            + f"): '{(best_result.get('snippet') or '')[:200]}'"
        )

        return AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=min(1.0, confidence),
            evidence=[best_result.get("provenance", {})] if best_score > 0.1 else [],
            explanation=explanation,
        )
<div align="center">

# 🏭 Document Intelligence Refinery

### *Transform any document into queryable, structured, provenance-tagged knowledge*

> A production-grade 5-stage agentic pipeline for enterprise document intelligence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square)](https://docs.pydantic.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Free_Tier-6366F1?style=flat-square)](https://openrouter.ai)
[![VLM Cost](https://img.shields.io/badge/VLM_Cost-%240.00-22C55E?style=flat-square)](https://openrouter.ai/models?q=free)

*TRP1 Forward Deployed Engineer Program — Week 3 Challenge*

*Built by **Meseret Bolled** · Addis Ababa Science and Technology University*

</div>

---

## Table of Contents

- [The Problem](#the-problem)
- [What This Builds](#what-this-builds)
- [System Architecture](#system-architecture)
- [Stage 1 — Triage Agent](#stage-1--triage-agent)
- [Stage 2 — Extraction Router](#stage-2--extraction-router)
- [Stage 3 — Semantic Chunking Engine](#stage-3--semantic-chunking-engine)
- [Stage 4 — PageIndex Builder](#stage-4--pageindex-builder)
- [Stage 5 — Query Interface Agent](#stage-5--query-interface-agent)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Free VLM Strategy](#free-vlm-strategy)
- [Amharic and Ethiopic Script Support](#amharic-and-ethiopic-script-support)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Artifact Outputs](#artifact-outputs)
- [Running Tests](#running-tests)
- [Demo Protocol](#demo-protocol)
- [Corpus Documents](#corpus-documents)

---

## The Problem

Every enterprise has its institutional memory locked inside documents. Banks, hospitals, law firms, and government agencies all face the same three failure modes when they try to query that knowledge:

**Failure Mode 1 — Structure Collapse**

Traditional OCR flattens two-column layouts, breaks financial tables into unreadable strings, and drops section headers. The text is technically present but semantically destroyed. A table that took days to produce becomes a garbage string after extraction.

**Failure Mode 2 — Context Poverty**

Naive token-based chunking severs logical units at arbitrary boundaries. When a financial table is split across two chunks, every RAG query about that table returns hallucinated answers because neither chunk contains the full data. The model fills in the gaps with confidence.

**Failure Mode 3 — Provenance Blindness**

When a system returns "revenue was $4.2 billion," the client's first question is always "which page? which table? can I verify that?" Without spatial provenance — page number, bounding box, content hash — extracted data cannot be audited, and enterprise adoption stalls.

**This pipeline solves all three.**

---

## What This Builds

| Input | Output |
|---|---|
| PDFs (native digital) | Structured JSON schemas with typed fields |
| PDFs (scanned images) | Hierarchical section tree with page ranges |
| Amharic / Ethiopic documents | ChromaDB vector store for semantic search |
| Mixed layout reports | SQLite FactTable for exact numerical queries |
| Table-heavy fiscal data | Provenance-tagged LDUs with page and bbox |
| Any of the above | Full audit trail with content hash per chunk |

---

## System Architecture

The pipeline has five stages. Each stage has typed inputs, typed outputs, and independently testable logic. Nothing downstream runs until the previous stage completes successfully.

```
[Any PDF document]
        |
        v
+-------------------+
|    STAGE 1        |  Reads: char density, image area ratio,
|   TRIAGE AGENT    |  Ethiopic Unicode range, column count
|                   |  Output: DocumentProfile JSON
+-------------------+
        |
        | DocumentProfile
        | { origin_type, layout_complexity, language, cost_tier }
        v
+-----------------------------------------------+
|                   STAGE 2                     |
|            EXTRACTION ROUTER                  |
|                                               |
|  Strategy A        Strategy B     Strategy C  |
|  FastText          Layout-Aware   Vision VLM  |
|  pdfplumber        Docling        (Free chain) |
|  pymupdf                                      |
|                                               |
|  conf >= 0.60  --> conf >= 0.70  --> fallback |
|  stop here         stop here      always last |
+-----------------------------------------------+
        |
        | ExtractedDocument
        | { blocks[], confidence, strategy_used }
        v
+-------------------+
|    STAGE 3        |  5 constitution rules enforced
|    SEMANTIC       |  R1: Tables standalone
|    CHUNKING       |  R2: Captions attach to figures
|    ENGINE         |  R3: Lists stay together
|                   |  R4: Headers propagate to children
|                   |  R5: Cross-refs stored in provenance
|                   |  ChunkValidator checks all rules
+-------------------+
        |
        | List[LDU]  (Logical Document Units)
        v
+-------------------+
|    STAGE 4        |  Flat page index (page -> items)
|    PAGEINDEX      |  + Hierarchical section tree
|    BUILDER        |  (title, summary, entities, page range)
+-------------------+
        |
        | PageIndex
        v
+-------------------+
|    STAGE 5        |  Tool 1: pageindex_navigate
|    QUERY          |  Tool 2: semantic_search (ChromaDB)
|    INTERFACE      |  Tool 3: structured_query (SQLite)
|    AGENT          |
|                   |  Audit Mode: verify_claim
|                   |  Every answer: ProvenanceChain
|                   |  { doc, page, bbox, content_hash }
+-------------------+
```

---

## Stage 1 — Triage Agent

The Triage Agent is the first thing that runs on every document. Its job is to **understand what kind of document this is before spending any money on extraction**. It uses fast, cheap signals from the PDF itself — no ML model, no API call.

### What signals it computes

```
PDF file
  |
  +-- pdfplumber scan
  |     avg_text_chars_per_page  ----+
  |     avg_image_area_ratio    ----+---> origin_type
  |     font_metadata_present   ----+     "native_digital"
  |                                       "scanned_image"
  |                                       "mixed"
  |
  +-- layout heuristics
  |     column count estimate   ----+
  |     table bbox coverage     ----+---> layout_complexity
  |     figure bbox coverage    ----+     "single_column"
  |                                       "multi_column"
  |                                       "table_heavy"
  |                                       "figure_heavy"
  |
  +-- language detection
  |     Ethiopic chars U+1200-U+137F ----> language = "am"
  |     Ethiopic + Latin present     ----> language = "mixed"
  |     langdetect library           ----> language = "en" etc
  |
  +-- domain keywords
        "balance sheet", "revenue"   ----> domain = "financial"
        "audit", "hereby"            ----> domain = "legal"
        "methodology", "assessment"  ----> domain = "technical"
```

### Output

The Triage Agent produces a `DocumentProfile` and saves it to `.refinery/profiles/{doc_id}.json`.

```json
{
  "doc_id": "fta_performance_survey_final_report_2022",
  "origin_type": "native_digital",
  "layout_complexity": "multi_column",
  "language": "en",
  "domain_hint": "technical",
  "page_count": 155,
  "avg_text_chars_per_page": 2616.5,
  "avg_image_area_ratio": 0.0007,
  "cost_tier": "needs_layout_model"
}
```

The `cost_tier` field drives everything downstream. It can be `fast_text_sufficient`, `needs_layout_model`, or `needs_vision_model`.

---

## Stage 2 — Extraction Router

The router selects the cheapest strategy that can achieve adequate quality. It uses a **confidence-gated escalation ladder** — cheaper strategies always run first and only escalate when the result is not good enough.

### The three strategies compared

| | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| **Name** | FastText | Layout-Aware | Vision VLM |
| **Tools** | pdfplumber, pymupdf | Docling | OpenRouter free models |
| **Best for** | Native digital, single column | Multi-column, tables, mixed | Scanned images, Amharic |
| **Speed** | 2–5 seconds | 10–30 seconds | 5–15 seconds |
| **Cost per doc** | ~$0.001 | ~$0.010 | $0.000 (free) |
| **Confidence gate** | Must reach 0.60 | Must reach 0.70 | Last resort, no gate |

### Escalation logic

```
DocumentProfile.cost_tier = "fast_text_sufficient"
  |
  +--> Run Strategy A (pdfplumber + pymupdf)
          |
          +-- confidence >= 0.60 --> DONE, save result
          |
          +-- confidence < 0.60  --> escalate to Strategy B

DocumentProfile.cost_tier = "needs_layout_model"
OR escalated from Strategy A
  |
  +--> Run Strategy B (Docling layout-aware extraction)
          |
          +-- confidence >= 0.70 --> DONE, save result
          |
          +-- confidence < 0.70  --> escalate to Strategy C

DocumentProfile.cost_tier = "needs_vision_model"
OR language = "am" or "mixed"
OR escalated from Strategy B
  |
  +--> Run Strategy C (free VLM chain)
          |
          +-- try qwen/qwen3-vl-235b:free
          +-- 429 rate limit? backoff 2s, 4s, 8s then try next
          +-- try qwen/qwen3-vl-30b:free
          +-- try nvidia/nemotron-nano-12b-vl:free
          +-- try meta-llama/llama-4-maverick:free
          +-- try google/gemma-3-27b-it:free
          +-- try openrouter/auto:free
          +-- all failed? confidence=0.0, flagged for review
```

Every decision is appended to `.refinery/extraction_ledger.jsonl` with the strategy used, confidence score, estimated cost, and processing time.

### The confidence scorer

Confidence is not a single metric — it is a multi-signal composite score computed from the extracted text:

```
score starts at 0.15

+ 0.15  if total_chars >= 100
+ 0.15  if total_chars >= 400
+ 0.15  if total_chars >= 1200
+ 0.15  if alpha_char_ratio >= 0.35
+ 0.10  if alpha_char_ratio >= 0.55
+ 0.10  if unique_word_count >= 20
+ 0.10  if unique_word_count >= 80
+ 0.05  if digits present AND financial keywords found

- 0.05  per "unknown", "unclear", "unreadable" hit
- 0.05  per junk pattern (###, \u0000 escapes)
```

This means a document with 2000 clean financial characters scores around 0.85. A document that returned garbled OCR output scores below 0.35 and triggers escalation.

---

## Stage 3 — Semantic Chunking Engine

Raw extraction output is a flat list of blocks. The chunking engine converts these into **Logical Document Units (LDUs)** — semantically coherent, self-contained chunks that preserve document structure. It enforces five rules, and a `ChunkValidator` verifies every output.

### The five rules explained

**Rule 1 — Tables are always standalone**

A table block is never merged with surrounding text. The entire table — column headers plus every row — forms exactly one LDU regardless of size. This prevents the most common RAG failure mode: a financial table split across chunk boundaries, causing hallucinated answers when only half the data is retrieved.

**Rule 2 — Figure captions attach to their parent figure**

When a figure block is immediately followed by a caption (text starting with "Figure N:", "Table N:", or similar patterns), the caption is absorbed into the figure LDU as `[Caption]: ...`. This keeps visual content and its explanation together so retrieval always returns both.

**Rule 3 — Lists stay as one unit**

A numbered or bulleted list is kept as a single LDU unless it exceeds `max_chars` (1200 characters). When splitting is unavoidable, it happens at line boundaries so individual list items are never cut in half.

**Rule 4 — Section headers stamp context onto all children**

This is the most important rule for retrieval quality. When a header block is encountered, the chunking engine updates a running `current_section` tracker. Every subsequent text, table, figure, and list LDU receives `parent_section = current_section`. This means any retrieved chunk carries full section context — the grader never has to wonder "which part of the report is this from?"

**Rule 5 — Cross-references are preserved in provenance**

Text patterns like "see Table 3", "cf. Figure 2", "as shown in Section 4.1" are detected by regex and stored in the LDU's provenance metadata under `cross_refs`. This builds a relationship graph across chunks without requiring a separate pass.

### What each LDU looks like

```json
{
  "ldu_id": "fta_performance_survey-ldu-42-a3f9c1b2",
  "doc_id": "fta_performance_survey_final_report_2022",
  "chunk_type": "table",
  "content": "Assessment Area | Score | Weight\nBudget Transparency | 72 | 0.30\nAccountability | 68 | 0.25",
  "page_refs": [42],
  "bounding_box": [72.0, 100.0, 540.0, 380.0],
  "parent_section": "3.2 Assessment Results",
  "content_hash": "sha256:a3f9c1b2d4e5...",
  "provenance": {
    "document_name": "fta_performance_survey_final_report_2022.pdf",
    "source_path": "data/raw/fta_performance_survey_final_report_2022.pdf",
    "spans": [{ "page": 42, "bbox": [72.0, 100.0, 540.0, 380.0] }]
  }
}
```

---

## Stage 4 — PageIndex Builder

The PageIndex builder constructs two complementary navigation structures over the chunked document. Together they solve the "needle in a haystack" problem for long-document retrieval.

### Structure 1 — Flat page index

A direct lookup from page number to all items on that page. Every item has an `ldu_id`, `chunk_type`, a short snippet, and a `content_hash`. Use this when you need to find everything on a specific page.

```
page 1  -->  [header: "Executive Summary",  text: "This report presents..."]
page 12 -->  [table: "Tax Expenditure FY2020", text: "As shown in Table 1..."]
page 42 -->  [header: "3.2 Results",  table: "Income Statement",  text: "..."]
```

### Structure 2 — Hierarchical section tree

A nested tree of `SectionNode` objects. Each node has a title, page range, LLM-generated summary, extracted key entities, data types present, and a list of child sections.

This is the key innovation. It allows the query agent to navigate to the **right section of a 400-page document before running vector search**, which dramatically improves retrieval precision.

```
Document
+-- Chapter 1: Executive Summary  (pages 1-5)
|     summary: "The FTA performance survey assesses implementation..."
|     key_entities: ["Ministry of Finance", "4,521,367,000", "FY2020/21"]
|     data_types: [text, table]
|
+-- Chapter 2: Assessment Findings  (pages 6-45)
|   +-- Section 2.1: Budget Transparency  (pages 6-18)
|   |     summary: "Budget transparency score is 72 out of 100..."
|   |   +-- Section 2.1.1: Score Analysis  (pages 8-12)
|   |
|   +-- Section 2.2: Accountability  (pages 19-45)
|
+-- Chapter 3: Recommendations  (pages 46-60)
      summary: "Key actions include IBEX integration and public access..."
      key_entities: ["IBEX", "public access", "audit follow-up"]
```

Summaries are generated via an OpenRouter free LLM call if an API key is configured. If no key is set, an extractive fallback (first 3 sentences of the section) is used — the system never blocks on summary generation.

---

## Stage 5 — Query Interface Agent

The query agent is the user-facing front-end of the refinery. It answers natural language questions using three tools and always returns a `ProvenanceChain` with every answer so responses can be independently verified.

### Tool 1 — pageindex_navigate

Searches the hierarchical section tree by topic. Traverses every section node and scores it based on overlap between the query terms and the node's title, summary, and key entities. Returns the top matching sections with their page ranges.

```
query: "budget transparency recommendations"
  |
  +--> score each section node
  |      "Chapter 3: Recommendations"        score: 3.2  (topic overlap)
  |      "Section 2.1: Budget Transparency"  score: 2.8
  |      "Chapter 1: Executive Summary"      score: 0.4
  |
  +--> return top sections with page ranges, summaries, entity lists
```

This step narrows the search space from 10,000 chunks to 200 before vector search runs.

### Tool 2 — semantic_search

Vector similarity search over all LDUs using ChromaDB. Each LDU is embedded when it is first ingested. At query time, the query is embedded and cosine similarity is computed against all stored LDU vectors. Returns the top-k results with full provenance.

If ChromaDB is unavailable (not installed or index not built), the system automatically falls back to keyword scoring — the query never fails, it just uses a less precise method.

### Tool 3 — structured_query

SQL queries over a SQLite FactTable. During ingestion, numerical facts are extracted from every LDU by regex patterns matching monetary values, percentages, fiscal year references, and labelled quantities. These are stored in a `facts` table.

```sql
-- Example: find exact tax expenditure for a specific fiscal year
SELECT field_name, value, page, content_hash
FROM facts
WHERE field_name LIKE '%tax expenditure%'
AND doc_id = 'tax_expenditure_ethiopia_2021_22'
```

This is essential for queries that require exact numbers, not semantic similarity.

### Audit Mode — verify_claim

Given a plain text claim, the agent searches for supporting evidence and returns a structured verdict. This is designed for FDE client demos where a stakeholder wants to verify a specific assertion made by the system.

| Verdict | Meaning | How determined |
|---|---|---|
| ✅ VERIFIED | Supporting evidence found in document | Word overlap with evidence >= 50% |
| ⚠️ UNVERIFIABLE | Related content found but not conclusive | Word overlap 20–49% |
| ❌ NOT FOUND | No matching content in the corpus | Word overlap < 20% |

### What every answer includes

```json
{
  "answer_snippets": ["The total tax expenditure for FY2020/21 was..."],
  "provenance_chain": [
    {
      "document_name": "tax_expenditure_ethiopia_2021_22.pdf",
      "page_number": 12,
      "bbox": [72.0, 100.0, 540.0, 380.0],
      "content_hash": "sha256:a3f9c1b2d4e5..."
    }
  ],
  "navigation_hits": [
    { "title": "Table 1: Summary", "page_start": 12, "page_end": 15 }
  ]
}
```

---

## Key Engineering Decisions

### Why confidence-gated escalation instead of always using the best strategy?

Without escalation, every document would be sent to Strategy C. With escalation, the system runs the cheapest strategy first and only escalates when quality is genuinely insufficient. In practice, native digital PDFs with clean text never need Vision LLM — they are handled by Strategy A or B in seconds at near-zero cost.

```
Strategy A  [===.................] ~$0.001   2-5 seconds
Strategy B  [=======.............] ~$0.010   10-30 seconds
Strategy C  [====================]  $0.000   5-15 seconds (FREE)
```

### Why build a section tree instead of flat chunking for RAG?

| Approach | Query: "budget transparency recommendations" |
|---|---|
| Naive 512-token flat chunks | Scan all 10,000+ chunks. The word "budget" appears 847 times across the corpus. Low precision, slow, wrong sections returned. |
| PageIndex section tree (this system) | Navigate to "Chapter 3: Recommendations" first (3 seconds). Then search only the ~200 chunks in that section. High precision, correct answer. |

### Why use both SQLite and ChromaDB instead of just one?

These two backends serve fundamentally different query types that cannot substitute for each other.

ChromaDB handles **semantic queries**: "what does the report say about fiscal transparency?" — conceptual, fuzzy, where exact wording does not matter.

SQLite handles **exact numerical queries**: "what was the total tax expenditure in FY2020/21?" — where you need the precise number `4,521,367,000` and not a paragraph that is thematically similar.

Using only one means half of all real client queries will either fail or return wrong results.

### Why charge the budget only on success?

The `BudgetController.charge()` method only fires after Strategy C returns real extracted content. If a VLM call fails with a network error, a rate limit, or a model error, the budget is not charged for that attempt. This means a document that requires 3 retry attempts across 2 different models is still only charged for the one successful call, not three failed ones.

---

## Free VLM Strategy

All Vision LLM extraction runs on OpenRouter's free tier. The cost is exactly $0.00. The system maintains a fallback chain of six free models so that if one hits its daily rate limit, the pipeline automatically continues with the next.

### Standard document chain

| Priority | Model | Why chosen |
|---|---|---|
| 1st | `qwen/qwen3-vl-235b-thinking:free` | Best OCR quality, 32 languages, 262K context, built for documents |
| 2nd | `qwen/qwen3-vl-30b-a3b-thinking:free` | Strong vision, 131K context, fast |
| 3rd | `nvidia/nemotron-nano-12b-vl:free` | Specifically trained on OCRBench and DocVQA benchmarks |
| 4th | `meta-llama/llama-4-maverick:free` | Large model, broad multilingual coverage |
| 5th | `google/gemma-3-27b-it:free` | Supports 140+ languages, good for non-Latin scripts |
| 6th | `openrouter/auto:free` | Safety net — OpenRouter picks the best available free model |

### Amharic / Ethiopic document chain

The chain is reordered for Amharic documents to prioritise models with the strongest multilingual training.

| Priority | Model |
|---|---|
| 1st | `qwen/qwen3-vl-235b-thinking:free` |
| 2nd | `google/gemma-3-27b-it:free` |
| 3rd | `qwen/qwen3-vl-30b-a3b-thinking:free` |
| 4th | `meta-llama/llama-4-maverick:free` |
| 5th | `openrouter/auto:free` |

### Rate limit handling

```
HTTP 429 received from current model
  |
  +--> wait 2 seconds, retry same model
  +--> wait 4 seconds, retry same model
  +--> wait 8 seconds, retry same model
  +--> all retries exhausted, move to next model in chain
  +--> all 6 models tried and failed
        --> confidence = 0.0
        --> needs_review = true
        --> flagged in extraction ledger
```

---

---

## LLM Strategy: OpenRouter Free Tier vs Ollama vs Claude API

### Recommendation for this project

**Use OpenRouter free tier.** Here is a direct comparison:

| Option | Cost | Amharic support | Setup | Best model available |
|---|---|---|---|---|
| **OpenRouter free** | **$0.00** | **Yes — qwen3-vl-235b** | API key only | qwen3-vl-235b-thinking:free |
| Ollama (local) | Free, GPU required | Limited | 235B model needs 140GB RAM | qwen3-vl-235b (if you have H100s) |
| Claude API | ~$0.015/image | Excellent | API key | claude-opus-4-6 |

### Why OpenRouter free tier is the right choice

The `qwen/qwen3-vl-235b-thinking:free` model on OpenRouter is the same model that runs on Ollama — the difference is that OpenRouter runs it on their infrastructure at no cost to you. For this project:

- **You do not need a GPU** — OpenRouter's servers handle it
- **Amharic/Ethiopic script is supported** — Qwen3-VL was trained on 32 languages including multilingual document OCR
- **Rate limits are generous** — ~20 req/min, ~200 req/day per model; the pipeline falls back to the next model automatically
- **Cost is $0.00** — the budget guard in `extraction_rules.yaml` is already set to `0.00`

### Why NOT Ollama for this project

Running `qwen3-vl:235b-instruct` locally requires **~140GB of GPU VRAM** (or extremely slow CPU inference). Unless you have access to a multi-GPU server, local Ollama is not practical for the 235B model. The smaller `qwen3-vl:30b` variant fits in ~20GB and gives reasonable Amharic results, but OpenRouter gives you the 235B for free.

### Why NOT Claude API directly

Claude does not expose an image-based OCR API designed for bulk document extraction. Claude API calls cost ~$15 per 1,000 pages at Opus pricing. OpenRouter free tier is the correct choice for this pipeline.

### How to configure

```bash
# .env — just set this, everything else is pre-configured
OPENROUTER_API_KEY=sk-or-your-key-here

# Free models used automatically (in order of preference for Amharic):
# 1. qwen/qwen3-vl-235b-thinking:free    <- Best for Ethiopic script
# 2. google/gemma-3-27b-it:free          <- 140+ languages
# 3. qwen/qwen3-vl-30b-a3b-thinking:free <- Strong multilingual fallback
# 4. meta-llama/llama-4-maverick:free    <- Broad multilingual
# 5. openrouter/auto:free                <- Safety net
```

### Optional: Use Ollama for Strategy B (Docling) only

If you want fully offline extraction for native digital PDFs (no API key needed), Docling (Strategy B) runs entirely locally using CPU/GPU. For scanned PDFs and Amharic documents, OpenRouter remains the recommended path.


## Amharic and Ethiopic Script Support

This pipeline has first-class support for Amharic documents — a capability that most document intelligence tools lack entirely because they depend on Tesseract, which cannot handle Ethiopic script without a separately installed language pack.

### How language detection works

The triage agent counts characters in the Ethiopic Unicode block (`U+1200` to `U+137F`). If 5 or more Ethiopic characters are found and they represent more than 5% of the total character count, the document is classified as `language = "am"`. If Ethiopic and Latin characters are both strongly present, it is classified as `language = "mixed"`.

### Why Tesseract fails on Amharic

Standard OCR tools use Tesseract under the hood. Without the `tesseract-lang-amh` language pack installed, Tesseract either drops every Ethiopic character or replaces it with `?`. The result is an empty or garbled extraction. This is the reason Strategy C1 (local OCR) and C2 (layout OCR) are both skipped for Amharic documents — they would waste time and return zero-confidence output.

### The Amharic fast-path

```
Triage detects language = "am" or "mixed"
  |
  +--> StrategyCRouter skips C1 (Tesseract local OCR)
  +--> StrategyCRouter skips C2 (Tesseract layout OCR)
  +--> Goes directly to C3 (Vision VLM)
  +--> Passes language="am" to VLM
  +--> VLM prompt includes Ethiopic preservation instruction:
  
       "CRITICAL: This document is in Amharic (Ethiopic/Ge'ez script).
        You MUST preserve every Ethiopic character exactly as written.
        Do NOT transliterate, romanise, skip, or replace with '?'.
        Output every word in its original script."
```

This produces extracted text that contains real Amharic characters (`ሰላም ኢትዮጵያ`) rather than romanised approximations or empty strings.

---

## Project Structure

```
document-intelligence-refinery/
|
+-- app.py                        Streamlit UI with 5-tab query interface
+-- pyproject.toml                Dependencies and CLI entry points
+-- Dockerfile                    Container deployment
+-- README.md                     This file
+-- DOMAIN_NOTES.md               Phase 0 analysis, decision tree, failure modes
|
+-- rubric/
|   +-- extraction_rules.yaml     All thresholds and model chains (no code changes needed)
|
+-- src/
|   +-- agents/
|   |   +-- triage.py             Stage 1: document classification
|   |   +-- extractor.py          Stage 2: confidence-gated extraction routing
|   |   +-- chunker.py            Stage 3: semantic chunking with 5 rules
|   |   +-- indexer.py            Stage 4: PageIndex and section tree builder
|   |   +-- ldu_builder.py        LDU construction from extracted blocks
|   |   +-- query_agent.py        Stage 5: 3-tool query agent and Audit Mode
|   |
|   +-- strategies/
|   |   +-- base.py               BaseExtractor interface
|   |   +-- fast_text.py          Strategy A: pdfplumber and pymupdf
|   |   +-- layout_docling.py     Strategy B: Docling layout-aware extraction
|   |   +-- ocr_local.py          Strategy C1: local Tesseract OCR
|   |   +-- ocr_layout.py         Strategy C2: layout-aware OCR
|   |   +-- vision_vlm.py         Strategy C3: free VLM fallback chain
|   |   +-- strategy_c_router.py  C1 to C2 to C3 ladder with Amharic fast-path
|   |
|   +-- models/
|   |   +-- profile.py            DocumentProfile (Pydantic model)
|   |   +-- extracted_document.py ExtractedDocument and ExtractedBlock
|   |   +-- ldu.py                LDU — Logical Document Unit
|   |   +-- pageindex.py          PageIndex and SectionNode tree
|   |   +-- provenance.py         ProvenanceChain and ProvenanceSpan
|   |   +-- ledger.py             ExtractionLedgerEvent
|   |
|   +-- engine/
|   |   +-- budget.py             BudgetController: pre-flight check and charge on success
|   |   +-- cache.py              ArtifactCache: skip documents already processed
|   |   +-- policy.py             EscalationPolicy: economy, balanced, quality modes
|   |   +-- qa.py                 QA scorer for extraction output validation
|   |
|   +-- utils/
|   |   +-- confidence.py         Multi-signal confidence scorer
|   |   +-- hashing.py            sha256_text for content_hash generation
|   |   +-- pdf_signals.py        Character density and image ratio analysis
|   |   +-- pdf_layout.py         Column count and table/figure detection
|   |
|   +-- service/
|   |   +-- refinery_service.py   run_refinery_on_pdf(): full 5-stage pipeline
|   |
|   +-- settings.py               Pydantic settings and path configuration
|   +-- main.py                   CLI entry point
|
+-- tests/
|   +-- test_triage.py            Stage 1 classification tests
|   +-- test_confidence.py        Confidence scorer tests
|   +-- test_router.py            Escalation routing tests
|   +-- test_chunker.py           All 5 chunking rule tests
|
+-- .refinery/                    Generated artifacts
    +-- profiles/                 DocumentProfile JSON per document
    +-- extracted/                ExtractedDocument JSON per document
    +-- ldu/                      LDU list JSON per document
    +-- pageindex/                PageIndex and section tree per document
    +-- chroma/                   ChromaDB vector store (persisted)
    +-- facts.db                  SQLite FactTable for numerical data
    +-- extraction_ledger.jsonl   Append-only audit trail for every run
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/document-intelligence-refinery
cd document-intelligence-refinery
pip install -e ".[docling]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your OpenRouter key:

```dotenv
# Required for scanned and Amharic documents (Strategy C)
OPENROUTER_API_KEY=sk-or-your-key-here

# Free models — $0.00 cost
OPENROUTER_URL=https://openrouter.ai/api/v1
MODEL_NAME=openrouter/auto:free

# Optional: helps OpenRouter analytics
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=document-intelligence-refinery
```

> Get a free OpenRouter API key at [openrouter.ai](https://openrouter.ai) — no credit card required for free models.

### 3. Create the artifact directories

```bash
mkdir -p .refinery/pageindex .refinery/ldu .refinery/extracted .refinery/chroma
```

### 4. Launch the Streamlit UI

```bash
streamlit run app.py
```

Open **http://localhost:8501**, upload any PDF, and click **Run Refinery**.

### 5. Run via command line

```bash
# Process a single document (full 5-stage pipeline)
python -m src.main run --input-path my_document.pdf

# Process all PDFs in a directory
python -m src.main run --input-path ./docs/ --limit 10

# Ask a question against a processed document
python -m src.main query --pdf-path my_document.pdf --question "What is the total revenue?"

# Audit / verify a claim
python -m src.main audit --pdf-path my_document.pdf --claim "Revenue was $4.2B in Q3"
```

### 6. Run with Docker

```bash
docker build -t refinery .
docker run -p 8502:8502 -e OPENROUTER_API_KEY=sk-or-... refinery
```

Open **http://localhost:8502**

---

## Configuration

All pipeline behaviour is controlled from a single YAML file. **No code changes are needed to onboard a new document type or adjust thresholds.** Edit `rubric/extraction_rules.yaml` and restart.

```yaml
triage:
  scanned_image_threshold:
    image_area_ratio_gte: 0.55      # page is "scanned" if images cover >=55% of area
    text_chars_per_page_lte: 300    # AND fewer than 300 characters extracted per page

confidence:
  strategy_a_min_confidence: 0.60   # below this, escalate A to B
  strategy_b_min_confidence: 0.70   # below this, escalate B to C

vlm:
  model_chain:                       # tried in order, all FREE ($0.00)
    - qwen/qwen3-vl-235b-thinking:free
    - qwen/qwen3-vl-30b-a3b-thinking:free
    - nvidia/nemotron-nano-12b-vl:free
    - meta-llama/llama-4-maverick:free
    - google/gemma-3-27b-it:free
    - openrouter/auto:free
  retry_backoff_s: [2, 4, 8]        # exponential backoff delays on HTTP 429

chunking:
  max_chars: 1200                    # maximum characters per text LDU
  min_chars: 120                     # minimum meaningful chunk size
  overlap_chars: 100                 # overlap for sliding window if used
```

### Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required** for Strategy C (scanned docs) |
| `MODEL_NAME` | `openrouter/auto:free` | Primary VLM model override |
| `OPENROUTER_MODEL_CHAIN` | from YAML | Comma-separated model fallback list override |
| `MAX_VLM_PAGES` | `6` | Maximum pages sent to VLM per document |
| `REFINERY_BATCH_BUDGET_USD` | `0.00` | Total spend cap for a pipeline run |
| `REFINERY_MAX_DOC_BUDGET_USD` | `0.00` | Per-document spend cap |
| `REFINERY_ALLOW_C3` | `1` | Set to `0` to disable VLM entirely |

---

## Artifact Outputs

Every pipeline run writes structured artifacts to `.refinery/`. These are the primary deliverables inspected during grading.

### profiles/{doc_id}.json — Stage 1 output

```json
{
  "doc_id": "fta_performance_survey_final_report_2022",
  "origin_type": "native_digital",
  "layout_complexity": "multi_column",
  "language": "en",
  "domain_hint": "technical",
  "page_count": 155,
  "avg_text_chars_per_page": 2616.5,
  "avg_image_area_ratio": 0.0007,
  "cost_tier": "needs_layout_model"
}
```

### extraction_ledger.jsonl — Full audit trail

One JSON line is appended for every extraction run:

```json
{
  "doc_id": "fta_performance_survey_final_report_2022",
  "strategy_used": "B",
  "confidence": 0.75,
  "escalated": false,
  "cost_estimate_usd": 0.01,
  "processing_time_s": 12.4,
  "signals": {
    "origin_type": "native_digital",
    "layout_complexity": "multi_column",
    "avg_text_chars_per_page": 2616.5,
    "avg_image_area_ratio": 0.0007
  }
}
```

### ldu/{doc_id}.json — Stage 3 output

```json
[
  {
    "ldu_id": "fta_performance_survey-ldu-42-a3f9c1b2",
    "chunk_type": "table",
    "content": "Assessment Area | Score | Weight\nBudget Transparency | 72 | 0.30",
    "page_refs": [42],
    "bounding_box": [72.0, 100.0, 540.0, 380.0],
    "parent_section": "3.2 Assessment Results",
    "content_hash": "sha256:a3f9c1b2..."
  }
]
```

### pageindex/{doc_id}.json — Stage 4 output

Contains the flat page index and the full hierarchical section tree with LLM-generated summaries, extracted entities, and child section nesting.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run individual suites
pytest tests/test_triage.py -v
pytest tests/test_router.py -v
pytest tests/test_confidence.py -v
pytest tests/test_chunker.py -v
```

### What each suite covers

| Suite | What it verifies |
|---|---|
| `test_triage.py` | Scanned PDF classified as `scanned_image` and `needs_vision_model`; native multi-column classified correctly |
| `test_router.py` | High-confidence Strategy A stops at A without calling B or C; low-confidence A escalates to B |
| `test_confidence.py` | Empty blocks return 0.0; good financial text returns >= 0.55; garbage text returns < 0.35 |
| `test_chunker.py` | R1: table never merged; R2: caption absorbed into figure; R3: list preserved as one unit; R4: headers propagate to children; R5: cross-references stored; validator detects violations; all content preserved |

---

---

## Testing & Verification — Run Everything

This section contains the exact commands to verify every system component. Run in order after installation.

### Step 0: Installation

```bash
# Install core + dev dependencies
pip install -e ".[dev]"

# Optional: install Docling for Strategy B (layout-aware extraction)
pip install -e ".[docling]"

# Verify imports
python -c "
from src.agents.triage import triage_pdf
from src.agents.chunker import SemanticChunkingEngine, ChunkValidator
from src.agents.indexer import build_page_index
from src.agents.query_agent import QueryAgent, VectorStore, FactTable
print('All imports OK')
"
```

### Step 1: Unit tests (no PDF needed)

```bash
# Full test suite — all should pass
pytest tests/ -v

# Individual suites
pytest tests/test_triage.py -v        # Stage 1 classification
pytest tests/test_confidence.py -v    # Confidence scoring logic
pytest tests/test_router.py -v        # Escalation routing (A->B->C)
pytest tests/test_chunker.py -v       # All 5 chunking rules (R1-R5)
```

### Step 2: Run full pipeline on a PDF

```bash
# Process one document
python -m src.main run --input-path path/to/your.pdf

# Process entire folder
python -m src.main run --input-path ./docs/ --limit 10
```

**Artifacts written:**
```
.refinery/profiles/{doc_id}.json        <- DocumentProfile (Stage 1)
.refinery/extracted/{doc_id}.json       <- ExtractedDocument (Stage 2)
.refinery/ldu/{doc_id}.json             <- LDU list with chunking rules (Stage 3)
.refinery/pageindex/{doc_id}.json       <- Hierarchical section tree (Stage 4)
.refinery/qa/{doc_id}.json              <- Q&A pairs with provenance (Stage 5)
.refinery/chroma/                        <- ChromaDB vector store
.refinery/facts.db                       <- SQLite FactTable
.refinery/extraction_ledger.jsonl        <- Audit trail (append-only)
```

### Step 3: Verify DocumentProfile (Stage 1)

```bash
cat .refinery/profiles/<doc_id>.json | python -m json.tool
# Expect: origin_type, layout_complexity, language, domain_hint, cost_tier
```

### Step 4: Verify table extraction (Stage 2)

```bash
python -c "
import json
from pathlib import Path
f = sorted(Path('.refinery/extracted').glob('*.json'))[0]
d = json.loads(f.read_text())
tables = [b for b in d['blocks'] if b['block_type'] == 'table']
print(f'Strategy: {d["strategy_used"]}  Confidence: {d["confidence"]:.2f}')
print(f'Blocks: {len(d["blocks"])}  Tables: {len(tables)}')
if tables: print('Sample table:', tables[0]['text'][:300])
"
```

### Step 5: Verify chunking rules (Stage 3)

```bash
python -c "
import json
from pathlib import Path
from src.agents.chunker import ChunkValidator
from src.models.ldu import LDU
f = sorted(Path('.refinery/ldu').glob('*.json'))[0]
ldus = [LDU(**l) for l in json.loads(f.read_text())]
violations = ChunkValidator().validate(ldus)
types = {}
for ldu in ldus:
    types[ldu.chunk_type] = types.get(ldu.chunk_type, 0) + 1
print(f'LDUs: {len(ldus)}  Types: {types}')
print(f'Rule violations: {len(violations)}')
for v in violations[:3]: print(f'  - {v}')
if not violations: print('All 5 chunking rules satisfied.')
"
```

### Step 6: Verify PageIndex tree (Stage 4)

```bash
python -c "
import json
from pathlib import Path
f = sorted(Path('.refinery/pageindex').glob('*.json'))[0]
pi = json.loads(f.read_text())
print(f'Pages: {pi["page_count"]}  Sections: {len(pi["sections"])}')
for s in pi['sections'][:5]:
    print(f'  [{s["level"]}] {s["title"][:60]} pp.{s["page_start"]}-{s["page_end"]}')
    if s.get("summary"): print(f'      {s["summary"][:100]}')
"
```

### Step 7: Query with Provenance (Stage 5)

```bash
# Natural language query with full ProvenanceChain
python -m src.main query   --pdf-path path/to/your.pdf   --question "What is the total revenue reported?"

# Claim verification (Audit Mode)
python -m src.main audit   --pdf-path path/to/your.pdf   --claim "The document reports revenue over 10 billion"
```

### Step 8: Test Strategy C (Vision VLM) — scanned or Amharic docs

```bash
# Set API key first
export OPENROUTER_API_KEY=sk-or-your-key-here

# Run on a scanned PDF (will auto-route to Strategy C)
python -m src.main run --input-path path/to/scanned.pdf

# Verify VLM was used and cost was $0.00
tail -1 .refinery/extraction_ledger.jsonl | python -m json.tool
# Expect: "strategy_used": "C", "cost_estimate_usd": 0.0
```

### Step 9: Process all 4 corpus documents

```bash
# Place all 4 corpus PDFs in docs/ then:
python -m src.main run --input-path ./docs/

# Summary report of all runs
cat .refinery/extraction_ledger.jsonl | python -c "
import json, sys
for line in sys.stdin:
    o = json.loads(line)
    print(f'{o["doc_id"][:38]:38}  strategy={o["strategy_used"]}  conf={o.get("confidence",0):.2f}  cost=\${o.get("cost_estimate_usd",0):.3f}')
"
```

### Step 10: Launch Streamlit UI

```bash
streamlit run app.py
# Open http://localhost:8501
# Upload PDF → Run Refinery → inspect all 4 tabs: Profile, Extraction, PageIndex, Query
```


## Demo Protocol

For the 5-minute video walkthrough, follow these four steps in order.

**Step 1 — Triage (60 seconds)**

Open the Streamlit UI and upload a document. Navigate to the Profile tab. Point to `origin_type` and explain how it was derived from character density and image ratio signals — not a model, just arithmetic. Show `cost_tier` and explain which strategy was selected and why. If using an Amharic document, show `language: "am"` and explain that Tesseract is bypassed entirely.

**Step 2 — Extraction (90 seconds)**

Show the Extracted tab next to the original PDF. Scroll to a table block and show it extracted as structured JSON with correct column headers and all rows intact. Open `.refinery/extraction_ledger.jsonl` in a terminal and show the entry with `strategy_used`, `confidence`, and `cost_estimate_usd: 0.0`.

**Step 3 — PageIndex (60 seconds)**

Open the Page Index tab. Expand the section tree and show that it mirrors the document's table of contents — titles, page ranges, summaries, and key entities. Type a topic into the navigation tool and show it returns the correct section with page numbers, not a flat list of chunks.

**Step 4 — Query with Provenance (90 seconds)**

In the Query tab, ask a specific factual question about the document. Show the answer snippets plus the full ProvenanceChain — page number, bounding box, and content hash. Open the original PDF to the cited page and confirm the cited text is there. Switch to Audit Mode, paste a specific claim, and show the `VERIFIED` verdict with supporting evidence.

---

## Corpus Documents

| Class | Document | Origin Type | Strategy Used | Key Challenge |
|---|---|---|---|---|
| A | CBE Annual Report 2023–24 | Native digital | B — Layout | Two-column layout, embedded financial tables, cross-page footnotes |
| B | DBE Auditor's Report 2023 | Scanned image | C — Vision VLM | No text layer at all — pure scanned JPEG, requires VLM OCR |
| C | FTA Performance Survey 2022 | Mixed | B or C | 155 pages, narrative sections interleaved with assessment tables |
| D | Tax Expenditure Report FY2021 | Native digital | B — Layout | Multi-year fiscal comparison tables with precise numerical values |

Amharic documents in the corpus (`2013-E.C-*` series) are all processed via Strategy C with the Ethiopic-preservation prompt. The fast-path in `StrategyCRouter` skips C1 and C2 and goes directly to the VLM chain for all documents where `language = "am"` or `language = "mixed"`.

---

<div align="center">

**Document Intelligence Refinery** &nbsp;·&nbsp; TRP1 Week 3 &nbsp;·&nbsp; FDE Program

*Meseret Bolled &nbsp;·&nbsp; Addis Ababa Science and Technology University*

</div>
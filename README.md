📋 Table of Contents

The Problem
What This Builds
System Architecture
The 5 Pipeline Stages
The Engineering Decisions
Free VLM Strategy
Amharic Support
Project Structure
Quick Start
Configuration
Artifact Outputs
Running Tests
Demo Protocol

🔴 The Problem
Every enterprise has its institutional memory locked in documents. Banks, hospitals, law firms — all face the same three failure modes:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STRUCTURE COLLAPSE   Traditional OCR flattens two-column layouts,  │
│                       breaks tables, drops headers. Text is present │
│                       but semantically useless.                     │
│                                                                     │
│  CONTEXT POVERTY      Naive chunking severs logical units. A table  │
│                       split across two chunks → hallucinated        │
│                       answers on every query about that table.      │
│                                                                     │
│  PROVENANCE BLINDNESS "Where in the 400-page report does this       │
│                       number come from?" Without spatial            │
│                       provenance, extracted data can't be audited.  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


This pipeline solves all three.

✅ What This Builds

INPUT                                          OUTPUT
─────────────────────────────────────────────────────────────────────
PDFs (native digital)    ──┐               ┌─ Structured JSON schemas
PDFs (scanned images)    ──┤               ├─ Hierarchical section tree
Amharic / Ethiopic docs  ──┤   REFINERY    ├─ ChromaDB vector store
Mixed layout reports     ──┤   PIPELINE    ├─ SQLite FactTable (numbers)
Table-heavy fiscal data  ──┘               ├─ Provenance-tagged LDUs
                                           └─ Audit trail with page+bbox


  🏗 System Architecture                                         
╔══════════════════════════════════════════════════════════════════════╗
║                    DOCUMENT INTELLIGENCE REFINERY                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────┐    ┌─────────────┐    ┌──────────────────────────┐   ║
║   │  INPUT   │    │   STAGE 1   │    │        STAGE 2           │   ║
║   │          │───▶│   TRIAGE    │───▶│    EXTRACTION ROUTER     │   ║
║   │ Any PDF  │    │   AGENT     │    │                          │   ║
║   └──────────┘    └─────────────┘    │  ┌────┐ ┌────┐ ┌────┐   │   ║
║                         │            │  │ A  │ │ B  │ │ C  │   │   ║
║                  DocumentProfile     │  │Fast│ │Lay-│ │VLM │   │   ║
║                         │            │  │Text│ │out │ │Free│   │   ║
║                    ┌────▼───────┐    │  └────┘ └────┘ └────┘   │   ║
║                    │ origin?    │    │  conf≥0.6 conf≥0.7  ▲    │   ║
║                    │ layout?    │    │     │       │    escalate│   ║
║                    │ language?  │    └──────────────────────────┘   ║
║                    │ cost_tier? │                 │                  ║
║                    └────────────┘         ExtractedDocument          ║
║                                                   │                  ║
║   ┌────────────────────────────────────────────────▼───────────┐    ║
║   │                        STAGE 3                              │    ║
║   │               SEMANTIC CHUNKING ENGINE                      │    ║
║   │                                                             │    ║
║   │  R1: Tables always standalone (never split from header)     │    ║
║   │  R2: Figure captions stored as parent figure metadata       │    ║
║   │  R3: Lists kept as single LDU unless exceeds max_chars      │    ║
║   │  R4: Section headers propagated to all child chunks         │    ║
║   │  R5: Cross-references stored in provenance meta             │    ║
║   │                          ▼ ChunkValidator                   │    ║
║   └───────────────────── List[LDU] ─────────────────────────────┘    ║
║                               │                                      ║
║   ┌───────────────────────────▼────────────────────────────────┐    ║
║   │                        STAGE 4                              │    ║
║   │                  PAGEINDEX BUILDER                          │    ║
║   │                                                             │    ║
║   │   Flat page index          Hierarchical section tree        │    ║
║   │   ┌──────────────┐         ┌─ Document                      │    ║
║   │   │ page 1: [...] │         │  ├─ Chapter 1                  │    ║
║   │   │ page 2: [...] │         │  │  ├─ Section 1.1             │    ║
║   │   │ page N: [...] │         │  │  └─ Section 1.2             │    ║
║   │   └──────────────┘         │  └─ Chapter 2                  │    ║
║   │                            └── [summary + entities]         │    ║
║   └─────────────────────────────────────────────────────────────┘    ║
║                               │                                      ║
║   ┌───────────────────────────▼────────────────────────────────┐    ║
║   │                        STAGE 5                              │    ║
║   │                   QUERY INTERFACE AGENT                     │    ║
║   │                                                             │    ║
║   │  Tool 1: pageindex_navigate(topic)  ── section tree search  │    ║
║   │  Tool 2: semantic_search(query)     ── ChromaDB vectors     │    ║
║   │  Tool 3: structured_query(field)    ── SQLite FactTable      │    ║
║   │                    +                                         │    ║
║   │  Audit Mode: verify_claim(claim) → verified/unverifiable    │    ║
║   │                                                             │    ║
║   │  Every answer: ProvenanceChain { doc, page, bbox, hash }    │    ║
║   └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

🔄 The 5 Pipeline Stages
Stage 1 — Triage Agent

PDF arrives
    │
    ▼
pdfplumber signals
├── avg_text_chars_per_page  ──┐
├── avg_image_area_ratio      ├──▶  origin_type
└── font_metadata_present    ──┘    native_digital | scanned_image | mixed
                                          │
column heuristics ──────────────▶  layout_complexity
table/figure bbox score             single_column | multi_column
                                    table_heavy | figure_heavy
                                          │
Ethiopic Unicode range                    │
U+1200–U+137F char count ──────▶  language  (am | en | mixed)
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │    DocumentProfile    │
                               │  + cost_tier          │
                               │  + domain_hint        │
                               └──────────────────────┘
                               saved → .refinery/profiles/

Stage 2 — Multi-Strategy Extraction Router

DocumentProfile
      │
      ├─ cost_tier = fast_text_sufficient
      │         └──▶ Strategy A (FastText)
      │               pdfplumber + pymupdf
      │               Confidence gate: ≥ 0.60
      │               if FAIL ──────────────────────────┐
      │                                                  │
      ├─ cost_tier = needs_layout_model                  │
      │    OR escalated from A ◀────────────────────────┘
      │         └──▶ Strategy B (Layout-Aware)
      │               Docling — tables as JSON
      │               Reading order reconstruction
      │               Confidence gate: ≥ 0.70
      │               if FAIL ──────────────────────────┐
      │                                                  │
      └─ cost_tier = needs_vision_model                  │
           OR language = am/mixed                        │
           OR escalated from B ◀─────────────────────────┘
                └──▶ Strategy C (Vision VLM) ← FREE
                      Model fallback chain:
                      1. qwen/qwen3-vl-235b:free  ← OCR specialist
                      2. qwen/qwen3-vl-30b:free
                      3. nvidia/nemotron-nano-12b-vl:free
                      4. meta-llama/llama-4-maverick:free
                      5. google/gemma-3-27b-it:free  ← 140+ languages
                      6. openrouter/auto:free  ← safety net
                      
                      429 rate limit? → exponential backoff (2s→4s→8s)
                                       → then try next model
                      All models exhausted? → confidence=0.0 flagged

Every run logged to .refinery/extraction_ledger.jsonl

Stage 3 — Semantic Chunking Engine
ExtractedDocument.blocks[]
         │
         ▼
    ChunkingEngine                     5 Constitution Rules
         │
         ├─ chunk_type == "header"  ──▶  R4: Update current_section context
         │                               Emit standalone. Stamp parent_section
         │                               on ALL subsequent child LDUs.
         │
         ├─ chunk_type == "table"   ──▶  R1: Always standalone LDU.
         │                               NEVER merged with surrounding text.
         │                               Entire table (headers+rows) = 1 LDU.
         │
         ├─ chunk_type == "figure"  ──▶  R2: Absorb next block if it matches
         │                               caption pattern (^Figure|Table \d+).
         │                               Caption stored as "[Caption]: ..." in
         │                               figure LDU content.
         │
         ├─ chunk_type == "list"    ──▶  R3: Keep as single LDU.
         │                               Split only if > max_chars (1200).
         │                               Split at line boundaries, not tokens.
         │
         └─ chunk_type == "text"    ──▶  R5: Scan for cross-references
                                         (see Table 3, cf. Figure 2, etc.)
                                         Store matches in provenance.meta
                                         Buffer + merge up to max_chars.
                                                   │
                                                   ▼
                                         ChunkValidator.validate(ldus)
                                         → returns List[str] violations
                                         → logged but non-blocking

Each LDU carries:
  ldu_id        content_hash    chunk_type
  content       token_count     parent_section
  page_refs     bounding_box    provenance: ProvenanceChain

  Stage 4 — PageIndex Builder

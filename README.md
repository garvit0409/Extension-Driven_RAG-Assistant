# Extension-Driven Hybrid RAG Assistant

An advanced, production-ready Retrieval-Augmented Generation (RAG) assistant built with **Streamlit**, **LangChain**, and **Groq (Qwen 3)**. This application features a deterministic intent-based dispatch router that dynamically splits system workflows between an unstructured text generation pipeline and a compiled Pandas execution matrix to flawlessly handle multi-format datasets.

---

## 🎯 Key Features & Bug Fixes

* **Dual-Representation Ingestion Engine:** Maps files (`.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`) into isolated layers. For structured datasets, it maintains a master schema blueprint document for rapid metadata lookups alongside tokenized textual chunks in **ChromaDB**.
* **Intent-Based Dispatch Router:** Uses real-time keyword intent analysis to split system workflows. Analytical queries are routed to an executable code generation matrix, while semantic questions utilize the vector retrieval stream.
* **Hierarchical Duplicate/Hallucination Avoidance:** Avoids common RAG flaws (like the "only 3 rows shown" limitation) by injecting full-data tabular text snapshots when structural entities are targeted.
* **Self-Correcting Compliance Loop:** Employs a strict QA verification engine that intercepts model output to run automated truth checks against data chunks, instantly dropping and logging unverified text loops.
* **Relational Error Tracking:** Features a built-in local SQLite pipeline to log hallucination events or user-flagged errors with contextual metadata snapshotting for admin evaluation.

---

## 🏗️ Architecture Overview

The system architecture cleanly decouples unstructured document processing from tabular mathematical execution:

```text
                      +-----------------------------+
                      |   User Input (Streamlit)    |
                      +--------------+--------------+
                                     |
                        [Enrich Query with History]
                                     |
                                     v
                      +--------------+--------------+
                      |   Similarity Search (k=8)   |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      | Deterministic Router Switch |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
    [Analytical Query & Schema Match]             [Unstructured Text Match]
              |                                             |
              v                                             v
+-------------+-------------+                 +-------------+-------------+
|  Engine A: Pandas Matrix  |                 |  Engine B: Document RAG   |
|  - Generate Pandas Expr   |                 |  - Merge Extra Context    |
|  - Load CSV/Excel File    |                 |  - System Prompt Stricture|
|  - Eval & Humanize Code   |                 |  - Stream LLM Response    |
+-------------+-------------+                 +-------------+-------------+
              |                                             |
              +----------------------+----------------------+
                                     |
                                     v
                      +--------------+--------------+
                      | Self-Correcting Guardrail   |
                      |   (QA Evaluation Loop)      |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
                 [PASSED]                        [FAILED]
                     |                               |
                     v                               v
         Display Response to User       Log to SQLite (app_logs.db)
                                        Show Friendly Catch Warning

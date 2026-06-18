Extension-Driven Hybrid RAG Assistant
An advanced, production-ready Retrieval-Augmented Generation (RAG) assistant built with Streamlit, LangChain, and Groq (Qwen 3). This application features a deterministic intent-based dispatch router that dynamically splits system workflows between an unstructured text generation pipeline and a compiled Pandas execution matrix to flawlessly handle multi-format datasets.

🎯 Key Features & Bug Fixes
Dual-Representation Ingestion Engine: Maps files (.pdf, .docx, .txt, .csv, .xlsx) into isolated layers. For structured datasets, it maintains a master schema blueprint document for rapid metadata lookups alongside tokenized textual chunks in ChromaDB.

Intent-Based Dispatch Router: Uses real-time keyword intent analysis to split system workflows. Analytical queries are routed to an executable code generation matrix, while semantic questions utilize the vector retrieval stream.

Hierarchical Duplicate/Hallucination Avoidance: Avoids common RAG flaws (like the "only 3 rows shown" limitation) by injecting full-data tabular text snapshots when structural entities are targeted.

Self-Correcting Compliance Loop: Employs a strict QA verification engine that intercepts model output to run automated truth checks against data chunks, instantly dropping and logging unverified text loops.

Relational Error Tracking: Features a built-in local SQLite pipeline to log hallucination events or user-flagged errors with contextual metadata snapshotting for admin evaluation.

🏗️ Architecture Overview
The system architecture cleanly decouples unstructured document processing from tabular mathematical execution:

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
🛠️ Technology Stack
Frontend Interface: Streamlit

Orchestration Framework: LangChain / LangChain Core

Inference Engine: ChatGroq (qwen/qwen3-32b at temperature=0.0)

Vector Database: ChromaDB (all-documents collection)

Embedding Pipeline: HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)

Relational Logging: SQLite3

Data Management: Pandas, OpenPyXL (Excel parsing)

🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.9 or higher installed on your local environment.

2. Environment Configuration
Create a .env file in the root directory of your project and include your Groq API token:

Code snippet
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
3. Installation
Install the required system dependencies using pip:

Bash
pip install streamlit pandas docx langchain langchain-chroma langchain-community langchain-groq sentence-transformers python-dotenv openpyxl pypdf
4. Directory Structure setup
Ensure your project workspace contains the following directories:

Plaintext
├── .env
├── app.py                     # Main application script
├── app_logs.db                # Auto-generated SQLite logs
├── chroma_db/                 # Directory containing persistent vector records
└── data/                      # Local data folder where source files are placed
Place any files you want to inspect (e.g., salaries.csv, contract.pdf, meeting_notes.docx) directly inside the data/ folder.

💻 Running the Application
Execute the Streamlit startup workflow from your terminal interface:

Bash
streamlit run app.py
User Workflow Instruction:
Open the sidebar pane to verify database status.

Click Scan & Ingest Directory to parse structural blueprints and document chunks into ChromaDB.

Type natural language or statistical queries into the main chat window.

If the model generates an error or missing detail, click ⚠️ Report Wrong Answer beneath the message bubble to log it permanently for administration review.

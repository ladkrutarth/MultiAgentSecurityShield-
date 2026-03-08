# Veriscan-Cortex — Advanced Fraud Intelligence & Private Multi-Agent Dashboard

> **Course:** CS 5588 — Data Science Capstone | **Date:** February 2026

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Visual Architecture](#visual-architecture)
- [Local AI Intelligence](#local-ai-intelligence)
- [Pipeline Workflow](#pipeline-workflow)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)

---

## Project Overview

Veriscan is an end-to-end **Fraud Detection & Security Platform** that processes transaction data through a multi-stage intelligence pipeline:

**Data Ingestion → Feature Engineering → Hybrid Fraud Modeling → Secure Identity Auth → Private Agentic AI**

### 🛡️ What is Veriscan?
The name **Veriscan** represents the fusion of two core security principles:
- **VERI** (*Verification & Veracity*): A commitment to absolute identity truth through dynamic authentication and data-backed evidence.
- **SCAN** (*Scanning & Surveillance*): The power of autonomous agentic "scans" that explore transaction history, risk profiles, and now personalized financial advice.

### 🌟 Premium AI Specialized Agents
The dashboard features a dual-model specialization for mission-critical tasks:
1. **🛡️ Security AI Analyst**: Dedicated to real-time fraud detection, system shield monitoring, and anomaly detection protocols.
2. **💰 Financial AI Advisor**: A high-fidelity agent that provides comprehensive (>300 word) advisory reports on credit health, savings plans, and spending optimization.
3. **🧬 Spending DNA**: An 8-axis behavioral fingerprinting system for advanced identity verification and trust scoring.
4. **🕵️ Proactive Deception Grid (ADDF)**: Adaptive Deception Defense Framework **covers the entire system**. All traffic passes through ADDF; risk-based routing diverts suspicious sessions to decoy environments (real agents/data never touched). Uses a fast, separate AI model (or template-only) for decoy responses.


## System Architecture

Veriscan-Cortex is built as a layered, event-driven system. **The Deception Grid (ADDF) wraps the entire system**: every request passes through ADDF first; only then is traffic sent to real services or to the decoy environment.

```mermaid
flowchart LR
    subgraph ADDF["🕵️ ADDF — Adaptive Deception Defense Framework"]
        direction LR
        subgraph Ingress [Ingress]
            TxnReq[Transaction]
            LoginReq[Login]
            AdvisorReq[Advisor]
            SecurityReq[Security]
            DNAReq[DNA]
        end

        subgraph Router [Deception Router]
            RiskEval[Risk Evaluator]
            DivertCheck{Divert?}
        end

        subgraph Real [Real Environment]
            RealAPI[Real API]
            GuardAgent[🛡️ GuardAgent]
            FinAdvisor[💰 Financial Advisor]
            DNA[🧬 DNA]
            RAG[RAG]
        end

        subgraph Decoy [Decoy Environment]
            HoneypotAgent[🕵️ HoneypotAgent]
            DecoyTxn[Decoy TXN]
            ThreatLog[Threat Intel]
        end

        Ingress --> RiskEval
        RiskEval --> DivertCheck
        DivertCheck -->|No| RealAPI
        DivertCheck -->|Yes| HoneypotAgent
        RealAPI --> GuardAgent
        RealAPI --> FinAdvisor
        RealAPI --> DNA
        RealAPI --> RAG
        HoneypotAgent --> DecoyTxn
        HoneypotAgent --> ThreatLog
    end
```

### Architecture Layers

| Layer | Components | Responsibility |
|-------|------------|----------------|
| **ADDF (entire system)** | Ingress + Deception Router + Real + Decoy | **Wraps all traffic.** Every request is evaluated by the router; diverted sessions never touch real data. |
| **Ingress** | Transaction, Login, Advisor, Security, DNA requests | All entry points into the system (inside ADDF). |
| **Deception Router** | Risk Evaluator, Session Store, Divert? | Single gate: session risk (e.g. score > 20) → divert to decoy; else → real APIs. |
| **Real Environment** | GuardAgent, Financial Advisor, Spending DNA, RAG | Production agents and data; only reached when session is not diverted. |
| **Decoy (inside ADDF)** | HoneypotAgent, Decoy data, Threat Intel | Isolated responses for diverted sessions; tactic classification, FaaS detection, logging. |

### Data Flow

1. **All traffic inside ADDF**: Every request (transaction, login, advisor chat, security chat, DNA, risk) enters the system through ADDF. There is no path that bypasses the Deception Router.
2. **Normal path**: Request → Deception Router (Risk Evaluator + Session State) → **Not diverted** → Real API → GuardAgent, Financial Advisor, DNA, RAG.
3. **Deception path**: Request → Deception Router → **Diverted** (e.g. first medium+ risk for that session) → HoneypotAgent → Decoy responses + Threat Intel logging. Real agents and data are never used.
4. **Session persistence**: `session_id` (query/header/body) ties diversion state across all endpoints so a diverted session consistently receives decoy data everywhere.


| Step | Layer | What happens |
|------|--------|----------------|
| 1 | **Frontend** | User opens Streamlit (port 8502), chooses Financial / Security / DNA; UI calls `GET /api/health` to show which services are loaded. |
| 2 | **Startup** | FastAPI lifespan loads once: RAG Engine, GuardAgent (LLM), FinancialAdvisorAgent (optional LLM), Spending DNA Agent, ADDF (DeceptionRouter + HoneypotAgent). |
| 3 | **Request** | User sends a message or triggers an action; Streamlit sends the matching REST call (e.g. `POST /api/advisor/chat`, `POST /api/security/chat`, `GET /api/fraud/high-risk`). |
| 4 | **Backend** | API router receives request; session-aware endpoints (fraud, user risk, validate) use `session_id` → Deception Router decides real vs decoy. |
| 5 | **Agents** | Advisor: keyword routing → CSV tools → `_compose_reply` (LLM if available, else template). Security: GuardAgent tools + synthesis. DNA: 8-axis profile / compare. ADDF: decoy data + threat intel. |
| 6 | **Response** | FastAPI returns JSON (reply, tool_results, risk data, etc.); Streamlit renders text, charts, and tables to the user. |


## Visual Architecture

### 🧠 How the AI "Brain" Works
Veriscan-Cortex works like a professional security team. Instead of one slow AI doing everything, we used **specialized agents** that work together in a split second.

```mermaid
graph LR
    User([User Query]) --> ModelSelector{🔍 Model Selector}
    
    ModelSelector -->|Security| SecAnalyst[🛡️ Security Analyst]
    ModelSelector -->|Financial| FinAdvisor[💰 Financial Advisor]

    subgraph Security_Domain [Security]
        direction LR
        SecAnalyst --> Scanner[🔍 Scanner]
        SecAnalyst --> Profile[👤 Investigator]
    end

    subgraph Financial_Domain [Advisory]
        direction LR
        FinAdvisor --> Credit[💳 Credit]
        FinAdvisor --> Savings[💰 Savings]
        FinAdvisor --> Analysis[📈 Analysis]
    end

    Security_Domain --> Report[Security Audit]
    Financial_Domain --> Report2[Advisory Report]
```

#### 🧩 The Roles:
| Agent | Role | "The Personality" |
| :--- | :--- | :--- |
| **Guard (Router)** | The Receptionist | Decides instantly who is best to answer your question. |
| **Scanner** | The Watchman | Scans the whole system for high-risk threats in milliseconds. |
| **Profile** | The Private Eye | Looks deep into a specific user's history and risk scores. |
| **Knowledge** | The Lawyer | Knows all the CFPB rules and fraud theory by heart. |
| **Synthesis** | The Chief Analyst | The "Deep Thinker" that combines all data into a final report. |

#### 🔄 The Process:
1. **Listen:** The **Guard** hears your question.
2. **Assign:** If you ask for a user's risk, the **Profile** agent handles it. If it's a complex "What if?" question, the **Synthesis** agent takes over.
3. **Remember:** The **Memory** system ensures the AI remembers what you talked about earlier.
4. **Report:** You get a professional, data-backed security analysis in seconds.

### 🔍 Multi-Stage RAG Architecture
The RAG system features a **Multi-Stage Retrieval** pipeline over **1,400+ local documents**. It uses semantic search followed by a **Re-ranking Layer** that prioritizes high-confidence Expert Fraud Intelligence (100+ expert QA pairs) over raw transaction context.

```mermaid
graph LR
    subgraph Ingestion [Data Ingestion]
        TXN[(Transactions)]
        CFPB[(CFPB Complaints)]
    end

    subgraph Specialized_Models [AI Brains]
        SEC[🛡️ Security Analyst]
        FIN[💰 Financial Advisor]
    end

    subgraph VectorDB [Semantic Memory]
        Embed[all-MiniLM-L6-v2]
        Chroma[(ChromaDB)]
    end

    subgraph UI [Premium UX]
        Chart[📈 Sunset Charts]
        Text[⬛ Black-Text Labels]
    end

    TXN --> Embed
    CFPB --> Embed
    Embed --> Chroma
    SEC --> VectorDB
    FIN --> VectorDB
    SEC --> UI
    FIN --> UI
```

### 🛡️ Hybrid Fraud Intelligence (ML + Heuristics)
The scoring engine combines 19 statistical "Heuristic Signals" with a supervised **Random Forest Classifier** to learn non-linear fraud signatures.

```mermaid
graph LR
    subgraph Data [Ingestion]
        TXN[(Transactions)]
    end

    subgraph FE [Heuristics]
        direction LR
        Z[Z-Score]
        V[Velocity]
        E[Entropy]
    end

    subgraph ML [Core ML]
        RF[[Random Forest]]
    end

    subgraph Output [Risk Scoring]
        SCORE{Final Score}
    end

    TXN --> FE
    Z --> RF
    V --> RF
    E --> RF
    RF --> SCORE
```

---

## Local AI Intelligence

Veriscan features a cutting-edge, local-first AI stack designed for maximum data privacy and performance on Mac hardware.

- **LLM**: `Meta-Llama-3-8B-Instruct` (4-bit quantized).
- **Inference**: **MLX-LM** (Native GPU acceleration for M1/M2/M3 chips).
- **Embeddings**: `all-MiniLM-L6-v2` (Local execution via `sentence-transformers`).
- **Vector Database**: **ChromaDB** (Persistent local storage for RAG context).

---

## Repository Structure

```
Veriscan-Dashboard/
├── streamlit_app.py                    # Aggregator UI (Consumes Microservices)
├── api/                                # ⚡ FastAPI Microservices Layer
│   ├── main.py                         # REST API Router & Endpoints
│   └── schemas.py                      # Pydantic Data Models
├── Phase-2-Report.md                   # Technical Report
├── CONTRIBUTIONS.md                    # Team Breakdown
├── requirements.txt                    # Project Dependencies
│
├── agents/                             # 🤖 Specialized AI Agents
│   ├── base.py                         # Standardized Agent Interfaces
│   ├── financial_advisor_agent.py      # 💰 Financial Advisor Specialist
│   ├── memory.py                       # 🧠 Stateful Conversation Memory
│   └── spending_dna_agent.py           # 🧬 Behavioral Fingerprinting Agent
│
├── models/                             # Intelligence & Core Logic Layer
│   ├── local_llm.py                    # 🧠 MLX-LM Wrapper (Llama-3)
│   ├── guard_agent_local.py            # 🛡️ Security Analyst Facade
│   ├── rag_engine_local.py             # 🔍 RAG Engine (Local Indexing)
│   └── agent_tools_data.py             # ⚙️ Data Tools for Risk & Profiles
│
├── scripts/                            # Data Pipeline & Synthetic Data
│   ├── feature_engineering.py          # ⚙️ 19 Health Signals
│   ├── fix_agent_data.py               # 🩹 Data Reconciliation Utility
│   ├── generate_cfpb_dataset.py        # 🏦 Synthetic CFPB Compliant Data
│   ├── generate_financial_advisor_dataset.py # 💸 Advisor Context Generator
│   └── generate_spending_dna_dataset.py # 🧬 DNA Vector Generator
│
├── sql/                                # Snowflake SQL Layer
│   ├── create_tables.sql               # 📋 DDL: 5 Tables + 2 Views
│   └── analytical_queries.sql          # 📊 8 Analytical Queries
│
├── dataset/csv_data/                   # Production-Ready Data Store
│   ├── cfpb_credit_card.csv            # CFPB Complaints Base
│   ├── financial_advisor_dataset.csv   # 💰 Advisor Training/RAG Data
│   ├── fraud_detection_qa_dataset.json # 💡 Expert Intelligence Dataset
│   ├── fraud_scores_output.csv         # 🛡️ Hybrid ML Fraud Scores
│   ├── spending_dna_dataset.csv        # 🧬 Spending Fingerprints
│   └── pipeline_logs.csv               # Pipeline Audit Trail
│
└── docs/
    └── architecture_diagram.png        # System Architecture Diagram
```

---

## ☁️ Snowflake Data Platform

Veriscan integrates with **Snowflake** for scalable analytics and data warehousing.

| Table | Purpose |
|-------|--------|
| `RAW_TRANSACTIONS` | Source transaction data |
| `TRANSACTION_FEATURES` | 19 engineered signals |
| `FRAUD_SCORES` | ML + heuristic risk scores |
| `AUTH_PROFILES` | User security profiles |
| `PIPELINE_RUNS` | Pipeline audit trail |

**Views:** `ENRICHED_TRANSACTIONS` (joined data), `USER_RISK_DASHBOARD` (aggregated risk)

See `sql/create_tables.sql` for schema DDL and `sql/analytical_queries.sql` for 8 production-ready queries.

---

## 🚀 Microservices Architecture

Veriscan uses a **decoupled microservices architecture**. The ML, RAG, and Agentic AI components run as a standalone **FastAPI backend**, and the Streamlit dashboard consumes them via REST API.

```mermaid
graph LR
    subgraph Frontend ["🖥️ Streamlit Dashboard (Port 8502)"]
        UI[Dashboard UI]
    end

    subgraph Backend ["⚡ FastAPI Backend (Port 8000)"]
        API[REST API Router]
        Agent[Specialized AI Specialists]
        RAG[RAG Engine + Context Retrieval]
    end

    UI -->|POST /api/advisor/chat| API
    UI -->|POST /api/security/chat| API
    UI -->|POST /api/dna/dna-analysis| API
    UI -->|GET /api/user/ID/risk| API
    API --> Agent
    API --> RAG
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & loaded services (advisor, guard, DNA, ADDF) |
| `POST` | `/api/fraud/predict` | Single-transaction fraud prediction |
| `GET` | `/api/fraud/high-risk?limit=N` | Top N riskiest transactions (session-aware → decoy if diverted) |
| `GET` | `/api/user/{user_id}/risk` | User risk profile (session-aware) |
| `POST` | `/api/rag/query` | Semantic knowledge search (multi-stage re-ranking) |
| `POST` | `/api/advisor/chat` | Financial Advisor Chat (`user_id`, `message` → reply, tool_results) |
| `GET` | `/api/advisor/users` | List user IDs in advisor dataset |
| `POST` | `/api/security/chat` | Security AI Analyst chat (GuardAgent) |
| `GET` | `/api/dna/profile/{user_id}` | Spending DNA 8-axis profile |
| `POST` | `/api/dna/compare` | Compare session vs. DNA baseline |
| `POST` | `/api/transactions/validate` | Validate transaction (ADDF; risk-based diversion) |
| `GET` | `/api/deception/status?session_id=` | Deception session status (diverted, risk_score) |
| `GET` | `/api/honeypot/transactions/{user_id}` | Decoy transactions (diverted sessions) |
| `GET` | `/api/honeypot/filesystem` | Decoy filesystem (diverted sessions) |
| `GET` | `/api/honeypot/threat-intel` | Threat intel logs |
| `GET` | `/api/honeypot/logs` | Honeypot activity logs |

---

## Quick Start

### 1. Requirements
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+ (Anaconda environment recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Data Pipeline
```bash
# Prepare dataset (requires fraudTrain.csv in dataset/csv_data/)
python scripts/prepare_fraud_data.py

# Train the fraud model
python models/train_fraud_model.py

# Sync agent data files
python scripts/fix_agent_data.py
```

### 4. Launch the API Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Launch the Dashboard (separate terminal)
```bash
streamlit run streamlit_app.py --server.port 8502
```
*Note: On first run, the Llama-3 model (~4.9GB) will be downloaded automatically.*

---

## 🔄 Reproducibility & Deployment

| Aspect | Details |
|--------|--------|
| **Environment** | Python 3.9+, dependencies in `requirements.txt` |
| **Model Versioning** | `fraud_model_rf.joblib` + `encoders.joblib` (deterministic `random_state=42`) |
| **Dataset** | Kaggle `kartik2112/fraud-detection` (download separately). Note: The preparation script now utilizes **5x Fraud Oversampling** to ensure sufficient risk events scale for downstream analytics. |
| **Vector Store** | ChromaDB (rebuilt on demand via `rag_engine_local.py`) |
| **Config** | `scripts/ingest_config.yaml` (supports env var overrides) |
| **Secrets** | All credentials via environment variables; `.env` in `.gitignore` |


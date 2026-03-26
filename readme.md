# Career Copilot — Multi-Agent LLM Application

A career management assistant built with **LangGraph** and **LangChain** that uses a multi-agent AI architecture to help users manage job offers, track work experience, generate tailored CVs, and write personalized cover letters — all from a rich terminal interface.

---

## Prerequisites

| Requirement             | Details                                          |
| ----------------------- | ------------------------------------------------ |
| Python                  | 3.12 or higher                                   |
| Docker & Docker Compose | For the PostgreSQL + pgvector database           |
| LaTeX distribution      | For PDF generation (`pdflatex` must be in PATH)  |
| OpenRouter API key      | Or any OpenAI-compatible LLM endpoint            |
| Embedding API key       | For vector similarity search (e.g. Azure OpenAI) |

---

> This application relies on an **AI supervisor architecture**: a central LLM agent routes every user request to the appropriate specialist sub-graph. Because the routing and generation logic depends entirely on the model's reasoning abilities, **the application may not function correctly if the configured model is not sufficiently capable**. Models with strong instruction-following and structured-output capabilities (e.g. DeepSeek V3, GPT-4o, Claude 3.5) are strongly recommended. Weaker or smaller models may fail to route correctly, produce malformed structured outputs, or loop indefinitely.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Projet-LLM
```

### 2. Create a Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Database with Docker

The application uses PostgreSQL 16 with the `pgvector` extension for vector similarity search. Start it with:

```bash
docker compose up -d
```

This launches a container exposing PostgreSQL on port **5433** with:

- **User / Password**: `postgres` / `postgres`
- **Database**: `career-goat`

### 5. Configure Environment Variables

Edit `.env` and fill in your credentials. To change the embedding vector size, set `EMBEDDING_DIM` (default: 4096):

```
# Example .env
AI_API_KEY=your_api_key
AI_ENDPOINT=https://openrouter.ai/api/v1
AI_MODEL=deepseek/deepseek-v3.2

AI_EMBEDDING_API_KEY=your_embedding_api_key
AI_EMBEDDING_ENDPOINT=https://models.inference.ai.azure.com
AI_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=4096  # If you change this, you must reset the database (see below)

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/career-goat

DEBUG=false
```

### 6. Install LaTeX (Required for PDF Generation)

The application generates professional PDFs using `pdflatex`. You must install a LaTeX distribution with the `moderncv` and `fontawesome5` packages.

#### macOS (MacTeX via Homebrew)

```bash
brew install --cask basictex
# Add TeX to PATH (may require restarting the terminal)
sudo tlmgr update --self
sudo tlmgr install moderncv fontawesome5
```

#### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install texlive-full texlive-fonts-extra
```

> `texlive-full` includes `moderncv` and `fontawesome5` by default.

#### Windows (MiKTeX)

1. Download and install MiKTeX from [https://miktex.org/download](https://miktex.org/download)
2. Open **MiKTeX Console → Packages** and install: `moderncv`, `fontawesome5`

### Important: Changing EMBEDDING_DIM

If you change the `EMBEDDING_DIM` value, you must reset and re-initialize the database. Run:

```bash
python -c "from db.db import resetDbAndTables; resetDbAndTables()"
```

Then re-import your data as needed.

### 7. Initialize the Database

```bash
python -c "from db.db import createDbAndTables; createDbAndTables()"
```

## Running the Application

```bash
python main.py
```

## Refrences

https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb

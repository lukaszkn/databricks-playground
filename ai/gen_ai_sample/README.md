# Databricks Generative AI Samples

This repository contains three importable Databricks notebooks demonstrating Generative AI patterns using Foundation Model APIs:

- Python basics: chat, summarization, SQL generation
- Python RAG and embeddings
- SQL-first demo: chat, summarization, NL2SQL, and RAG in SQL

The notebooks are self-contained and create small in-notebook datasets to avoid external dependencies.

## Deliverables

- notebooks/genai_basics.py — Python Databricks notebook (chat, summarization from Spark DataFrame, schema-aware SQL generation with guardrails, optional JSON extraction)
- notebooks/genai_rag.py — Python Databricks notebook (embeddings, cosine similarity retrieval, RAG answer with sources, Mermaid diagram)
- notebooks/genai_sql_demo.sql — SQL-first Databricks notebook (chat, summarization, NL2SQL generation, RAG using SQL AI functions)

## Prerequisites

- A Databricks workspace with access to Foundation Model APIs (FM APIs)
- A running compute:
  - For Python notebooks: a cluster (DBR ML runtime recommended)
  - For SQL-first notebook: a SQL Warehouse or cluster that supports SQL AI functions
- Permissions to call FM endpoints from your workspace

No external provider secrets are required; these demos use Databricks FM APIs via MLflow deployments or SQL AI functions.

## Import and Run

1) Import
- In the Databricks UI, use Workspace → Import, and upload:
  - notebooks/genai_basics.py
  - notebooks/genai_rag.py
  - notebooks/genai_sql_demo.sql

2) Attach compute
- Attach a cluster (for .py notebooks) or a SQL Warehouse/cluster (for .sql notebook)

3) Configure models (optional)
- Each notebook auto-selects from recommended endpoint names or allows easy overrides (see below)

4) Run top-to-bottom
- Start with genai_basics.py, then genai_rag.py, and optionally explore genai_sql_demo.sql

## Model Configuration

The notebooks try a prioritized list of recommended endpoints. If your workspace uses different endpoint names, override them as shown below.

### Python notebooks (genai_basics.py and genai_rag.py)

Near the top of each notebook:

- Set overrides to a known-good endpoint and re-run the cells:

Example (genai_basics.py):
- CHAT_MODEL_OVERRIDE = "databricks-llama-3.1-8b-instruct"
- EMBEDDING_MODEL_OVERRIDE = "databricks-bge-large-en"

Defaults (candidate lists) include model names such as:
- Chat: databricks-llama-3.1-70b-instruct, databricks-llama-3.1-8b-instruct, databricks-mixtral-8x7b-instruct
- Embeddings: databricks-bge-large-en, databricks-gte-large-en

If none of the candidates are available, the notebook will raise a clear error prompting you to set the override variables.

### SQL notebook (genai_sql_demo.sql)

At the top of the notebook, set:
- SET chat_model = 'your-chat-endpoint';
- SET embed_model = 'your-embedding-endpoint';

Then run the cells. You can experiment with different endpoints by changing the variables.

## Notebook Summaries and Usage

### 1) Python Basics — notebooks/genai_basics.py

Sections include:
- Chat completion basics
  - Short system+user prompt, temperature/max_tokens controls
- Summarization from a Spark DataFrame
  - Creates a small dataset of support tickets
  - Summarizes into highlights, risks, and recommended actions
  - Notes on chunking and map-reduce approach for larger inputs
- Schema-aware SQL generation with guardrails
  - Creates a demo table (sales_demo) and a natural language request
  - Prompts the model to return a single SELECT query
  - Validates the generated SQL before execution:
    - Only SELECT statements
    - Reference to a single expected table
  - Executes safely and displays results
- Optional structured JSON extraction
  - Instructs the model to output strict JSON for specific fields
  - Parses JSON and displays as a table
- Configuration and model selection
  - Auto-selects from candidate endpoints
  - Simple override variables

Usage tips:
- Keep temperature low (e.g., 0–0.3) for deterministic SQL or JSON outputs
- Increase max_tokens for longer summaries, mindful of cost
- Adjust candidate endpoints and overrides to match your workspace

### 2) Python RAG and Embeddings — notebooks/genai_rag.py

Sections include:
- Sample corpus creation
  - In-notebook product/FAQ snippets with simple metadata
- Compute embeddings
  - Calls the embedding model and stores vectors in memory
- Similarity search (cosine)
  - Retrieves top-k passages for the user query
- RAG prompt assembly and answer generation
  - Builds a prompt using only the retrieved context
  - Generates a concise answer and prints sources used
- Mermaid diagram
  - Visualizes query → embed → retrieve → compose → generate flow
- Notes and extensions
  - Chunking strategies, persisting to Delta tables, vector search, caching, and evaluation

Usage tips:
- Tweak k for more or fewer context passages
- For larger corpora, store embeddings in a Delta table and use a vector index
- For cost control, embed once and cache vectors

### 3) SQL-first Demo — notebooks/genai_sql_demo.sql

Sections include:
- Chat completion
  - Single prompt with a system-style instruction and user message
- Summarization over a small dataset
  - Builds a support_tickets view and summarizes content
- Natural-language-to-SQL (generation only)
  - Asks the model to output a single ANSI SQL SELECT query for a demo table
  - Copy/paste the generated SQL into a new cell to execute
  - Read-only guidance (avoid DML/DDL)
- RAG in SQL
  - Creates a docs view and computes embeddings via ai_embed(model, text)
  - Uses cosine_similarity to retrieve top-k passages for a user query
  - Builds context and generates an answer using ai_generate_text

Usage tips:
- Use smaller chat models to reduce cost and latency while prototyping
- Keep prompts explicit: ask for “SQL only” and forbid backticks/explanations

## Safety and Guardrails

- Read-only SQL validation in Python basics
  - The sample validator allows only SELECT and restricts to a specific table name
- JSON-only instruction for structured extraction
  - Maintain low temperature and validate parse errors
- RAG prompt explicitly instructs the model to use only the provided context
  - The assistant should respond with “I don’t know” when context is insufficient

For production scenarios, consider:
- Stronger SQL parsers and AST-level validation
- Data masking and access controls
- Safety filters for responses
- Reliability features like retries, timeouts, and observability

## Troubleshooting

- “No available endpoints found”
  - Your workspace may name endpoints differently, or permissions may be missing
  - Set override variables (Python notebooks) or update SET chat_model/embed_model (SQL) to known-good endpoints
- Permission errors when calling models
  - Ensure your user/cluster/warehouse has access to FM APIs in the workspace
- Token/cost concerns
  - Reduce max_tokens, prefer smaller models, or batch requests more efficiently

## Next Steps

- Persist embeddings to a Delta table and index with Vector Search for scale
- Add evaluation workflows (groundedness, factuality, citations)
- Build parameterized jobs and pipelines for batch processing
- Integrate UI components for interactive chat or Q&A
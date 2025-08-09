# Databricks notebook source
# MAGIC %md
# MAGIC # Generative AI on Databricks: Basics (Chat, Summarization, SQL Generation)
# MAGIC
# MAGIC This notebook demonstrates core Generative AI workflows on Databricks using Foundation Model APIs.
# MAGIC It covers:
# MAGIC - Chat completion
# MAGIC - Summarization over a Spark DataFrame
# MAGIC - Schema-aware SQL generation with safe execution
# MAGIC
# MAGIC Import this .py as a Databricks notebook or open directly in Databricks Repos.
# MAGIC Each section includes guidance and notes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Prerequisites and notes
# MAGIC - Workspace must have access to Databricks Foundation Model APIs.
# MAGIC - No external secrets required; we call models via mlflow.deployments.
# MAGIC - You can override model selection in the config cell.
# MAGIC - Calls are minimal by default to reduce cost (small max_tokens).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Setup: client and model candidates
# MAGIC We initialize an MLflow deployments client and define prioritized model lists.

# COMMAND ----------

import json
import math
from typing import List, Dict, Any, Optional

import mlflow
from mlflow.deployments import get_deploy_client

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Initialize deployments client for Databricks Foundation Models
client = get_deploy_client("databricks")

# Prioritized candidate endpoints (adjust as needed for your workspace)
CANDIDATE_CHAT_MODELS: List[str] = [
    "databricks-llama-3.1-70b-instruct",
    "databricks-llama-3.1-8b-instruct",
    "databricks-mixtral-8x7b-instruct",
]

CANDIDATE_EMBEDDING_MODELS: List[str] = [
    "databricks-bge-large-en",
    "databricks-gte-large-en",
]

def _try_chat(endpoint: str) -> bool:
    """Return True if a minimal chat call succeeds."""
    try:
        resp = client.predict(endpoint=endpoint, inputs={
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with the word OK."},
            ],
            "max_tokens": 4,
            "temperature": 0.0,
        })
        # Basic structure check
        return bool(resp)
    except Exception as e:
        return False

def _try_embed(endpoint: str) -> bool:
    """Return True if a minimal embedding call succeeds."""
    try:
        resp = client.predict(endpoint=endpoint, inputs={
            "input": ["hello"]
        })
        return bool(resp)
    except Exception:
        return False

def select_first_available(candidates: List[str], probe_fn) -> Optional[str]:
    for ep in candidates:
        if probe_fn(ep):
            return ep
    return None

# Auto-select defaults (you can override below)
AUTO_CHAT_MODEL = select_first_available(CANDIDATE_CHAT_MODELS, _try_chat)
AUTO_EMBED_MODEL = select_first_available(CANDIDATE_EMBEDDING_MODELS, _try_embed)

# Fallback guidance if none found will be shown later.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Configuration: override models (optional)
# MAGIC You can hardcode overrides below. Leave as None to use auto-selected models.

# COMMAND ----------

# Set overrides here if desired, e.g., CHAT_MODEL_OVERRIDE = "databricks-llama-3.1-8b-instruct"
CHAT_MODEL_OVERRIDE: Optional[str] = None
EMBEDDING_MODEL_OVERRIDE: Optional[str] = None

CHAT_MODEL = CHAT_MODEL_OVERRIDE or AUTO_CHAT_MODEL
EMBEDDING_MODEL = EMBEDDING_MODEL_OVERRIDE or AUTO_EMBED_MODEL

if CHAT_MODEL is None or EMBEDDING_MODEL is None:
    raise RuntimeError(
        "No available Foundation Model endpoints found. "
        "Edit candidate lists or set CHAT_MODEL_OVERRIDE/EMBEDDING_MODEL_OVERRIDE to a known-good endpoint."
    )

# Default generation params (edit as needed)
GEN_PARAMS = {
    "max_tokens": 256,
    "temperature": 0.2,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Utilities
# MAGIC Helper functions for chat, embeddings, and display.

# COMMAND ----------

def extract_assistant_text(resp: Dict[str, Any]) -> str:
    """Best-effort extraction of assistant text from predict() response."""
    # OpenAI-like structure: { choices: [ { message: { content } } ] }
    try:
        choices = resp.get("choices")
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if content:
                return content
    except Exception:
        pass
    # Some models may return { data: [ { text: ... } ] }
    try:
        data = resp.get("data")
        if data and isinstance(data, list):
            first = data[0]
            if isinstance(first, dict):
                if "text" in first:
                    return first["text"]
                if "content" in first:
                    return first["content"]
    except Exception:
        pass
    # Fallback to JSON string
    return json.dumps(resp)

def chat_completion(messages: List[Dict[str, str]], params: Optional[Dict[str, Any]] = None) -> str:
    body = {"messages": messages}
    if params:
        body.update(params)
    resp = client.predict(endpoint=CHAT_MODEL, inputs=body)
    return extract_assistant_text(resp)

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.predict(endpoint=EMBEDDING_MODEL, inputs={"input": texts})
    # Try OpenAI-like 'data': [ { 'embedding': [...] } ]
    if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
        vecs = []
        for item in resp["data"]:
            if isinstance(item, dict) and "embedding" in item:
                vecs.append(item["embedding"])
        if vecs:
            return vecs
    # Otherwise attempt to read 'embeddings' field
    if isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    raise ValueError(f"Unexpected embedding response format: {type(resp)} -> {resp}")

def print_box(title: str, content: str):
    print(f"\n=== {title} ===\n{content}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Selected models
# MAGIC We chose the first responsive endpoint from the candidate lists.

# COMMAND ----------

print_box("Chat model", CHAT_MODEL)
print_box("Embedding model", EMBEDDING_MODEL)
print_box("Chat candidates", "\n".join(CANDIDATE_CHAT_MODELS))
print_box("Embedding candidates", "\n".join(CANDIDATE_EMBEDDING_MODELS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Chat completion basics
# MAGIC A simple system + user message demo with adjustable temperature and max_tokens.

# COMMAND ----------

messages = [
    {"role": "system", "content": "You are a helpful assistant that answers briefly."},
    {"role": "user", "content": "In one sentence, describe what Retrieval-Augmented Generation (RAG) is."},
]
answer = chat_completion(messages, GEN_PARAMS)
print_box("Assistant", answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Summarization from a Spark DataFrame
# MAGIC We'll summarize a handful of support tickets into an executive brief.

# COMMAND ----------

tickets = [
    (1, "Billing", "Overcharged on last invoice; please correct.", "NEGATIVE"),
    (2, "Mobile App", "App crashes when opening settings on Android 14.", "NEGATIVE"),
    (3, "Web", "Feature request: dark mode on dashboard.", "NEUTRAL"),
    (4, "Shipping", "Package arrived two days early. Thanks!", "POSITIVE"),
    (5, "Billing", "Need invoice in PDF and CSV formats.", "NEUTRAL"),
    (6, "Support", "Agent was very helpful resolving my login issue.", "POSITIVE"),
    (7, "Mobile App", "Push notifications do not arrive reliably.", "NEGATIVE"),
    (8, "Web", "Charts rendering slowly on Safari.", "NEGATIVE"),
]
tickets_df = spark.createDataFrame(tickets, ["id", "area", "text", "sentiment"])
display(tickets_df)

# COMMAND ----------

sample_rows = tickets_df.limit(8).collect()
bullet_points = "\n".join([f"- ({r.area}) {r.text} [sentiment={r.sentiment}]" for r in sample_rows])
prompt = (
    "Summarize the following customer support notes into: "
    "(1) 3-5 bullet highlights, (2) key risks, and (3) 1-2 recommended actions.\n\n"
    f"Notes:\n{bullet_points}\n\nBe concise."
)
messages = [
    {"role": "system", "content": "You write concise executive summaries."},
    {"role": "user", "content": prompt},
]
summary = chat_completion(messages, {**GEN_PARAMS, "max_tokens": 300})
print_box("Summary", summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Schema-aware SQL generation with safe execution
# MAGIC We'll ask the model to write a SELECT query for a demo table and validate it before execution.

# COMMAND ----------

from datetime import date

sales = [
    (1, "EMEA", "Laptop Pro 15", 2, 1500.0, "2025-08-01"),
    (2, "EMEA", "Laptop Air 13", 3, 999.0, "2025-08-02"),
    (3, "AMER", "Phone X", 5, 799.0, "2025-08-02"),
    (4, "APAC", "Phone X", 4, 799.0, "2025-08-03"),
    (5, "AMER", "Tablet Plus", 2, 650.0, "2025-08-03"),
    (6, "EMEA", "Laptop Pro 15", 1, 1500.0, "2025-08-03"),
    (7, "APAC", "Laptop Air 13", 2, 999.0, "2025-08-04"),
    (8, "AMER", "Phone X", 3, 799.0, "2025-08-04"),
]
sales_df = spark.createDataFrame(sales, [
    "order_id", "region", "product", "quantity", "unit_price", "order_date"
])
sales_df.createOrReplaceTempView("sales_demo")
display(sales_df)

# COMMAND ----------

schema_desc = ", ".join([f"{f.name}:{f.dataType.simpleString()}" for f in sales_df.schema.fields])
nl_request = (
    "Find the top 3 regions by total revenue and include each region's percent share of total revenue."
)
instruction = (
    "You are a data analyst. Given a table sales_demo with schema: "
    f"{schema_desc}. Write a single ANSI SQL SELECT that answers the question: "
    f"'{nl_request}'. Only reference sales_demo. Output SQL only, no explanations, no backticks."
)
messages = [
    {"role": "system", "content": "You produce correct, minimal SQL."},
    {"role": "user", "content": instruction},
]
sql_text = chat_completion(messages, {**GEN_PARAMS, "max_tokens": 200, "temperature": 0.0})
print_box("Generated SQL", sql_text)

# COMMAND ----------

import re

def validate_sql(sql_text: str, allowed_table: str = "sales_demo") -> str:
    sql = sql_text.strip().strip(";")
    # Must start with SELECT
    if not re.match(r"^(?is)\s*select\s", sql):
        raise ValueError("Only SELECT queries are allowed.")
    # No risky statements
    forbidden = [r"\bdrop\b", r"\bdelete\b", r"\binsert\b", r"\bupdate\b", r"\bcreate\b", r"\balter\b"]
    for pat in forbidden:
        if re.search(pat, sql, flags=re.IGNORECASE):
            raise ValueError(f"Forbidden statement detected: {pat}")
    # Only the allowed table
    if re.search(r"\bfrom\s+([\w\.]+)", sql, flags=re.IGNORECASE):
        m = re.search(r"\bfrom\s+([\w\.]+)", sql, flags=re.IGNORECASE)
        tbl = m.group(1)
        if tbl.lower() != allowed_table.lower():
            raise ValueError(f"Query must reference only table {allowed_table}, got {tbl}")
    return sql

safe_sql = validate_sql(sql_text)
print_box("Validated SQL", safe_sql)
result_df = spark.sql(safe_sql)
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Optional: Structured JSON extraction
# MAGIC Extract structured fields from unstructured text.

# COMMAND ----------

texts = [
    "Customer reports battery overheating during charging on Phone X, requests replacement urgently.",
    "User asked for invoice copy and VAT details; issue is low priority.",
]
schema_hint = {
    "product": "string",
    "category": "string",
    "urgency": "Low|Medium|High",
    "sentiment": "Positive|Negative|Neutral",
}
prompt = (
    "Extract JSON for the following fields: product, category, urgency, sentiment. "
    f"Use this schema hint: {json.dumps(schema_hint)}. "
    "Reply with JSON only, no code fences.\n\n"
    f"Text: {texts[0]}"
)
messages = [
    {"role": "system", "content": "You output strict JSON only."},
    {"role": "user", "content": prompt},
]
raw = chat_completion(messages, {**GEN_PARAMS, "max_tokens": 150, "temperature": 0.0})
print_box("Raw JSON", raw)
try:
    obj = json.loads(raw)
    json_df = spark.createDataFrame([obj])
    display(json_df)
except Exception as e:
    print("Failed to parse JSON; please adjust the instruction or reduce temperature.")
    print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Next steps
# MAGIC - Explore the RAG and Embeddings notebook for retrieval-augmented generation.
# MAGIC - Consider persisting data and prompts for reproducibility.
# MAGIC - Review Databricks docs for productionization patterns (batch, LLM chains, guardrails).
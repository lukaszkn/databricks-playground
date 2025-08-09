# Databricks notebook source
# MAGIC %md
# MAGIC # Generative AI on Databricks: RAG and Embeddings
# MAGIC
# MAGIC This notebook demonstrates a lightweight Retrieval-Augmented Generation (RAG) workflow using Databricks Foundation Model APIs.
# MAGIC It covers:
# MAGIC - Computing embeddings for a small corpus
# MAGIC - Similarity search (cosine) to retrieve top-k context
# MAGIC - Prompt assembly and answer generation using a chat model
# MAGIC - A Mermaid diagram of the flow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Prerequisites and notes
# MAGIC - Workspace must have access to Databricks Foundation Model APIs.
# MAGIC - No external secrets required; models are accessed via mlflow.deployments.
# MAGIC - You can override model selection in the config cell.
# MAGIC - Calls are minimal by default to reduce cost.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Setup: client and model candidates
# MAGIC Initialize MLflow deployments client and define prioritized model lists.

# COMMAND ----------

import json
from typing import List, Dict, Any, Optional, Tuple

import mlflow
from mlflow.deployments import get_deploy_client

# Initialize deployments client for Databricks Foundation Models
client = get_deploy_client("databricks")

# Prioritized candidate endpoints (adjust for your workspace)
CANDIDATE_EMBEDDING_MODELS: List[str] = [
    "databricks-bge-large-en",
    "databricks-gte-large-en",
]

CANDIDATE_CHAT_MODELS: List[str] = [
    "databricks-llama-3.1-70b-instruct",
    "databricks-llama-3.1-8b-instruct",
    "databricks-mixtral-8x7b-instruct",
]

def _try_chat(endpoint: str) -> bool:
    try:
        resp = client.predict(endpoint=endpoint, inputs={
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with OK."},
            ],
            "max_tokens": 4,
            "temperature": 0.0,
        })
        return bool(resp)
    except Exception:
        return False

def _try_embed(endpoint: str) -> bool:
    try:
        resp = client.predict(endpoint=endpoint, inputs={"input": ["hello"]})
        return bool(resp)
    except Exception:
        return False

def select_first_available(candidates: List[str], probe_fn) -> Optional[str]:
    for ep in candidates:
        if probe_fn(ep):
            return ep
    return None

# Auto-select defaults (override below if needed)
AUTO_EMBED_MODEL = select_first_available(CANDIDATE_EMBEDDING_MODELS, _try_embed)
AUTO_CHAT_MODEL = select_first_available(CANDIDATE_CHAT_MODELS, _try_chat)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Configuration: override models (optional)
# MAGIC Provide explicit endpoint names here to override auto-selection.

# COMMAND ----------

EMBEDDING_MODEL_OVERRIDE: Optional[str] = None
CHAT_MODEL_OVERRIDE: Optional[str] = None

EMBEDDING_MODEL = EMBEDDING_MODEL_OVERRIDE or AUTO_EMBED_MODEL
CHAT_MODEL = CHAT_MODEL_OVERRIDE or AUTO_CHAT_MODEL

if EMBEDDING_MODEL is None or CHAT_MODEL is None:
    raise RuntimeError(
        "No available Foundation Model endpoints found. "
        "Edit candidate lists or set *_OVERRIDE variables to known-good endpoints."
    )

# Default generation params
GEN_PARAMS = {
    "max_tokens": 256,
    "temperature": 0.2,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Utilities
# MAGIC Helper functions: response extraction, chat call, embeddings, and cosine similarity.

# COMMAND ----------

def extract_assistant_text(resp: Dict[str, Any]) -> str:
    try:
        choices = resp.get("choices")
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if content:
                return content
    except Exception:
        pass
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
    return json.dumps(resp)

def chat_completion(messages: List[Dict[str, str]], params: Optional[Dict[str, Any]] = None) -> str:
    body = {"messages": messages}
    if params:
        body.update(params)
    resp = client.predict(endpoint=CHAT_MODEL, inputs=body)
    return extract_assistant_text(resp)

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.predict(endpoint=EMBEDDING_MODEL, inputs={"input": texts})
    if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
        vecs = []
        for item in resp["data"]:
            if isinstance(item, dict) and "embedding" in item:
                vecs.append(item["embedding"])
        if vecs:
            return vecs
    if isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    raise ValueError(f"Unexpected embedding response format: {type(resp)} -> {resp}")

def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def norm(a: List[float]) -> float:
    return max((sum(x*x for x in a)) ** 0.5, 1e-12)

def cosine_sim(a: List[float], b: List[float]) -> float:
    return dot(a, b) / (norm(a) * norm(b))

def print_box(title: str, content: str):
    print(f"\n=== {title} ===\n{content}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Selected models
# MAGIC Show which endpoints were selected from the candidate lists.

# COMMAND ----------

print_box("Embedding model", EMBEDDING_MODEL)
print_box("Chat model", CHAT_MODEL)
print_box("Embedding candidates", "\n".join(CANDIDATE_EMBEDDING_MODELS))
print_box("Chat candidates", "\n".join(CANDIDATE_CHAT_MODELS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Sample corpus
# MAGIC We create a tiny corpus of product/FAQ snippets with ids, titles, and sources.

# COMMAND ----------

corpus: List[Dict[str, Any]] = [
    {"id": "doc-1", "title": "Phone X Battery", "text": "Phone X supports fast charging up to 30W. Overheating may occur if used during charging.", "source": "handbook_v1"},
    {"id": "doc-2", "title": "Phone X Notifications", "text": "For reliable push notifications, enable background refresh and allow battery optimization exceptions.", "source": "support_kb"},
    {"id": "doc-3", "title": "Tablet Plus Storage", "text": "Tablet Plus includes 128GB base storage and microSD expansion up to 1TB.", "source": "datasheet"},
    {"id": "doc-4", "title": "Laptop Air 13 Display", "text": "Laptop Air 13 has a 60Hz IPS display; external monitors up to 4K 60Hz via USB-C DP Alt Mode.", "source": "datasheet"},
    {"id": "doc-5", "title": "Laptop Pro 15 Ports", "text": "Laptop Pro 15 provides 2x Thunderbolt 4, HDMI 2.1, and an SD card slot.", "source": "handbook_v2"},
    {"id": "doc-6", "title": "Warranty Policy", "text": "All devices carry a 24-month limited warranty; batteries are covered for 12 months.", "source": "policy"},
    {"id": "doc-7", "title": "Performance Tuning", "text": "Disable unused startup apps and update GPU drivers to improve game performance on Laptop Pro.", "source": "blog"},
    {"id": "doc-8", "title": "Shipping Times", "text": "Standard shipping takes 3-5 business days; expedited options are available at checkout.", "source": "support_kb"},
]

len(corpus)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Compute embeddings
# MAGIC We embed each document text and store vectors in memory for retrieval.

# COMMAND ----------

texts = [d["text"] for d in corpus]
embeddings = embed_texts(texts)
for d, v in zip(corpus, embeddings):
    d["embedding"] = v

print_box("Embedding shape", f"docs={len(corpus)} dim={len(corpus[0]['embedding']) if corpus else 0}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Similarity search helper
# MAGIC Retrieve top-k passages for a user query using cosine similarity.

# COMMAND ----------

from heapq import nlargest

def top_k_retrieve(query: str, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    q_vec = embed_texts([query])[0]
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for doc in corpus:
        s = cosine_sim(q_vec, doc["embedding"])
        scored.append((s, doc))
    return nlargest(k, scored, key=lambda x: x[0])

# Demo retrieval
query = "How can I ensure reliable notifications on Phone X?"
topk = top_k_retrieve(query, k=3)
for rank, (score, doc) in enumerate(topk, start=1):
    print(f"{rank}. score={score:.4f} | {doc['id']} | {doc['title']} | source={doc['source']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) RAG prompt assembly and answer generation
# MAGIC We compose a prompt that includes the retrieved context with source tags and instruct the model to answer concisely using only the provided context.

# COMMAND ----------

def build_context(docs: List[Tuple[float, Dict[str, Any]]]) -> str:
    parts = []
    for score, d in docs:
        parts.append(f"[source={d['source']} id={d['id']} title={d['title']}] {d['text']}")
    return "\n\n".join(parts)

def rag_answer(question: str, k: int = 3) -> Dict[str, Any]:
    docs = top_k_retrieve(question, k=k)
    context = build_context(docs)
    instruction = (
        "You are a product support assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'. Be concise."
    )
    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer:"
    )
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    answer = chat_completion(messages, GEN_PARAMS)
    return {"answer": answer, "docs": docs}

# Demo RAG answer
qa = rag_answer(query, k=3)
print_box("Answer", qa["answer"])
print_box("Sources", "\n".join([f"{d['id']} ({d['source']}) - {d['title']}" for _, d in qa["docs"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Mermaid diagram: RAG flow
# MAGIC The following diagram illustrates the flow from query to answer:
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC   Q[User question] --> E[Embed query]
# MAGIC   E --> S[Similarity search over embeddings]
# MAGIC   S --> C[Compose prompt with top k contexts]
# MAGIC   C --> G[Generate answer with chat model]
# MAGIC   G --> A[Answer with sources]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) Notes and extensions
# MAGIC - Chunk larger documents and store embeddings in a Delta table.
# MAGIC - Use Databricks Vector Search or your own ANN index for scale.
# MAGIC - Add caching of query embeddings and retrieval results.
# MAGIC - Evaluate with held-out Q/A pairs; measure groundedness and citations.
# MAGIC - Add safety filters and content moderation as needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selected models recap
# MAGIC For convenience, we print the selected endpoints again.

# COMMAND ----------

print_box("Embedding model", EMBEDDING_MODEL)
print_box("Chat model", CHAT_MODEL)
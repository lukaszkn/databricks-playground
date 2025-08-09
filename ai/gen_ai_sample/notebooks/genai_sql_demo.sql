-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Generative AI on Databricks: SQL-first Demo
-- MAGIC
-- MAGIC This SQL-first notebook demonstrates:
-- MAGIC - Chat completion with a chosen Foundation Model
-- MAGIC - Summarization over a small support tickets dataset
-- MAGIC - Natural-language-to-SQL generation (copy the result and run)
-- MAGIC - Lightweight RAG using SQL vector functions and embeddings
-- MAGIC
-- MAGIC Notes:
-- MAGIC - Requires a SQL Warehouse or cluster with access to Foundation Model APIs and SQL AI functions.
-- MAGIC - Adjust the model names below to match available endpoints in your workspace.
-- MAGIC - Cells are organized so you can run top-to-bottom.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 0) Configuration
-- MAGIC Set your default chat and embedding models. If unsure, try a smaller instruct model first.

-- COMMAND ----------

SET chat_model = 'databricks-llama-3.1-8b-instruct';
SET embed_model = 'databricks-bge-large-en';

-- Verify settings (optional)
SELECT '${chat_model}' AS chat_model, '${embed_model}' AS embed_model;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1) Chat completion
-- MAGIC A minimal example using a system-style instruction + user question in a single prompt.

-- COMMAND ----------

SELECT
  ai_generate_text(
    '${chat_model}',
    'System: You are a helpful, concise assistant.
User: In one sentence, describe what Retrieval-Augmented Generation (RAG) is.'
  ) AS response;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2) Summarization over a small dataset
-- MAGIC Create a tiny support tickets dataset and summarize it.

-- COMMAND ----------

-- Create a small dataset of support tickets
CREATE OR REPLACE TEMP VIEW support_tickets AS
SELECT 1 AS id, 'Billing' AS area, 'Overcharged on last invoice; please correct.' AS text, 'NEGATIVE' AS sentiment UNION ALL
SELECT 2, 'Mobile App', 'App crashes when opening settings on Android 14.', 'NEGATIVE' UNION ALL
SELECT 3, 'Web', 'Feature request: dark mode on dashboard.', 'NEUTRAL' UNION ALL
SELECT 4, 'Shipping', 'Package arrived two days early. Thanks!', 'POSITIVE' UNION ALL
SELECT 5, 'Billing', 'Need invoice in PDF and CSV formats.', 'NEUTRAL' UNION ALL
SELECT 6, 'Support', 'Agent was very helpful resolving my login issue.', 'POSITIVE' UNION ALL
SELECT 7, 'Mobile App', 'Push notifications do not arrive reliably.', 'NEGATIVE' UNION ALL
SELECT 8, 'Web', 'Charts rendering slowly on Safari.', 'NEGATIVE'
;

SELECT * FROM support_tickets;

-- COMMAND ----------

-- Aggregate notes and summarize via the model
WITH agg AS (
  SELECT concat_ws('\n',
           collect_list(concat('- (', area, ') ', text, ' [sentiment=', sentiment, ']'))
       ) AS notes
  FROM support_tickets
)
SELECT ai_generate_text(
  '${chat_model}',
  concat(
    'System: You write concise executive summaries.',
    '\nUser: Summarize the following customer support notes into: ',
    '(1) 3-5 bullet highlights, (2) key risks, and (3) 1-2 recommended actions.',
    '\n\nNotes:\n', notes,
    '\n\nBe concise.'
  )
) AS summary
FROM agg;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3) Natural language to SQL (generation only)
-- MAGIC Ask the model to produce a single SELECT query for a demo table. Copy the result and run it in a new cell.
-- MAGIC
-- MAGIC Guardrails: in SQL-only mode we cannot easily validate/execute dynamic SQL directly. Use judgment and only run read-only queries you understand.

-- COMMAND ----------

-- Create a small sales table to query
CREATE OR REPLACE TEMP VIEW sales_demo AS
SELECT 1 AS order_id, 'EMEA' AS region, 'Laptop Pro 15' AS product, 2 AS quantity, 1500.0 AS unit_price, '2025-08-01' AS order_date UNION ALL
SELECT 2, 'EMEA', 'Laptop Air 13', 3, 999.0, '2025-08-02' UNION ALL
SELECT 3, 'AMER', 'Phone X', 5, 799.0, '2025-08-02' UNION ALL
SELECT 4, 'APAC', 'Phone X', 4, 799.0, '2025-08-03' UNION ALL
SELECT 5, 'AMER', 'Tablet Plus', 2, 650.0, '2025-08-03' UNION ALL
SELECT 6, 'EMEA', 'Laptop Pro 15', 1, 1500.0, '2025-08-03' UNION ALL
SELECT 7, 'APAC', 'Laptop Air 13', 2, 999.0, '2025-08-04' UNION ALL
SELECT 8, 'AMER', 'Phone X', 3, 799.0, '2025-08-04'
;

SELECT * FROM sales_demo;

-- COMMAND ----------

-- Generate SQL as text. Copy the output SQL and run it in a new cell.
WITH schema_desc AS (
  SELECT 'order_id:int, region:string, product:string, quantity:int, unit_price:double, order_date:string' AS schema_str
)
SELECT ai_generate_text(
  '${chat_model}',
  concat(
    'System: You are a precise data analyst who outputs SQL only (no backticks, no explanations).',
    '\nUser: Given a table sales_demo with schema: ',
    (SELECT schema_str FROM schema_desc),
    '. Write a single ANSI SQL SELECT that answers: ',
    '''Find the top 3 regions by total revenue and include each region''''' , '''s percent share of total revenue.''',
    ' Only reference sales_demo.'
  )
) AS generated_sql;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Example: paste the generated SQL below and run it.
-- MAGIC
-- MAGIC A safe baseline (read-only):
-- MAGIC
-- MAGIC ```sql
-- MAGIC -- SELECT region,
-- MAGIC --        SUM(quantity * unit_price) AS revenue,
-- MAGIC --        100 * SUM(quantity * unit_price) / SUM(SUM(quantity * unit_price)) OVER() AS pct_share
-- MAGIC -- FROM sales_demo
-- MAGIC -- GROUP BY region
-- MAGIC -- ORDER BY revenue DESC
-- MAGIC -- LIMIT 3
-- MAGIC ```

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4) RAG in SQL (embeddings + similarity search)
-- MAGIC We compute document embeddings, retrieve top-k by cosine similarity for a query, and then generate an answer using ONLY the retrieved context.

-- COMMAND ----------

-- Create a tiny corpus
CREATE OR REPLACE TEMP VIEW docs AS
SELECT 'doc-1' AS id, 'Phone X Battery' AS title, 'Phone X supports fast charging up to 30W. Overheating may occur if used during charging.' AS text, 'handbook_v1' AS source UNION ALL
SELECT 'doc-2', 'Phone X Notifications', 'For reliable push notifications, enable background refresh and allow battery optimization exceptions.', 'support_kb' UNION ALL
SELECT 'doc-3', 'Tablet Plus Storage', 'Tablet Plus includes 128GB base storage and microSD expansion up to 1TB.', 'datasheet' UNION ALL
SELECT 'doc-4', 'Laptop Air 13 Display', 'Laptop Air 13 has a 60Hz IPS display; external monitors up to 4K 60Hz via USB-C DP Alt Mode.', 'datasheet' UNION ALL
SELECT 'doc-5', 'Laptop Pro 15 Ports', 'Laptop Pro 15 provides 2x Thunderbolt 4, HDMI 2.1, and an SD card slot.', 'handbook_v2' UNION ALL
SELECT 'doc-6', 'Warranty Policy', 'All devices carry a 24-month limited warranty; batteries are covered for 12 months.', 'policy' UNION ALL
SELECT 'doc-7', 'Performance Tuning', 'Disable unused startup apps and update GPU drivers to improve game performance on Laptop Pro.', 'blog' UNION ALL
SELECT 'doc-8', 'Shipping Times', 'Standard shipping takes 3-5 business days; expedited options are available at checkout.', 'support_kb'
;

SELECT * FROM docs;

-- COMMAND ----------

-- Compute embeddings for the corpus
CREATE OR REPLACE TEMP VIEW docs_embed AS
SELECT
  id, title, text, source,
  ai_embed('${embed_model}', text) AS embedding
FROM docs
;

SELECT id, title, source, length(embedding) AS dim FROM docs_embed;

-- COMMAND ----------

-- Define the user question
SET question = 'How can I ensure reliable notifications on Phone X?';

-- Compute top-k similar documents using cosine similarity
WITH q AS (
  SELECT ai_embed('${embed_model}', '${question}') AS qvec
),
scored AS (
  SELECT
    d.id, d.title, d.text, d.source,
    cosine_similarity(d.embedding, q.qvec) AS score
  FROM docs_embed d CROSS JOIN q
),
topk AS (
  SELECT * FROM scored
  ORDER BY score DESC
  LIMIT 3
)
SELECT * FROM topk;

-- COMMAND ----------

-- Build context and generate the final answer (RAG)
WITH q AS (
  SELECT ai_embed('${embed_model}', '${question}') AS qvec
),
scored AS (
  SELECT
    d.id, d.title, d.text, d.source,
    cosine_similarity(d.embedding, q.qvec) AS score
  FROM docs_embed d CROSS JOIN q
),
topk AS (
  SELECT * FROM scored
  ORDER BY score DESC
  LIMIT 3
),
context AS (
  SELECT concat_ws(
           '\n\n',
           collect_list(concat('[source=', source, ' id=', id, ' title=', title, '] ', text))
         ) AS ctx
  FROM topk
)
SELECT ai_generate_text(
  '${chat_model}',
  concat(
    'System: You are a product support assistant. Answer ONLY using the provided context. ',
    'If insufficient, say: I don''t know.',
    '\nUser: Context:\n', (SELECT ctx FROM context),
    '\n\nQuestion: ', '${question}',
    '\n\nAnswer:'
  )
) AS answer;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5) Mermaid diagram: RAG flow
-- MAGIC
-- MAGIC ```mermaid
-- MAGIC graph TD
-- MAGIC   Q[User question] --> E[ai_embed query]
-- MAGIC   E --> S[cosine_similarity over doc embeddings]
-- MAGIC   S --> C[Compose prompt with top-k contexts]
-- MAGIC   C --> G[ai_generate_text with chat model]
-- MAGIC   G --> A[Answer with sources]
-- MAGIC ```

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6) Next steps
-- MAGIC - Persist embeddings in Delta and use Vector Search for scale.
-- MAGIC - Add caching for embeddings and retrieval results.
-- MAGIC - Evaluate responses for groundedness and citation accuracy.
-- MAGIC - Tune prompts and model parameters to balance cost and quality.
# Optimization Guide

Follow these guidelines to get the best performance out of DuckIR for your use case.

- [Creating the Index](#creating-the-index)
  - [Choosing a Quantization](#choosing-a-quantization-f32-u8-u4)
  - [Choosing a Distance Metric](#choosing-a-distance-metric-cosine-l2sq)
  - [When Your Data Fits in Memory](#when-your-data-fits-in-memory)
- [Querying](#querying)
  - [Tuning `n_probe`](#tuning-n_probe)
  - [Highest Query Throughput](#highest-query-throughput)
  - [Bulk Queries](#bulk-queries-lateral-join)
  - [Slow Queries?](#slow-queries)

## Creating the Index

### Choosing a Quantization (`f32`, `u8`, `u4`)

- **`u8`**: *Our default*, 1 byte per dimension (×4 compression). Within 1% of `f32`'s recall on most modern embeddings.
- **`u4`**: Half a byte per dimension (×8 compression). Smallest index; fastest when the index is read from disk.
  - **When to use**: Your disk is slow (e.g., EBS or hard drive).
  - **When to avoid**: Out-of-distribution queries and [data that drifts](/docs/LIMITATIONS.md) (search quality can degrade).
- **`f32`**: 4 bytes per dimension (no compression).
  - **When to use**: Small datasets where memory or disk space is not a concern.
  - **When to avoid**: Your data is larger than memory, or your disk is slow (e.g., EBS or hard drive).

### Choosing a Distance Metric (`cosine`, `l2sq`)

This depends on your embedding model and sometimes on your use case. We recommend [Weaviate's guide to distance metrics](https://weaviate.io/blog/distance-metrics-in-vector-search). If your embeddings are normalized (most models' are), both metrics rank the same.

The metric must match the distance function in your queries: `l2sq` serves `array_distance` (`<->`), and `cosine` serves `array_cosine_distance` (`<=>`). Otherwise, the search runs without the index.

### When Your Data Fits in Memory

If your data fits in memory, you can deactivate our optimizations for out-of-core execution:
- Deactivate paging before issuing queries: `SET duckir_vss_cluster_paging=false`
- Create your vector index without 1-bit sketches: `CREATE INDEX ... USING DUCKIR_VSS (emb) WITH (cluster_sketches = false);`

#### Considerations

- Note that all queries issued to DuckDB will contend for memory with DuckIR. Therefore, if your workload runs more than vector search, we recommend keeping our defaults.

## Querying

### Tuning `n_probe`

`n_probe` determines how many clusters are probed **per row group**. The default value is 24. `n_probe` can be overridden per session with `SET duckir_vss_n_probe`. A value of `n_probe=0` probes all the data.

What to expect from `n_probe` 24 → 96 with `u8`, with the index in memory:

| Dataset | Recall@10 | p50 latency |
|---|---|---|
| mxbai (769K × 1024) | 0.967 → 0.986 | 1.2× higher |
| jina (1.4M × 768) | 0.923 → 0.968 | 1.2× higher |
| openai5m (5M × 1536) | 0.948 → 0.975 | 1.6× higher |
| cohere10m (10M × 1024) | 0.890 → 0.957 | 1.9× higher |

### Highest Query Throughput

A list literal is the simplest way to write the query vector:

```sql
SELECT id
FROM t
ORDER BY array_distance(
  embedding,
  [0.2502, 0.7944, ..., -0.5050]
)
LIMIT 10;
```

However, DuckDB parses the list literal and binds it element by element. Funnily enough, **that costs more than the vector search itself**. If you want the highest query throughput, pass the list literal as a string to `duckir_vss_encode_query` instead:

```sql
SELECT id
FROM t
ORDER BY array_distance(
  embedding,
  duckir_vss_encode_query('[0.2502, 0.7944, ..., -0.5050]')
)
LIMIT 10;
```

`duckir_vss_encode_query` reads the string and returns its `float32` values as a `BLOB`, which the search uses directly. This bypasses the bind overhead.

From a client, you can instead bind the vector's raw bytes as a parameter. For example, in Python:

```python
import numpy as np

q = np.asarray(vec, dtype='<f4')
con.execute("SELECT id FROM t ORDER BY array_distance(embedding, ?) LIMIT 10", [q.tobytes()])
```

### Bulk Queries (`LATERAL JOIN`)

You can run many queries at once with a `LATERAL JOIN` (a.k.a. NEAREST, SIMILARITY JOIN). This amortizes the cost of DuckDB's query planning:

```sql
-- If queries come from another table
SELECT q.id, s.id
FROM queries q, LATERAL (
  SELECT id
  FROM t1
  WHERE id < 500
  ORDER BY array_distance(
    embedding,
    q.embedding
  )
  LIMIT 100
) s;
```

```sql
-- Or as a list of literals
SELECT qs.qid, d.id, d.title, d.distance
FROM (VALUES
        (1, [0.12, 0.98, ..., 0.33]::FLOAT[128]),
        (2, [0.45, 0.01, ..., 0.77]::FLOAT[128]),
        (3, [0.90, 0.20, ..., 0.05]::FLOAT[128])
     ) qs(qid, q),
     LATERAL (
        SELECT id, title, array_distance(emb, qs.q) AS distance
        FROM docs
        ORDER BY array_distance(emb, qs.q)
        LIMIT 10
     ) d
ORDER BY qs.qid, d.distance;
```

### Slow Queries?

Make sure the index is being used by prepending `EXPLAIN` to your query and checking that a `DUCKIR_VSS` operator is in the plan. If it's missing for a query you think the index should serve, please [open an issue](https://github.com/Noorts/DuckIR/issues).

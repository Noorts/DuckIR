# Out-of-Core

DuckIR can go beyond memory, even on in-memory databases. Out-of-core capabilities are ON by default. If your data fits in memory, you can opt out: `SET duckir_vss_cluster_paging=false`. Note that you must run this command before issuing queries.

At bind time, DuckIR only loads the part of the index that stays in memory (the resident data). As of today, it is mainly:

- Cluster centroids (quantized to 8 bits).
- A mapping from DuckDB row ids to internal ids.
- The trailing row group of the table (see [below](#maintenance)).

**Minimum memory recommendation**: 8 GiB of memory is enough for every index size we've tested, up to 113M 1024-dimensional embeddings (450 GB of embeddings).

## Vector Search Queries

The search over our local indexes works fully out-of-core. Clusters are paged individually when a search needs them.

The PDX library stores clusters in a special order to maximize cluster locality when a block is fetched. See our [vector index](/docs/VECTOR_INDEX.md#cluster-order) doc for more information.

### Why Not 1-bit RaBitQ + Candidate Reranking?

Most modern vector systems (e.g., [Turbopuffer](https://turbopuffer.com/blog/ann-v3), [Infino](https://github.com/infino-ai/infino), [pg_vectorsearch](https://github.com/timescale/pg_vectorsearch)) implement a two-stage pipeline to go out-of-core: RaBitQ estimation (in DRAM) + reranking of $k \times c$ full-precision candidates (fetched from a slower storage unit). This design is wonderful, but in DuckDB it encounters a huge blocker: **DuckDB's smallest fetch granularity from disk is 256 KiB**.

This means that reranking $k \times c$ candidates would take $k \times c$ random fetches of 256 KiB blocks each (about 40-150× read amplification). Ways to circumvent that might be: i) reorder the DuckDB rows after indexing, ii) duplicate the full-precision embeddings in DuckIR storage.

For now, DuckIR focuses on keeping recall as high as possible without a full-precision reranking stage. See our [vector index quantization](/docs/VECTOR_INDEX.md#quantization) doc for more information.

## Cluster Cache

We build an in-memory cache of hot clusters using DuckDB's eviction queues. Hot clusters are clusters that are constantly probed by queries in the current workload. Therefore, **our index leverages the available memory as much as possible.**

## Pruning via 1-bit RaBitQ Sketches

While we cannot leverage a "1-bit RaBitQ + candidate reranking" strategy, we can still avoid fetching many 256 KiB blocks. For this, we use a technique from [RemoteANN](https://arxiv.org/abs/2610.25172). By storing a 1-bit RaBitQ code of every vector, we can determine with high confidence whether the vectors in a cluster can make it into the nearest neighbors of a search.

These 1-bit sketches also live out-of-core and use a tiered cache (hot vs. cold). Under memory pressure, our cache design makes DuckDB evict data in this order: cold clusters (evicted first), hot clusters, cold sketches, hot sketches (evicted last).

Sketches result in 2-24× faster queries under memory pressure, while recall remains unchanged. However, note that sketches are not beneficial when a cluster's data is in memory, because PDX distance kernels are usually as fast as the 1-bit sketch check. Therefore, our default policy for checking sketches is `when_cluster_not_cached`. To turn sketch checks off, run `SET duckir_vss_cluster_sketch_check = 'never';` before issuing queries.

By default, sketches are built and maintained alongside the index, adding some overhead to index creation (+4% for `f32`, +10% for `u8`, +20% for `u4`) and memory usage. If you know your embeddings will fit in memory, you can opt out of building sketches when you create the index: `CREATE INDEX ... USING DUCKIR_VSS (emb) WITH (cluster_sketches = false);`.

Finally, note that sketches need additional space on disk (4% with `f32`, 14% with `u8`, and 28% with `u4`).

## `CREATE INDEX`

Local indexes are created in parallel, one row group per DuckDB worker. Indexes are built as concurrently as `memory_limit` allows. `CREATE INDEX` will fail with an out-of-memory error if not enough memory is available.

Each local index creation requires about the memory of one row group: under 2 GB for the default 122,880 rows, even uncompressed (d < 1536).

After a `CREATE INDEX`, we recommend closing and reopening the database to free the memory DuckDB reserved for scanning the indexed column, which it doesn't count toward `memory_limit` (see [#95](https://github.com/Noorts/DuckIR/pull/95)). You can do this with `DETACH mydb` / `ATTACH 'mydb.duckdb'` or restarting DuckDB.

## Maintenance

Maintenance operations require, at most, the amount of memory of a single row group. In order to maintain `INSERT` performance, we keep the "active" row group in memory. The active row group in DuckDB is the trailing one, which receives new rows on `INSERT`.

## In-Memory Databases

We recommend using DuckDB with a database file. In an in-memory database (`:memory:`), nothing is ever checkpointed, so an index larger than `memory_limit` writes its parts to the temporary files every time DuckDB evicts them.

## +100M Rows: What to Expect

We have tested DuckIR using the full MS MARCO v2.1 corpus, embedded by Cohere, available on [Hugging Face](https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3). This is around 450 GB of embeddings (113M × 1024). We used a Zen 4 CPU with 12 cores and an NVMe disk. Results:

| RAM | Index | `CREATE INDEX` | Search p50 (ms) | p90 | p99 |
|---|---|---|---|---|---|
| 64 GiB | `u8` | 18 min | 79 | 93 | 168 |
| 64 GiB | `u4` | 14 min | 59 | 68 | 107 |
| 8 GiB | `u8` | 27 min | 608 | 718 | 793 |
| 8 GiB | `u4` | 26 min | 813 | 963 | 1,099 |

All searches are measured cold at recall@10 = 0.95.
Note: With only 8 GiB of memory, we had to set DuckDB's `memory_limit` slightly below its default (6 GiB instead of 6.4 GiB). This is because DuckDB's allocator doesn't count towards `memory_limit` what `jemalloc` actually uses to allocate our blocks on Linux. 

# Out-of-Core

DuckIR can go beyond memory, even on in-memory databases. Out-of-core capabilities are ON by default. If your data fits in memory, you can opt out: `SET duckir_vss_cluster_paging=false`. Note that you must run this command before issuing queries.

At bind time, DuckIR only loads the part of the index that stays in memory (the resident data). As of today, it is mainly:   
- Cluster centroids (quantized to 8 bits).
- A mapping from DuckDB row ids to internal ids.

At 250M vectors of 1024 dimensions (1 TB of disk), you would need around 5 GB of memory for DuckIR to work. We are currently working to lower this number to the MB range.

## Vector Search Queries

The search over our local indexes works fully out-of-core. Clusters are paged individually when a search needs them.

The PDX library stores clusters in a special order to maximize cluster locality when a block is fetched. See our [vector index](/docs/VECTOR_INDEX.md#cluster-order) doc for more information.

### Why Not 1-bit RaBitQ + Candidate Reranking?
Most modern vector systems (e.g., [Turbopuffer](https://turbopuffer.com/blog/ann-v3), [Infino](https://github.com/infino-ai/infino), [pg_vectorsearch](https://github.com/timescale/pg_vectorsearch)) implement a two-stage pipeline to go out-of-core: RaBitQ estimation (in DRAM) + reranking of $k \times c$ full-precision candidates (fetched from a slower storage unit). This design is wonderful, but in DuckDB it encounters a huge blocker: **DuckDB's smallest fetch granularity from disk is 256 KiB**.
  
This means that reranking $k \times c$ candidates would take $k \times c$ random fetches of 256 KiB blocks each (about 40-150× read amplification). Ways to circumvent that might be: i) reorder the DuckDB rows after indexing, ii) duplicate the full-precision embeddings in DuckIR storage. 

For now, DuckIR focuses on keeping recall as high as possible without a full-precision reranking stage. See our [vector index quantization](/docs/VECTOR_INDEX.md#quantization) doc for more information.

## Cluster Cache

We build an in-memory cache of hot clusters using DuckDB's eviction queues. Hot clusters are clusters that are constantly probed by queries in the current workload. Therefore, **our index leverages the available memory as much as possible.**

## `CREATE INDEX`

Local indexes are created in parallel, one row group per DuckDB worker. Indexes are built as concurrently as `memory_limit` allows. `CREATE INDEX` will fail with an out-of-memory error if not enough memory is available. 

Each local index creation requires about the memory of one row group: under 2 GB for the default 122,880 rows, even uncompressed (d < 1536). 

## Maintenance

Maintenance operations require, at most, the amount of memory of a single row group. 

## In-Memory Databases

We recommend using DuckDB with a database file. In an in-memory database (`:memory:`), nothing is ever checkpointed, so an index larger than `memory_limit` writes its parts to the temporary files every time DuckDB evicts them. 
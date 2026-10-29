# Out-of-Core

DuckIR can go beyond memory, even on in-memory databases. Out-of-core capabilities are ON by default. If your data fits in memory, you can opt out: `SET duckir_vss_cluster_paging=false`. Note that you must run this command before issuing queries.


## Our Data Layout on Disk

After a `CREATE INDEX`, we store chains of DuckDB segments per local index. A directory (stored in its own chain of segments) keeps pointers to each local index's chain. 

![Persistence with Local Indexes](/docs/diagrams/persistence.png)

At bind time, only the resident data and the directory with the chain of segments are read. The directory stores each row group chain's full segment list and each cluster's byte offset. This allows us to find the segment where a cluster lives without walking the entire chain.

As of today, the resident data is mainly:
- Cluster centroids (quantized to 8 bits).
- A mapping from DuckDB row ids to internal ids.

At 250M vectors of 1024 dimensions (1 TB of disk), you would need around 5 GB of memory for DuckIR to work. We are currently working to lower this number to the MB range.

## CREATE INDEX

Local indexes are created in parallel, one row group per DuckDB worker. Indexes are built as concurrently as `memory_limit` allows. `CREATE INDEX` will fail with an out-of-memory error if not enough memory is available. 

Each local index creation requires about the memory of one row group: under 2 GB for the default 122,880 rows, even uncompressed (d < 1536). 

## Vector Search Queries

The search over our local indexes works fully out-of-core. Clusters are paged individually when a search needs them.

The PDX library stores clusters in a special order to maximize cluster locality when a block is fetched. See our [vector index](/docs/VECTOR_INDEX.md#cluster-order) doc for more information.

## Cluster Cache

We build an in-memory cache of hot clusters using DuckDB's eviction queues. Hot clusters are clusters that are constantly probed by queries in the current workload. Therefore, **our index leverages the available memory as much as possible.**

## Maintenance

Maintenance operations require, at most, the amount of memory of a single row group. 

## In-Memory Databases

We recommend using DuckDB with a database file. In an in-memory database (`:memory:`), nothing is ever checkpointed, so an index larger than `memory_limit` writes its parts to the temporary files every time DuckDB evicts them. 
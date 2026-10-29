# Maintenance

Our vector index updates alongside your table, keeping transactional correctness. Our maintenance is lightweight: 
- `INSERT`s: ~30K / sec
- `DELETE`s: ~7K-300K / sec

Measured on a single-threaded worker on the [Hacker News dataset](https://clickhouse.com/docs/getting-started/example-datasets/hackernews-vector-search-dataset). The performance of `DELETE`s depends on their sparseness. Completely random deletes ran at 7K / sec.

We pull off these numbers by keeping one local index per row group. Insertions only happen in the currently active row group (the last one). As a result, the speed of insertions and deletions remains constant, even in extremely large collections.

Importantly, **a full index rebuild** is never needed because our index mirrors DuckDB's row groups.

## Inserts and Deletes
`DELETE`s are applied to the index when the transaction commits. They are kept as tombstones in each row group. Once more than 10% of a row group's rows are tombstones, we compact the index so that disk space is not wasted.

`INSERT`ed row ids are buffered. They are only indexed by the first search that runs after the commit, one DuckDB row group at a time. This means that the first query after a bulk load pays the indexing cost of the new rows. You can manually trigger a sync by running: `CALL duckir_vss_sync_index('my_idx');`.

## Concurrency

Maintenance operations cannot happen concurrently and exclude all searches while they run. Bulk `INSERT`s use all available DuckDB workers.

## Checkpoint Merges and Dropped Row Groups
When a checkpoint merges or drops row groups, the next search rebuilds the affected part of the index from the table. 


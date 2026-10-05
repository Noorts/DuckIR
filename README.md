<h1 align="center">
  The PDXearch DuckDB Extension
</h1>
<h3 align="center">
  A state-of-the-art IVF index for lightweight but fast (filtered) vector similarity search.
</h3>
<br>

- [Why PDXearch?](#why-pdxearch)
- [Install](#install)
- [Usage](#usage)
  - [Getting Started](#getting-started)
  - [Configuration](#configuration)
    - [Index Creation](#index-creation)
    - [Index Search](#index-search)
    - [Index Metadata](#index-metadata)
  - [Persistence](#persistence)
- [Known Limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Why PDXearch?

DuckDB offers vector similarity search (VSS) out of the box, through its
fixed-size `ARRAY` column type and distance functions
([docs](https://duckdb.org/docs/stable/sql/data_types/array#functions)). These
functions return exact results, but are often too slow on large datasets.

The official DuckDB [VSS extension](https://duckdb.org/docs/stable/core_extensions/vss)
introduces a graph-based (HNSW) VSS index. You can create this index on your
table to speed up the vector search queries. Unfortunately, although
these graph-based indexes deliver fast search, they take up a considerable
amount of memory and take long to construct.

The PDXearch extension aims to address these drawbacks. It achieves competitive
search performance, while using less memory and being significantly faster to
construct. This is made possible by a state-of-the-art partition-based (IVF)
index. To be precise, we rely on the CWI's [PDX](https://github.com/cwida/pdx)
data layout and the accompanying search framework called PDXearch. Furthermore,
this extension integrates tightly with DuckDB's internals to parallelize across
row groups, allowing us to squeeze more performance out of modern hardware.

## Install

> [!WARNING]
> The extension is unstable and experimental. We're actively working on adding
> features and improving stability. The extension will be made available as a
> community extension once it's ready. For now the extension has to be built
> locally.

To build the extension locally, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Usage

### Getting Started

Our syntax is almost identical to that of the
official VSS extension ([VSS docs](https://duckdb.org/docs/stable/core_extensions/vss)).

1. Start a DuckDB instance with an in-memory database and allow loading unsigned extensions.

    ```bash
    duckdb -unsigned
    ```

2. Load the locally built extension by providing a full path to it.

    ```sql
    LOAD '<Fill in>/PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension';
    ```

3. Ensure you have a table with a fixed-sized `FLOAT[num-dims]` column for your embeddings (below called `embedding`). The table can have any number of other columns. Rows whose `embedding` is `NULL` are not indexed, so an index scan never returns them, not even when fewer than K rows with an embedding match the query (a search without the index would list them last).

    ```sql
    CREATE TABLE t1 (id INTEGER, embedding FLOAT[512]);
    ```

    ```sql
    INSERT INTO t1 (id, embedding) SELECT i as id, repeat([i], 512) FROM range(20000) t(i);
    ```

4. Create the PDXearch index to speed up vector similarity search queries. Optionally, configure the index (e.g., the `metric` option). For all options see the [configuration](#configuration) section.

    ```sql
    CREATE INDEX t1_idx ON t1 USING PDXEARCH (embedding) WITH (metric = 'l2sq');
    ```

5. Run an approximate filtered vector similarity search where the top 100 rows are returned.

    ```sql
    SELECT * FROM t1 WHERE id < 500
        ORDER BY array_distance(embedding, repeat([1000.51], 512)::FLOAT[512]) LIMIT 100;
    ```

6. Or send bulk queries using a `LATERAL` join:

    ```sql
    SELECT q.id, s.id FROM queries q, LATERAL (
        SELECT id FROM t1 WHERE id < 500
        ORDER BY array_distance(embedding, q.embedding) LIMIT 100) s;
    ```

### Configuration

#### Index Creation

As shown below, during index creation you can set index creation options in the `WITH` clause. These options cannot be modified after index creation. Drop and recreate the index instead.

```sql
CREATE INDEX t1_idx ON t1 USING PDXEARCH (vec) WITH (metric = 'l2sq', quantization = 'f32');
```

Available options:

- `metric`
  - The distance metric this index speeds up. One index can only optimize one distance metric. If you want two or more distance metrics to be optimized for the same column, then create multiple indexes, where the `metric` option differs.
  - `'l2sq'` (*default*; Euclidean distance, optimizes `array_distance`), `'cosine'` (Cosine similarity distance, optimizes `array_cosine_distance`). Inner product distance is not supported yet.
- `quantization`
  - The precision of the embeddings stored inside the index. Using quantization decreases search latency and index size, but also slightly decreases recall.
  - `f32` (full precision, 4 bytes), `u8` (*default*; scalar quantization, 1 byte).
- `n_probe`
  - Determines the number of partitions/clusters/lists that are explored. Increasing `n_probe` increases the effort spent during a search, thus likely increasing recall, but also increasing search latency. Setting this `n_probe` option will store this value in the index. It can be temporarily overwritten at search time using `pdxearch_n_probe` (see below).
  - `[0, 2147483647]`. *Default* is `24`. This is per row group, which likely has 480 lists. Set `n_probe` to `0` to ensure all clusters are probed. If `n_probe` exceeds the index's number of lists, then all clusters will be probed.
- `seed`
  - The index uses RNG for some internal mechanisms. Set the seed to make behavior reproducible (e.g., for tests or bugs).
  - `[-2147483647, 2147483647]`. *Default* is random.

> [!NOTE]
> There is currently no option to set the number of lists/partitions/clusters manually. We set this [automatically](https://github.com/Noorts/PDXearch/blob/e994ac5f8a99fa3467e18670f4e8056fd5ad9572/src/include/index/pdxearch_wrapper.hpp#L131-L146) based on the row group's size.

#### Index Search

DuckDB by default will use all available threads. To set the number of threads to 1, use `SET threads = 1;`. See the [DuckDB documentation](https://duckdb.org/docs/stable/sql/statements/set) for more information.

Before running a search query, you can set the number of clusters to probe to 48 using `SET pdxearch_n_probe = 48`. This will temporarily overwrite the `n_probe` value stored in the index. Use `RESET pdxearch_n_probe` to unset this overwrite. For more information, see the `n_probe` description above.

The index is used for queries of the form `ORDER BY <distance function>(embedding, <constant query vector>) [ASC] LIMIT k [OFFSET o]`. With an `OFFSET`, the index searches the `k + o` nearest neighbours and DuckDB skips the first `o`. Orders that put `NULL` distances first (`NULLS FIRST`, or `SET default_null_order = 'nulls_first'`), descending orders and additional `ORDER BY` keys run without the index.

Prepend your search query with `EXPLAIN` to show the optimized query plan of your vector search query. Optimized query plans will include a `PDXEARCH_INDEX_SCAN` or a `PDXEARCH_INDEX_FILT_SCAN` operator. For more information about `EXPLAIN` see the [DuckDB documentation](https://duckdb.org/docs/stable/guides/meta/explain).

#### Index Metadata

Execute `CALL pdxearch_index_info();` to print metadata about all PDXearch indexes. This includes an approximate lower bound on the index's size in memory.

Execute `FROM duckdb_indexes();` for general information about all indexes.

### Persistence

In a database file, the index is saved with the database. DuckDB's checkpoints write it, and the first query that uses
it after the database opens loads it into memory. A checkpoint only rewrites the parts of the index (one per DuckDB row
group) that changed since the previous checkpoint. Changes committed after the last checkpoint are replayed from
DuckDB's WAL as usual.

When the index is loaded, it is checked against the table, so it never misses committed rows or returns deleted ones,
even when DuckDB could not replay its log into the index ([duckdb#26112](https://github.com/duckdb/duckdb/issues/26112)).
If the database closes without a checkpoint right after `CREATE INDEX` (for example, after a crash), the index is built
again from the table the first time it is used. An index saved in a storage format this version of PDXearch cannot read
raises an error: drop it and create it again.

## Known Limitations

- **Memory**: A loaded index is held in memory in full; DuckDB cannot evict parts
  of it to disk. The first query after the database opens pays for loading it and
  for checking it against the table.

- **Concurrency**: Any number of KNN queries can search an index
  concurrently, but the maintenance operations exclude all searches while they
  run. `DELETE`s are applied to the index when the transaction commits.
  `INSERT`ed rows are indexed by the first search that runs after the commit
  (or eagerly by `CALL pdxearch_sync_index('index_name');`), one DuckDB row
  group at a time, so the first query after a bulk load pays the indexing cost
  of the new rows (a few seconds per million rows). The index mirrors DuckDB's
  row groups; when a checkpoint merges or drops row groups, the same first
  search rebuilds the affected part of the index from the table.

- **Filter types**: Filtered searches support any predicate on the indexed 
  table's own columns that DuckDB evaluates
  in the table scan or in filter operators directly above it (e.g. comparisons,
  `OR`s across columns, expressions over several columns, `IN` and `NOT IN` lists
  of constants, `IN` and `EXISTS` subqueries), also when the search runs over a
  view or a subquery of the table. Searches that read a value such a subquery
  computes (e.g. `id * 2 AS x`), subqueries whose result is larger than the
  indexed table, correlated subqueries, and joins with other tables still run
  without the index. You can check whether your query is
  currently being optimized by prepending the `EXPLAIN` keyword to your search
  query and checking if a PDXearch operator is part of the query plan.

- **Late materialization**: A search inside a CTE that DuckDB inlines at more
  than one place (e.g. `NOT MATERIALIZED`) still gets DuckDB's late
  materialization rewrite, which returns the same rows but is slower.

- **Maximum `k`**: In `LATERAL` joins, we only support a maximum `k` (`k + o` with an `OFFSET`) of
  `STANDARD_VECTOR_SIZE` (default: 2048).

## Acknowledgements

The extension would not be possible without the underlying technologies and the
lessons learned from other extensions.

- **[PDX](https://github.com/cwida/pdx)**: We use the PDX data layout and
  PDXearch framework.

- **[Super K-Means](https://github.com/lkuffo/SuperKMeans)**: We use
  the Super K-Means library for fast k-means clustering.

- **[VSS](https://github.com/duckdb/duckdb-vss)**: We've taken inspiration from
  the VSS interface and we reuse parts of the VSS extension's code.

## License

The extension is licensed under the [MIT license](LICENSE).

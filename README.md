<h1 align="center">
  DuckIR
</h1>
<h4 align="center">
  Information Retrieval suite for DuckDB
</h4>
<p align="center">
  Vector Search ✅ | Full Text Search ⏱️ | Hybrid Search ⏱️ 
</p>
<br>

## Why DuckIR?

- **Search fully integrated with DuckDB**: Query execution, predicate pushdown, transactional correctness, checkpoints, and crash recovery. 
- **Indexes larger than memory**: We can handle creating indexes and searching over embeddings larger than your memory.
- **Maintenance**: Our vector index updates alongside your table.
- **Morsel-driven parallelism**: Operators work one row group at a time, as DuckDB does. Our extension uses DuckDB's thread pool.
- **Portable**: Linux (x86, ARM), macOS, Wasm, Windows (x86).
- **Fast Vector Indexing and Search**: Index millions of vectors in seconds, search them in milliseconds.   
- **Filtered Search** on any predicate DuckDB can evaluate.
- *FTS and Hybrid Search are WIP*.

## Usage

### From Hugging Face to DuckIR

1. Start a DuckDB instance and allow loading unsigned extensions.

```bash
duckdb -unsigned
```

2. Load the locally built extension by providing a full path to it.

```sql
LOAD '<Fill in>/PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension';
```

3. Create a DuckDB table with an empty vector index
```sql
  CREATE TABLE movies (
    title VARCHAR, 
    genres VARCHAR[], 
    rating DOUBLE, 
    embedding FLOAT[1536]
  );
  CREATE INDEX movies_idx 
  ON movies 
  USING PDXEARCH (embedding) 
    WITH (metric = 'cosine');
```

4. Insert the data directly from Hugging Face:
```sql
  INSERT INTO movies              
      SELECT title, genres, imdb.rating, plot_embedding::FLOAT[1536]
      FROM 'hf://datasets/MongoDB/embedded_movies@~parquet/default/train/0000.parquet';
```

5. Run (filtered) vector search queries. The filters are evaluated first, pushing down the predicates whenever possible:

```sql
  SET VARIABLE q = (
    SELECT embedding 
    FROM movies 
    WHERE title = 'The Matrix'
  );
  -- If q is NULL, the query runs without the vector search

  SELECT title, genres, rating
  FROM movies
  WHERE list_contains(genres, 'Comedy') AND rating > 7
  ORDER BY array_cosine_distance(
    embedding, getvariable('q')
  )
  LIMIT 5;
```

6. Results:

```
The Gods Must Be Crazy         [Action, Comedy]            7.3
True Lies                      [Action, Comedy, Thriller]  7.2
Tai-Chi Master                 [Action, Comedy, Drama]     7.3
The Blind Swordsman: Zatoichi  [Action, Comedy, Crime]     7.6
The Legend of Drunken Master   [Action, Comedy]            7.6
```

7. You can also build an index on a table that already has embeddings in it.

### Vector Index Creation

You can set `CREATE INDEX` options using the `WITH` clause. These options cannot be modified after index creation. Drop and recreate the index instead. 

```sql
CREATE INDEX t1_idx 
ON t1 USING PDXEARCH (embedding) 
WITH (metric = 'l2sq', quantization = 'f32');
```

Available options:

- `metric`
  - `'l2sq'` (*default*; Squared Euclidean Distance, optimizes `array_distance`) 
  - `'cosine'` (Cosine similarity distance, optimizes `array_cosine_distance`). 
- `quantization`: The precision of the embeddings stored inside the index. 
  - `f32` (full-precision, 4 bytes, no compression)
  - `u8` (*default*; scalar quantization, 1 byte, ×4 compression, small quality loss).
- `n_probe`: Increasing `n_probe` increases the effort spent during vector search, thus increasing recall, but also increasing search latency. You can set it at search time using `pdxearch_n_probe` (see below).
  - `[0, 2147483647]`. *Default* is `24`. This is per row group, which likely has 480 lists. Set `n_probe` to `0` to run a brute force search. 
- `seed`
  - `[-2147483647, 2147483647]`. *Default* is random.

### Vector Search

The index is used for queries of the form: 
```sql
SELECT * 
FROM t
[WHERE <predicate>]
ORDER BY <distance_function>(embedding, q) [ASC] 
LIMIT k [OFFSET o]
```  

- `t`: Can be a table, view, or CTE.

- `OFFSET o`: The index searches the `k + o` nearest neighbours and skips the first `o` results.    

You can set the number of clusters to probe on a search using `SET pdxearch_n_probe = 48`. Note that this will overwrite the value for subsequent searches. You can reset it by doing: `RESET pdxearch_n_probe;`.

**Threading**: We use all the threads available in DuckDB. To force a number of threads, use `SET threads = 1;`. 

### Filtered Vector Search


We support arbitrary predicates: **any predicate DuckDB can evaluate on the table's columns**. Including but not limited to: conjunctions, disjunctions, comparisons, `IN`, `NOT IN`, `BETWEEN`, `LIKE`, `EXISTS`, `NOT EXISTS`, expressions over a column, volatile predicates, filter over `embedding[i]`, subqueries as filters, `IN (<subquery>)` (SEMI JOINs), `list_contains`. 

Index search also kicks in if the predicate is rewritten as a `HASH JOIN` (e.g., `WHERE id IN (SELECT item_id FROM purchases)`). Even if it goes out-of-core or the join key is compressed. 

See [#27](https://github.com/Noorts/PDXearch/pull/27) for a full list of supported query shapes.   


### LATERAL JOIN (a.k.a. NEAREST, SIMILARITY JOIN)

You can run many queries at a time

```sql
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

*[EXPERIMENTAL]* Mid-range selectivities are known to be hard for vector search. To tackle this on LATERAL JOINs, we build a temporary index at query time only with the passing tuples. You can enable this feature with `SET pdxearch_experimental_on_the_fly_indexing = 'auto';` option. 

### Index Metadata

Execute `CALL pdxearch_index_info();` to print metadata about all vector indexes. This includes the index's size in-memory.

### Persistence

The index is saved with the database file. DuckDB's checkpoints persist the index, and the first query that uses it after the database opens loads the (small) part of the index that must stay in memory. A checkpoint only rewrites the row groups of the index that changed since the previous checkpoint. Changes committed after the last checkpoint are replayed from DuckDB's WAL into the table, and the index catches up on its next use.

When the index is loaded, it is validated against the table, so it never misses committed rows or returns deleted ones, even when DuckDB could not replay its log into the index ([duckdb#26112](https://github.com/duckdb/duckdb/issues/26112)). If the database closes without a checkpoint right after `CREATE INDEX` (for example, after a crash), the index is built again from the table the first time it is used. 

See [PR#35](https://github.com/Noorts/PDXearch/pull/35) for a more detailed explanation of persistence in DuckIR.

## Operational Notes

- **In-Memory Databases**: We recommend using DuckDB with a database file. In an in-memory database (`:memory:`) nothing is ever checkpointed, so an index larger than `memory_limit` writes its parts to the temporary files every time DuckDB evicts them. 
  
- **Minimum Memory**: Building an index needs room for one DuckDB row group at a time: its float embeddings and about 1.5x its index (about 0.5 GB with `u8` and 0.9 GB with `f32` for 122,880 rows and embeddings of 768 dimensions). Row groups are built as concurrently as `memory_limit` allows. `CREATE INDEX` will fail with an out-of-memory error if not enough memory is available. 

- **Concurrency**: An index can be searched concurrently. However, maintenance operations exclude all searches while they run. 

- **Inserts and Deletes**: `DELETE`s are applied to the index when the transaction commits. `INSERT`ed rows are buffered. They are only indexed by the first search that runs after the commit, one DuckDB row group at a time. Thus, the first query after a bulk load pays the indexing cost of the new rows (a few seconds per million rows). You can manually trigger a sync doing: `CALL pdxearch_sync_index('my_idx');` 

- **Our index mirrors DuckDB's row groups**: when a checkpoint merges or drops row groups, the next search rebuilds the affected part of the index from the table. 

## Limitations

- **Late Materialization on CTEs**: A search inside a CTE that DuckDB inlines at more than one place (e.g. `NOT MATERIALIZED`) loses our opt-out of DuckDB's late materialization (when $k \leq 50$). DuckDB's late materialization adds a second scan of the table, joined back on rowid and sorted again. The results are the same, but slower.

- **Maximum `k`**: In `LATERAL` joins, we only support a maximum `k` (`k + o` with an `OFFSET`) of DuckDB's `STANDARD_VECTOR_SIZE` (default: 2048).

- **Only FLOAT[d]**: We don't support `DOUBLE[n]` or other `ARRAY` types. You can cast with `::FLOAT[n]`, as we do in our Hugging Face example.

- **`INSERT` + `SEARCH` in the same transaction**: An uncommitted `INSERT` will not be reflected in a vector search result.

- **Query vector cannot come from a scalar subquery**: Query vectors must be a constant, a parameter, or: 
```sql
SET VARIABLE q = (SELECT embedding FROM t2 WHERE id = 42); 

SELECT id 
FROM t1
ORDER BY array_distance(
  t1.embedding, 
  getvariable('q')
) LIMIT 100;
```

The following query would not do an index search: 
```sql
SELECT id 
FROM t1
ORDER BY array_distance(
  t1.embedding, 
  (SELECT embedding FROM t2 WHERE id = 42)
) LIMIT 100;
```

- **What needs to fit in memory**: We are working on making DuckIR fully out-of-core. Right now, for vector indexes, the centroids and a rowid mapping must stay resident (about 20 bytes per row at 768 dimensions).

**Query shapes that run without an index:**

- `NULL` first orders (`NULLS FIRST`, or `SET default_null_order = 'nulls_first'`), descending orders
- Additional `ORDER BY` keys
- `IN`/`EXISTS` subqueries whose result is larger than the indexed table.
- Correlated subqueries
- Joins with other tables. This doesn't use the index: `JOIN categories c ON t.cat = c.id WHERE c.name = 'x' ORDER BY ... LIMIT 10`. This will: `WHERE t.cat IN (SELECT id FROM categories WHERE name = 'x')`.
- Searches that read a value that a subquery computes (e.g. `id * 2 AS x`) with $k > 50$.

You can check whether your query is currently being optimized by prepending the `EXPLAIN` keyword to your search query and checking if a `PDXEARCH` operator is part of the query plan.

## Acknowledgements

The extension would not be possible without the underlying technologies and the lessons learned from other extensions.

- **[PDX](https://github.com/cwida/pdx)**: We use the PDX data layout and PDXearch framework.

- **[Super K-Means](https://github.com/lkuffo/SuperKMeans)**: We use the Super K-Means library for fast k-means clustering.

- **[VSS](https://github.com/duckdb/duckdb-vss)**: We've taken inspiration from the VSS interface and we reuse parts of the VSS extension's code.

## License

The extension is licensed under the [MIT license](LICENSE).

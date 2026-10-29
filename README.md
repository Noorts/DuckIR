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
- **Indexes larger than memory**: We handle creating indexes and searching over embeddings beyond memory limits.
- **Maintenance**: Our vector index updates alongside your table. DuckIR never needs full index rebuilds because our index mirrors DuckDB's row groups.
- **Morsel-driven parallelism**: Operators work one row group at a time, as DuckDB does. 
- **Portable**: Linux (x86, ARM), macOS, Wasm, Windows (x86).
- **Fast Vector Indexing and Search**: Index millions of vectors in seconds and search them in milliseconds.   
- **Filtered Search** on any predicate DuckDB can evaluate.
- *FTS and Hybrid Search are WIP*.

> **Status: alpha.** Production use is not advised. Dropping and recreating indexes might be needed between updates. Check our known [limitations](/docs/LIMITATIONS.md).

## Usage

### From Hugging Face to DuckIR

1. Start a DuckDB v2.0.0 instance and allow loading unsigned extensions.

```bash
duckdb -unsigned
```

2. Load the extension you built (see [DEVELOPMENT.md](DEVELOPMENT.md)) by providing a full path to it.

```sql
LOAD '<Fill in>/DuckIR/build/release/extension/duckir/duckir.duckdb_extension';
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
  USING DUCKIR_VSS (embedding) 
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
ON t1 USING DUCKIR_VSS (embedding) 
WITH (metric = 'l2sq', quantization = 'f32');
```

Available options:

- `metric`
  - `'l2sq'` (*default*; squared Euclidean distance, optimizes `array_distance`) 
  - `'cosine'` (cosine similarity, optimizes `array_cosine_distance`). 
- `quantization`: The precision of the embeddings stored inside the index. 
  - `f32`: full precision, no compression.
  - `u8`: *default*; scalar quantization, 1 byte per dimension, ×4 compression; ~1% quality loss relative to `f32`.
  - `u4`: SAQ, 4 bits per dimension, ×8 compression; needs at least 64 dimensions; <1% quality loss relative to `u8` in most embedding datasets (Cohere, MXBAI, OpenAI).
- `n_probe`: Controls the effort spent during vector search: higher values increase recall, but also search latency. You can set it at search time using `duckir_vss_n_probe` (see below).
  - `[0, 2147483647]`. *Default* is `24`. This is per row group, which typically has ~480 lists. Set `n_probe` to `0` to run a brute force search. 
- `seed`
  - `[-2147483648, 2147483647]`. *Default* is random.

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

- `OFFSET o`: The index searches the `k + o` nearest neighbors and skips the first `o` results.    

You can set the number of clusters to probe per search using `SET duckir_vss_n_probe = 48`. Note that this will overwrite the value for subsequent searches. You can reset it by running: `RESET duckir_vss_n_probe;`.

**Threading**: We use all the threads available in DuckDB. To set the number of threads, use `SET threads = 1;`. 

### Filtered Vector Search

We support arbitrary predicates: **any predicate DuckDB can evaluate on the table's columns**, including but not limited to: conjunctions, disjunctions, comparisons, `IN`, `NOT IN`, `BETWEEN`, `LIKE`, `EXISTS`, `NOT EXISTS`, expressions over a column, volatile predicates, filters over `embedding[i]`, subqueries as filters, `IN (<subquery>)` (SEMI JOINs), and `list_contains`. 

Index search also kicks in if the predicate is rewritten as a `HASH JOIN` (e.g., `WHERE id IN (SELECT item_id FROM purchases)`), even if it goes out-of-core or the join key is compressed. 

See [#27](https://github.com/Noorts/DuckIR/pull/27) for a full list of supported query shapes.   


### LATERAL JOIN (a.k.a. NEAREST, SIMILARITY JOIN)

You can run many queries at once:

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

### Index Metadata

Execute `CALL duckir_vss_index_info();` to print metadata about all vector indexes. This includes each index's in-memory size.


### Writing Queries

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
q = np.asarray(vec, dtype='<f4')
con.execute("SELECT id FROM t ORDER BY array_distance(embedding, ?) LIMIT 10", [q.tobytes()])
```

To issue queries in bulk, see [LATERAL JOIN](#lateral-join-aka-nearest-similarity-join).

## Our Vector Index
We create a local index per DuckDB row group, allowing for lightweight maintenance, morsel-driven parallelism, and pushing down SQL predicates onto the vector search. 

Our local indexes are `PDX-IVF` indexes, maintained with the `LIRE` protocol. To learn more about PDX-IVF, check our [Vector Index](/docs/VECTOR_INDEX.md) doc. 

## Maintenance, Persistence and Out-of-Core
See our docs for:
- [Maintenance](/docs/MAINTENANCE.md)
- [Persistence](/docs/PERSISTENCE.md)
- [Out-of-Core](/docs/OUT_OF_CORE.md)

## Limitations
See [LIMITATIONS.md](/docs/LIMITATIONS.md)

## Roadmap
- Q1'2027: Full Text Search and Hybrid Search with local indexes.
- Q1'2027: Beta release
- Q2'2027: Integration in [DuckLake](https://ducklake.select/).

## Acknowledgements

The extension would not be possible without the underlying technologies and the lessons learned from other extensions.

- **[PDX](https://github.com/cwida/pdx)**: Our core vector search library.   
- **[SuperKMeans](https://github.com/lkuffo/SuperKMeans)**: We use the SuperKMeans library for fast k-means clustering.   
- **[DuckDB-VSS](https://github.com/duckdb/duckdb-vss)**: We've taken inspiration from the VSS interface, and we reuse parts of the VSS extension's code.

## License

The extension is licensed under the [MIT license](LICENSE).

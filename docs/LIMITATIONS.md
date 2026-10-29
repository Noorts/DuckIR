# Known Limitations

- **Maximum Dimensionality**: 16,384.   
  
- **Minimum Dimensionality**: 64 for `u4`; no minimum for `u8` and `f32`.

- **No inner product**: `array_inner_product` and `array_negative_inner_product` queries run without the index.

- **`INSERT` + `SEARCH` in the same transaction**: An uncommitted `INSERT` will not be reflected in a vector search result.

- **Late Materialization on CTEs**: A search inside a CTE that DuckDB inlines at more than one place (e.g., `NOT MATERIALIZED`) loses our opt-out of DuckDB's late materialization (when $k \leq 50$). DuckDB's late materialization adds a second scan of the table, joined back on rowid and sorted again. The results are the same, but slower.

- **Maximum `k`**: In `LATERAL` joins, we only support a maximum `k` (`k + o` with an `OFFSET`) of DuckDB's `STANDARD_VECTOR_SIZE` (default: 2048).

- **Only FLOAT[d]**: We don't support `DOUBLE[d]` or other `ARRAY` types. You can cast with `::FLOAT[d]`, as we do in our Hugging Face example.

- **`u4` is NOT drift resilient**: The SAQ index (`u4`) is data-aware: it is fitted once and never refitted. The fit happens at `CREATE INDEX` if the table already has at least 2,048 rows; otherwise, on the rows of the first local index promoted to IVF. Later rows are encoded with that fit, so a strong shift in the data might drop recall. This doesn't happen in `u8` or `f32`, where drift is naturally handled by the fact that `INSERT`s always go into the last row group.

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

### Query shapes that run without an index:

- `NULL` first orders (`NULLS FIRST`, or `SET default_null_order = 'nulls_first'`), descending orders
- Additional `ORDER BY` keys
- `IN`/`EXISTS` subqueries whose result is larger than the indexed table.
- Correlated subqueries
- Joins with other tables. This doesn't use the index: `JOIN categories c ON t.cat = c.id WHERE c.name = 'x' ORDER BY ... LIMIT 10`. This will: `WHERE t.cat IN (SELECT id FROM categories WHERE name = 'x')`.
- Searches that read a value that a subquery computes (e.g., `id * 2 AS x`) with $k > 50$.

You can check whether your query is currently being optimized by prepending the `EXPLAIN` keyword to your search query and checking if a `DUCKIR_VSS` or `Duckir Vss` operator is part of the query plan.

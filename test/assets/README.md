# Test Assets

## Datasets

The SIFT dataset included in this repository has 128 dimensions and 245,760 rows
(2 row groups). The Parquet file has the `id` (0, 1, ...) and `emb` columns. Note
that the dataset's 128-dimensional `emb` array column is composed of integers.
These integers are converted to float when loaded into DuckDB. Details about the
original dataset can be found at
[corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/).

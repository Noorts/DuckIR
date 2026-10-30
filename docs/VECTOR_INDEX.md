# Vector Index

DuckIR creates one local index per DuckDB row group, allowing for lightweight maintenance, morsel-driven parallelism, and pushing down SQL predicates onto the vector search. 

Our local indexes are `PDX-IVF` indexes, maintained with the [LIRE protocol](https://dl.acm.org/doi/pdf/10.1145/3600006.3613166). 

## PDX-IVF

PDX-IVF is an IVF index whose vectors within each cluster are stored in the PDX layout. The [PDX layout](https://dl.acm.org/doi/pdf/10.1145/3725333) is a data layout for vector embeddings specialized for dimension pruning. 

Dimension pruning is useful when clusters are hot in memory, reducing the time to probe a cluster by 2-10×, depending on the embeddings' dimensionality.

One of the main advantages of PDX is that distance kernels are fully intertwined with filtered queries depending on their selectivity, resulting in highly efficient probing of clusters, even at mid-range selectivities.

PDX codebase: https://github.com/cwida/PDX

## Quantization

**Centroids**: In PDX-IVF, centroids are quantized to 8 bits using [LVQ](https://arxiv.org/pdf/2304.04759).  

**Embeddings**: We support three quantization schemes:

### `f32`: No compression
`f32`, short for `float32`, means no compression. Embeddings are stored as-is in the [PDX layout](https://github.com/cwida/PDX#float32).

### `u8`: ×4 compression 
`u8` stores 1 byte per dimension, resulting in a small quality loss (measured at around 1% on modern embedding datasets). `u8` uses the simplest form of scalar quantization: each local index maps its values between the minimum and maximum of its embeddings. 

Embeddings are stored in the [PDX layout for 8-bit values](https://github.com/cwida/PDX#8-bits).

### `u4`: ×8 compression 
`u4` stores 4 bits per dimension. Remarkably, the quality loss is measured at less than 1% relative to `u8` in embedding datasets (Cohere, MXBAI, OpenAI). `u4` is an implementation of [SAQ](https://dl.acm.org/doi/pdf/10.1145/3769824). 

SAQ allocates more bits to more important dimensions. Vectors are PCA-rotated, and then a DP algorithm on a sample chooses how many bits to allocate to each segment.

A [recent survey](https://www.researchgate.net/profile/Mingyu-Yang-26/publication/414393985_A_Survey_of_Vector_Quantization_for_Approximate_Nearest_Neighbor_Search/links/6aab62ad7ea1f607bf9269b3/A-Survey-of-Vector-Quantization-for-Approximate-Nearest-Neighbor-Search.pdf) shows SAQ is SOTA in terms of recall vs. size. However, it loses in speed. 

We have made SAQ's speed competitive:
- [**PDX 8-bit layout**](https://github.com/cwida/PDX#8-bits) in the first SAQ chunk (64 dimensions) for fast distance calculations.
- **PDX-like pruning**: For every chunk of 64 dimensions, we try to prune vectors in a vectorized fashion. 
- Since vectors are PCA-rotated, we can't use our existing ADSampling-based pruning. We incorporate [DADE-based](https://arxiv.org/pdf/2411.17229) pruning. This is faster and more efficient than the pruning strategy presented in SAQ's paper. The pruning bounds are calibrated in `CalibrateSAQPruningBound`.

**Our PDX scan is 1.5× - 3.0× faster** than SAQ's own kernels with AVX-512 across 6 vector embedding datasets (OpenAI/1536, MXBAI/1024, arXiv/768, GIST/960, CLIP/512, and Yandex/200).

Additionally, we opt out of SAQ's dynamic bit allocation at lower dimensionalities ($d < 256$), where we observe SAQ brings no benefit to plain 4-bit RaBitQ. 

## Cluster Order

Clusters are stored in **nearest-neighbor order**, so that a single fetch from storage brings clusters that are potentially needed together for a query. This design was inspired by RemoteANN: https://arxiv.org/abs/2610.25172

## Small Row Groups

When a row group has fewer than 2,048 rows, searches run without an index. This means that all vectors in a row group are always searched (unless the pushed-down predicate skips the row group). Once a row group reaches 2,048 rows, we create an index for it.

## Indexing via Clustering

We use hierarchical $k$-means with SuperKMeans. SuperKMeans uses the PDX layout and dimension pruning ([ADSampling](https://dl.acm.org/doi/pdf/10.1145/3589282)) to accelerate the assignment phase of $k$-means by orders of magnitude. 

Our clustering algorithm is very fast. A row group can be indexed in a few seconds, mostly slowed down by materializing the clusters.

Local indexes are created in parallel, as concurrently as `memory_limit` allows, with one local index build per DuckDB worker.

In fact, our clustering is so fast that it prompted an experimental feature for LATERAL JOIN queries with a filter.

## On-the-fly Index Creation

*[EXPERIMENTAL]* Mid-range selectivities are known to be hard for vector search. To tackle this on LATERAL JOINs, we build a temporary index at query time with only the passing tuples. You can enable this feature with `SET duckir_vss_experimental_on_the_fly_indexing = 'auto';`. It is not available for `u4` indexes.

In the future, we would like this feature to be enabled by default, driven by a cost model.
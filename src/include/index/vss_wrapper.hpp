#pragma once

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/optional_idx.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/indexes/flat.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/ivf_searcher.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/saq.hpp"
#include "index/vss_index_utils.hpp"
#include "index/vss_storage.hpp"

namespace duckdb {

class VSSWrapper {
public:
	static constexpr PDX::DistanceMetric DEFAULT_DISTANCE_METRIC = PDX::DistanceMetric::L2SQ;
	static constexpr PDX::Quantization DEFAULT_QUANTIZATION = PDX::Quantization::U8;
	static constexpr PDX::CentroidQuantization CENTROID_QUANTIZATION = PDX::CentroidQuantization::LVQ8;
	static constexpr uint32_t SAQ_BITS_PER_DIMENSION = 4;
	static constexpr float SAQ_PRUNING_SIGNIFICANCE = 0.01f;
	static constexpr int32_t DEFAULT_N_PROBE = 24; // 5% of the data in a full search
	// Clusters average about 256 rows, one u8 block
	static constexpr uint32_t DEFAULT_MAX_CLUSTER_SIZE = 384;
	static constexpr uint32_t MAX_CLUSTER_SIZE_UPPER_BOUND = 2048;

private:
	const uint32_t num_dimensions;
	// Whether the embeddings (both the query embeddings and those stored in the
	// index) are normalized. The PDXearch kernel only implements Euclidean
	// distance (L2SQ). To compute the cosine and inner product distances we
	// normalize and then use L2SQ.
	const bool is_normalized;

	// The fields below are index options that can be set during index creation:
	// `CREATE INDEX ON t USING DUCKIR_VSS(vec) WITH (n_probe = 10, seed = 42)`.
	// See `create/vss_index_create_plan.cpp` for the validation logic, and
	// `vss_index.cpp` for the usage.
	const PDX::DistanceMetric distance_metric;
	const PDX::Quantization quantization;
	// Between 16 and 128 is common. Setting this to 0 will probe all clusters.
	// Can be set at index build time using the 'n_probe' index option, else
	// uses `DEFAULT_N_PROBE`. At query time the 'duckir_vss_n_probe' runtime
	// setting will take precedence over the n_probe saved here.
	const uint32_t n_probe;
	// Seed that currently affects the random rotation matrix generation.
	const int32_t seed;

	// The clusters of a row group that still receives rows split at this size: the
	// duckir_vss_experimental_max_cluster_size setting when the index was created or loaded.
	const uint32_t max_cluster_size;

protected:
	PDX::MemoryAllocator &memory_allocator;
	unique_ptr<float[]> rotation_matrix;
	unique_ptr<PDX::ADSamplingPruner> pruner;
	std::optional<PDX::SAQModel> saq_model;

public:
	VSSWrapper(PDX::Quantization quantization, PDX::DistanceMetric distance_metric, uint32_t num_dimensions,
	           uint32_t n_probe, int32_t seed, uint32_t max_cluster_size, unique_ptr<float[]> rotation_matrix_p,
	           std::optional<PDX::SAQModel> saq_model_p, PDX::MemoryAllocator &memory_allocator)
	    : num_dimensions(num_dimensions), is_normalized(PDX::DistanceMetricRequiresNormalization(distance_metric)),
	      distance_metric(distance_metric), quantization(quantization), n_probe(n_probe), seed(seed),
	      max_cluster_size(max_cluster_size), memory_allocator(memory_allocator),
	      rotation_matrix(std::move(rotation_matrix_p)), saq_model(std::move(saq_model_p)) {
		if (!rotation_matrix) {
			if (quantization == PDX::Quantization::SAQ) {
				rotation_matrix = GenerateIdentityRotationMatrix(num_dimensions);
			} else {
				rotation_matrix = GenerateRandomRotationMatrix(num_dimensions, seed);
			}
		}
	}
	virtual ~VSSWrapper() = default;

	void CreatePruner() {
		pruner =
		    make_uniq<PDX::ADSamplingPruner>(num_dimensions, rotation_matrix.get(), num_dimensions, &memory_allocator);
	}

	// In place, when the distance metric needs it.
	void NormalizeEmbeddings(float *const embeddings, const size_t num_embeddings) const {
		if (is_normalized) {
			const PDX::Quantizer quantizer(num_dimensions);
			for (size_t i = 0; i < num_embeddings; i++) {
				quantizer.NormalizeQuery(embeddings + i * num_dimensions, embeddings + i * num_dimensions);
			}
		}
	}

	// Warning: modifies the input_embeddings.
	void PreprocessEmbeddings(float *const input_embeddings, float *const output_embeddings,
	                          const size_t num_embeddings, PDX::ParallelExecutor *const executor) const {
		NormalizeEmbeddings(input_embeddings, num_embeddings);
		pruner->PreprocessEmbeddings(input_embeddings, output_embeddings, num_embeddings, executor);
	}

	uint32_t GetNumDimensions() const {
		return num_dimensions;
	}
	PDX::DistanceMetric GetDistanceMetric() const {
		return distance_metric;
	}
	PDX::Quantization GetQuantization() const {
		return quantization;
	}
	bool IsNormalized() const {
		return is_normalized;
	}
	uint32_t GetNProbe() const {
		return n_probe;
	}
	int32_t GetSeed() const {
		return seed;
	}
	uint32_t GetMaxClusterSize() const {
		return max_cluster_size;
	}
	float *GetRotationMatrix() const {
		return rotation_matrix.get();
	}
	const PDX::SAQModel *GetSAQModel() const {
		return saq_model ? &saq_model.value() : nullptr;
	}
	bool NeedsSAQFit() const {
		return quantization == PDX::Quantization::SAQ && !saq_model;
	}

	// The config of every PDX index over this index's rows (the row groups' and the join's on-the-fly ones), which all
	// take transformed embeddings. num_clusters 0 is PDX's default for the number of rows. A null executor is serial.
	PDX::PDXIndexConfig MakePDXIndexConfig(const idx_t num_clusters, PDX::ParallelExecutor *const executor,
	                                       const idx_t base_row_id) const {
		PDX::PDXIndexConfig config;
		config.num_dimensions = num_dimensions;
		config.distance_metric = distance_metric;
		config.seed = static_cast<uint32_t>(seed);
		config.num_clusters = static_cast<uint32_t>(num_clusters);
		config.kmeans_iters = 8;
		config.hierarchical_indexing = true;
		config.executor = executor;
		config.allocator = &memory_allocator;
		config.is_data_transformed = true;
		config.base_row_id = static_cast<size_t>(base_row_id);
		config.centroid_quantization = CENTROID_QUANTIZATION;
		config.saq_pruning_significance = SAQ_PRUNING_SIGNIFICANCE;
		return config;
	}

	// An approximate lower bound of the index's size in memory.
	virtual uint64_t GetInMemorySizeInBytes() const = 0;
	// The part of it outside the memory allocator, which the index reserves.
	virtual uint64_t GetSelfManagedMemorySizeInBytes() const = 0;
};

struct VSSRowGroupBounds {
	row_t row_start;
	idx_t count;
};

struct PDXRowGroup {
	row_t row_start;
	row_t row_end;
	// Null when the index holds its clusters: paging is off, the index is Flat, or the row group is being built or
	// changed. Declared before the index, which points to it, to outlive it.
	unique_ptr<VSSClusterCache> cluster_cache;
	std::unique_ptr<PDX::IPDXIndex> index;
	// Held while the index is built, so that rows of this row group arriving in a later batch wait for it.
	std::mutex mutex;
	// Where the index was last persisted. A checkpoint rewrites only the row groups that changed since.
	VSSBlockChain persisted_chain;
	// Where a paged index changed since the checkpoint lives until the next one. When not empty, it is the current
	// home, and stays it in an in-memory database, where no checkpoint runs.
	VSSTemporaryChain temporary_chain;
	// Deletes since the row group's chain was written. The index bytes still hold these rows.
	idx_t num_unwritten_deletes = 0;
	bool is_dirty = true;
	// DuckDB appends no more rows to the row group (VSSIndex::IsClosedRowGroup).
	bool is_closed = false;

	uint64_t GetInMemorySizeInBytes() const {
		return sizeof(*this) + (cluster_cache ? cluster_cache->GetInMemorySizeInBytes() : 0) +
		       (index ? index->GetInMemorySizeInBytes() : 0);
	}

	uint64_t GetSelfManagedMemorySizeInBytes() const {
		return sizeof(*this) + (cluster_cache ? cluster_cache->GetInMemorySizeInBytes() : 0) +
		       (index ? index->GetSelfManagedMemorySizeInBytes() : 0);
	}
};

// The VSSWrapper for the parallel implementation. The parallel implementation uses a separate index for each row
// group. This allows the creation of the index and searching in it to be parallelized at the row group level.
template <PDX::Quantization Q>
class VSSWrapperParallel : public VSSWrapper {
private:
	const idx_t row_group_size;
	// Sorted by row_start. Structural changes only happen during the index build and under the index's exclusive lock,
	// so scans read it without locking.
	std::vector<unique_ptr<PDXRowGroup>> row_groups;
	// Lock for the above vector structure.
	std::mutex row_groups_mutex;
	// The persisted chains of removed row groups, freed at the next checkpoint.
	std::vector<VSSBlockChain> orphaned_chains;

public:
	// Below this many embeddings a row group is not worth clustering and gets an exact Flat index.
	static constexpr size_t MIN_EMBEDDINGS_FOR_CLUSTERING = 2048;

	// The number of clusters depends on the number of embeddings in the row group. We have three levels.
	// In general we aim for a 1:256 ratio of clusters to embeddings.
	static constexpr size_t ComputeNumClustersForRowGroup(const size_t num_embeddings) {
		if (num_embeddings < MIN_EMBEDDINGS_FOR_CLUSTERING) {
			// 1. No clustering: not worth indexing. One cluster minimizes overhead.
			return 1;
		} else if (num_embeddings < static_cast<size_t>(DEFAULT_ROW_GROUP_SIZE * 0.25)) {
			// 2. Small row group: It has less than 30720 embeddings (25% of a full rowgroup).
			return 120;
		} else {
			// 3. Default: As the DuckDB row group size is usually 122880, we set 480 clusters per row group. While some
			//    row groups might be smaller, 480 is still a good number, even if the row group falls down to 40k
			//    embeddings.
			return 480;
		}
	}

	// Headroom for unaccounted scratches (we are being conservative here).
	static constexpr double BUILD_PEAK_FACTOR = 1.1;
	// Up to 10% of a row group's rows can be deleted since its chain was written without triggering a rewrite.
	static constexpr double MAX_UNWRITTEN_DELETES_RATIO = 0.1;

	uint64_t EstimateBuildPeakBytes(const idx_t num_embeddings) const {
		const idx_t bytes_per_embedding = GetNumDimensions() * sizeof(PDX::pdx_data_t<Q>) + sizeof(uint32_t);
		return static_cast<uint64_t>(BUILD_PEAK_FACTOR * static_cast<double>(num_embeddings * bytes_per_embedding));
	}

	VSSWrapperParallel(PDX::DistanceMetric distance_metric, uint32_t num_dimensions, uint32_t n_probe, int32_t seed,
	                   uint32_t max_cluster_size, idx_t row_group_size, unique_ptr<float[]> rotation_matrix,
	                   std::optional<PDX::SAQModel> saq_model, PDX::MemoryAllocator &memory_allocator)
	    : VSSWrapper(Q, distance_metric, num_dimensions, n_probe, seed, max_cluster_size, std::move(rotation_matrix),
	                 std::move(saq_model), memory_allocator),
	      row_group_size(row_group_size) {
	}

	idx_t GetRowGroupSize() const {
		return row_group_size;
	}

	// Fits the index's SAQ model and rotation on rows in the identity rotation (normalized raw rows), the rotation of
	// an index not fitted yet. The existing Flat indexes are transformed into the fit's PCA-rotated space and marked
	// dirty.
	void FitSAQIfNeeded(const float *const rows, const idx_t num_rows, PDX::ParallelExecutor *const executor) {
		if (!NeedsSAQFit()) {
			return;
		}
		const uint32_t d = GetNumDimensions();
		skmeans::SerialExecutor serial_executor;
		auto fit = PDX::SAQQuantizer::Fit(rows, num_rows, d, SAQ_BITS_PER_DIMENSION, static_cast<uint32_t>(GetSeed()),
		                                  executor ? *executor : serial_executor, memory_allocator);
		auto fitted_rotation = make_uniq_array<float>(static_cast<size_t>(d) * d);
		memcpy(fitted_rotation.get(), fit.rotation.get(), static_cast<size_t>(d) * d * sizeof(float));
		auto fitted_pruner = make_uniq<PDX::ADSamplingPruner>(d, fitted_rotation.get(), d, &memory_allocator);
		std::vector<std::unique_ptr<PDX::IPDXIndex>> rotated_flat_indexes(row_groups.size());
		for (idx_t i = 0; i < row_groups.size(); i++) {
			const auto &row_group = *row_groups[i];
			// The row group being built
			if (!row_group.index) {
				continue;
			}
			const auto flat_index = dynamic_cast<const PDX::FlatIndex *>(row_group.index.get());
			if (!flat_index) {
				throw InternalException(
				    "DuckIR VSS: a row group of this u4 index is already clustered, but the index's "
				    "SAQ model hasn't been fitted. If you see this, please report this bug.");
			}
			const auto row_ids = flat_index->GetRowIds();
			skmeans::AllocatedBuffer<float> rotated;
			if (!row_ids.empty()) {
				const auto embeddings = flat_index->GetEmbeddings();
				rotated = skmeans::AllocateBuffer<float>(memory_allocator, row_ids.size() * d);
				fitted_pruner->PreprocessEmbeddings(embeddings.get(), rotated.get(), row_ids.size(), executor);
			}
			auto rotated_flat_index = make_uniq<PDX::FlatIndex>(
			    MakeRowGroupIndexConfig(row_group.row_start, row_ids.size(), row_group.is_closed, executor),
			    *fitted_pruner);
			rotated_flat_index->BuildIndex(row_ids.data(), rotated.get(), row_ids.size());
			rotated_flat_index->SetExecutor(nullptr);
			rotated_flat_indexes[i] = std::move(rotated_flat_index);
		}
		// Swapped in only once every Flat is rebuilt, so a fit that throws (e.g. out of memory) changes nothing. The
		// indexes on the old pruner go before it.
		for (idx_t i = 0; i < row_groups.size(); i++) {
			if (rotated_flat_indexes[i]) {
				row_groups[i]->index = std::move(rotated_flat_indexes[i]);
				row_groups[i]->is_dirty = true;
			}
		}
		pruner = std::move(fitted_pruner);
		rotation_matrix = std::move(fitted_rotation);
		saq_model = std::move(fit.model);
	}

	// Builds the index of the row group [row_start, row_start + count) from its (non-NULL) rows. Rows of this row
	// group that arrive in a later batch are appended to the index that was already built. The build runs on executor
	// (null: serial). With a reader (duckir_vss_cluster_paging), the index then moves to a temporary chain. Returns how
	// many bytes the row group's self-managed memory grew by.
	int64_t SetUpIndexForRowGroup(const row_t *const row_ids, float *const embeddings, const idx_t num_embeddings,
	                              const row_t row_start, const idx_t count, const bool is_closed,
	                              PDX::ParallelExecutor *const executor, optional_ptr<VSSBlockChainReader> reader,
	                              BufferManager &buffer_manager) {
		D_ASSERT(num_embeddings > 0 && num_embeddings <= count);
		PDXRowGroup *row_group = nullptr;
		std::unique_lock<std::mutex> build_lock;
		{
			const std::lock_guard<std::mutex> lock(row_groups_mutex);
			auto it =
			    std::lower_bound(row_groups.begin(), row_groups.end(), row_start,
			                     [](const unique_ptr<PDXRowGroup> &rg, row_t start) { return rg->row_start < start; });
			// Rare case: If the row group already exists, we will append to it
			// This happens when parallel scan hands rows of the same rowgroup to more than one task
			if (it != row_groups.end() && (*it)->row_start == row_start) {
				row_group = it->get();
			} else {
				// Common path:
				// No rowgroup index exists yet: create one and build the index under its mutex.
				auto new_row_group = make_uniq<PDXRowGroup>();
				new_row_group->row_start = row_start;
				new_row_group->row_end = row_start + static_cast<row_t>(count);
				new_row_group->is_closed = is_closed;
				row_group = new_row_group.get();
				build_lock = std::unique_lock<std::mutex>(row_group->mutex);
				row_groups.insert(it, std::move(new_row_group));
			}
		}

		if (build_lock.owns_lock()) {
			// Only a sync fits here, under the index's exclusive lock: CREATE INDEX fits before its parallel builds
			// whenever a row group could reach MIN_EMBEDDINGS_FOR_CLUSTERING rows (FitSAQModelForCreate).
			if (NeedsSAQFit() && num_embeddings >= MIN_EMBEDDINGS_FOR_CLUSTERING) {
				FitSAQIfNeeded(embeddings, num_embeddings, executor);
				// After fitting, rows need to be re-rotated with the fitted pruner (in place, a block at a time).
				const idx_t num_dimensions = GetNumDimensions();
				auto block = make_uniq_array<float>(STANDARD_VECTOR_SIZE * num_dimensions);
				for (idx_t start = 0; start < num_embeddings; start += STANDARD_VECTOR_SIZE) {
					const idx_t block_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, num_embeddings - start);
					float *const block_rows = embeddings + start * num_dimensions;
					memcpy(block.get(), block_rows, block_size * num_dimensions * sizeof(float));
					pruner->PreprocessEmbeddings(block.get(), block_rows, block_size, executor);
				}
			}
			row_group->index = BuildRowGroupIndex(*row_group, row_ids, embeddings, num_embeddings, executor);
			// The executor ends with the build: later appends and deletes run serially.
			row_group->index->SetExecutor(nullptr);
			if (reader) {
				WriteTemporaryChain(*row_group, *reader, buffer_manager);
			}
			return static_cast<int64_t>(row_group->GetSelfManagedMemorySizeInBytes());
		}
		// Rare path
		// Only reached when DuckDB splits a row group over several scan tasks (e.g. PRAGMA verify_parallelism hands
		// out one vector per task): the first batch built the index above, the later ones are appended to it.
		const std::lock_guard<std::mutex> lock(row_group->mutex);
		// The first batch's build failed: its error ends CREATE INDEX.
		if (!row_group->index) {
			throw Exception(ExceptionType::INDEX, "DuckIR VSS: the index build of a row group failed");
		}
		const auto size_before = static_cast<int64_t>(row_group->GetSelfManagedMemorySizeInBytes());
		if (row_group->cluster_cache) {
			MaterializeRowGroup(*row_group, *reader, buffer_manager);
		}
		for (idx_t i = 0; i < num_embeddings; i++) {
			row_group->index->Append(static_cast<size_t>(row_ids[i]), embeddings + i * GetNumDimensions());
		}
		row_group->is_dirty = true;
		if (reader) {
			WriteTemporaryChain(*row_group, *reader, buffer_manager);
		}
		return static_cast<int64_t>(row_group->GetSelfManagedMemorySizeInBytes()) - size_before;
	}

	unique_ptr<PDX::IIterativeSearch>
	BeginSearchForRowGroup(const idx_t row_group_idx, const float *const preprocessed_query_embedding,
	                       const idx_t limit, PDX::TopKHeap &top_k_heap,
	                       const std::vector<size_t> *const passing_row_ids,
	                       const std::vector<uint32_t> *const clusters_access_order = nullptr,
	                       const PDX::CentroidRankingQuery *const ranking_query = nullptr) {
		auto search_cursor = row_groups[row_group_idx]->index->BeginIterativeSearch(
		    preprocessed_query_embedding, static_cast<uint32_t>(limit), top_k_heap, passing_row_ids,
		    /*is_query_transformed=*/true, clusters_access_order, ranking_query);
		return unique_ptr<PDX::IIterativeSearch>(search_cursor.release());
	}

	// The row group's clusters, nearest to the query first (searches of one query share it).
	std::vector<uint32_t> GetClustersAccessOrderForRowGroup(const idx_t row_group_idx,
	                                                        const float *const preprocessed_query_embedding,
	                                                        const PDX::CentroidRankingQuery *const ranking_query) {
		return row_groups[row_group_idx]->index->GetClustersAccessOrder(preprocessed_query_embedding,
		                                                                /*is_query_transformed=*/true, ranking_query);
	}

	const PDX::IPDXIndex &GetRowGroupIndex(const idx_t row_group_idx) const {
		return *row_groups[row_group_idx]->index;
	}

	// Maintenance, one writer at a time (the index's exclusive lock). The rows belong to the DuckDB row group that
	// `row_groups[row_group_idx]` mirrors; the row group's end grows with them. They are appended in bulk,
	// after moving the ones to append to the front of row_ids' embeddings.
	// If the rowgroup is a Flat index and it has enough embeddings, it is promoted to an IVF index.
	void AppendRows(const idx_t row_group_idx, const row_t *const row_ids, float *const transformed_embeddings,
	                const idx_t count, optional_ptr<VSSBlockChainReader> reader, BufferManager &buffer_manager,
	                PDX::ParallelExecutor *const executor) {
		auto &row_group = *row_groups[row_group_idx];
		const idx_t num_dimensions = GetNumDimensions();
		std::vector<size_t> ids;
		ids.reserve(count);
		row_t max_row_id = row_group.row_start;
		for (idx_t i = 0; i < count; i++) {
			D_ASSERT(row_ids[i] >= row_group.row_start);
			// Rare: a rebuild of this row group (e.g., a checkpoint merge) ran earlier in the same
			// sync and fetched all its committed rows, including (some) staged ones. The staged range now
			// brings them here a second time.
			if (row_group.index->Contains(static_cast<size_t>(row_ids[i]))) {
				continue;
			}
			if (ids.size() != i) {
				memcpy(transformed_embeddings + ids.size() * num_dimensions,
				       transformed_embeddings + i * num_dimensions, num_dimensions * sizeof(float));
			}
			ids.push_back(static_cast<size_t>(row_ids[i]));
			max_row_id = MaxValue<row_t>(max_row_id, row_ids[i]);
		}
		if (ids.empty()) {
			return;
		}
		if (row_group.cluster_cache) {
			MaterializeRowGroup(row_group, *reader, buffer_manager);
		}
		// Before the append: a BulkAppend that throws may have appended part of the rows.
		row_group.row_end = MaxValue<row_t>(row_group.row_end, max_row_id + 1);
		row_group.is_dirty = true;
		// The executor ends with the sync: later appends and deletes run serially.
		row_group.index->SetExecutor(executor);
		try {
			row_group.index->BulkAppend(ids.data(), transformed_embeddings, ids.size());
		} catch (...) {
			row_group.index->SetExecutor(nullptr);
			throw;
		}
		row_group.index->SetExecutor(nullptr);
		auto flat_index = dynamic_cast<PDX::FlatIndex *>(row_group.index.get());
		if (flat_index && flat_index->GetClusterSize(0) >= MIN_EMBEDDINGS_FOR_CLUSTERING) {
			if (NeedsSAQFit()) {
				// The fit rebuilds every Flat index, this one included, in its rotation.
				FitSAQIfNeeded(flat_index->GetEmbeddings().get(), flat_index->GetClusterSize(0), executor);
				flat_index = static_cast<PDX::FlatIndex *>(row_group.index.get());
			}
			PromoteToIVF(row_group, *flat_index, executor);
		}
	}

	void DeleteRow(const row_t row_id) {
		const auto row_group_idx = LookupRowGroup(row_id);
		if (!row_group_idx.IsValid()) {
			return;
		}
		auto &row_group = *row_groups[row_group_idx.GetIndex()];
		// Rows the index does not hold (NULL embeddings, deletes replayed from the WAL) leave the row group clean. The
		// others are persisted only once enough pile up: every load reconciles the index with the table.
		if (row_group.index->Delete(static_cast<size_t>(row_id))) {
			row_group.num_unwritten_deletes++;
			if (static_cast<double>(row_group.num_unwritten_deletes) >
			    MAX_UNWRITTEN_DELETES_RATIO * static_cast<double>(row_group.row_end - row_group.row_start)) {
				row_group.is_dirty = true;
			}
		}
	}

	// Position in `row_groups` of the row group holding row_id.
	optional_idx LookupRowGroup(const row_t row_id) const {
		auto it = std::upper_bound(row_groups.begin(), row_groups.end(), row_id,
		                           [](row_t id, const unique_ptr<PDXRowGroup> &rg) { return id < rg->row_start; });
		if (it == row_groups.begin()) {
			return optional_idx();
		}
		--it;
		if (row_id >= (*it)->row_end) {
			return optional_idx();
		}
		return optional_idx(static_cast<idx_t>(it - row_groups.begin()));
	}

	const PDXRowGroup &GetRowGroup(const idx_t row_group_idx) const {
		return *row_groups[row_group_idx];
	}

	std::vector<VSSRowGroupBounds> FindStaleRowGroups(const std::vector<VSSRowGroupBounds> &physical_row_groups) const {
		std::vector<VSSRowGroupBounds> stale;
		idx_t mirror_idx = 0;
		row_t table_end = 0;
		// The physical row groups are the ones in the DuckDB table, and the mirror row groups are the ones in the
		// index.
		for (const auto &physical : physical_row_groups) {
			const row_t physical_end = physical.row_start + static_cast<row_t>(physical.count);
			table_end = physical_end;
			idx_t overlapping = 0;
			bool aligned = true;
			while (mirror_idx < row_groups.size() && row_groups[mirror_idx]->row_start < physical_end) {
				const auto &mirror = *row_groups[mirror_idx];
				if (mirror.row_end > physical.row_start) {
					overlapping++;
					aligned = aligned && mirror.row_start == physical.row_start && mirror.row_end <= physical_end;
				}
				if (mirror.row_end > physical_end) {
					break; // Also overlaps the next row group, which is then flagged as well.
				}
				mirror_idx++;
			}
			if (overlapping > 1 || (overlapping == 1 && !aligned)) {
				stale.push_back(physical);
			}
		}
		for (; mirror_idx < row_groups.size(); mirror_idx++) {
			const auto &mirror = *row_groups[mirror_idx];
			// Mirroring rowgroups that are beyond the end of the table
			if (mirror.row_start >= table_end) {
				stale.push_back({mirror.row_start, static_cast<idx_t>(mirror.row_end - mirror.row_start)});
			}
		}
		return stale;
	}

	// Marks the row groups DuckDB closed since (is_closed of their row_start), whose clusters then lose their room for
	// appends: now if they are in memory, else when they are next fully loaded (MaterializeRowGroup).
	void MarkClosedRowGroups(const std::function<bool(row_t)> &is_closed) {
		for (auto &row_group : row_groups) {
			if (row_group->is_closed || !is_closed(row_group->row_start)) {
				continue;
			}
			row_group->is_closed = true;
			if (!row_group->cluster_cache) {
				row_group->index->ShrinkToFit();
			}
		}
	}

	void RemoveRowGroupsOverlapping(const row_t start, const row_t end) {
		const std::lock_guard<std::mutex> lock(row_groups_mutex);
		const auto overlaps = [&](const unique_ptr<PDXRowGroup> &row_group) {
			return row_group->row_start < end && row_group->row_end > start;
		};
		// We keep the persisted chains of the removed row groups so that SerializeToDisk can
		// free them later (via SerializeToDisk -> PersistDirtyRowGroups) when it writes the next snapshot.
		for (auto &row_group : row_groups) {
			if (overlaps(row_group) && !row_group->persisted_chain.segments.empty()) {
				orphaned_chains.push_back(std::move(row_group->persisted_chain));
			}
		}
		row_groups.erase(std::remove_if(row_groups.begin(), row_groups.end(), overlaps), row_groups.end());
	}

	// Frees the orphaned chains, then rewrites each dirty row group into a new chain. `write_partial_blocks` runs after
	// each row group, so the allocator never holds more than one rewritten row group in memory. A temporary chain
	// without unwritten deletes is copied as it is. A paged row group is loaded fully first. With page_clusters,
	// each rewritten row group is paged again from its new chain.
	// This method is called from SerializeToDisk, at CHECKPOINT. So this never runs for an :memory: database.
	void PersistDirtyRowGroups(FixedSizeAllocator &allocator, const std::function<void()> &write_partial_blocks,
	                           optional_ptr<VSSBlockChainReader> reader, BufferManager &buffer_manager,
	                           const bool page_clusters) {
		for (auto &chain : orphaned_chains) {
			chain.Free(allocator);
		}
		orphaned_chains.clear();
		for (auto &row_group : row_groups) {
			if (!row_group->is_dirty) {
				continue;
			}
			bool copy_temporary_chain =
			    !row_group->temporary_chain.blocks.empty() && row_group->num_unwritten_deletes == 0;
			// Without the memory to load it fully, the row group is persisted with its unwritten deletes, which the
			// next load reconciles with the table: its temporary chain is copied, or its persisted chain stays. It
			// keeps its home and its tombstones (paging it again would drop them) and stays dirty.
			bool persist_with_unwritten_deletes = false;
			// Before its chain is freed: the full index is read from it.
			if (row_group->cluster_cache && !copy_temporary_chain) {
				try {
					MaterializeRowGroup(*row_group, *reader, buffer_manager);
				} catch (std::exception &ex) {
					if (ErrorData(ex).Type() != ExceptionType::OUT_OF_MEMORY) {
						throw;
					}
					if (row_group->temporary_chain.blocks.empty()) {
						continue;
					}
					copy_temporary_chain = true;
					persist_with_unwritten_deletes = true;
				}
			}
			row_group->persisted_chain.Free(allocator);
			VSSBlockChainWriter writer(allocator);
			VSSChainOutputStream out(writer);
			if (copy_temporary_chain) {
				row_group->temporary_chain.CopyTo(buffer_manager, out);
			} else {
				row_group->index->SaveToStream(out);
			}
			row_group->persisted_chain = writer.Finish();
			if (!persist_with_unwritten_deletes) {
				row_group->temporary_chain = VSSTemporaryChain();
				row_group->num_unwritten_deletes = 0;
				row_group->is_dirty = false;
			}
			write_partial_blocks();
			if (page_clusters) {
				// The new chain has its blocks now, and writing them can have moved other chains' segments.
				reader->UpdateBlockPointers(allocator.GetInfo());
				if (!persist_with_unwritten_deletes) {
					PageRowGroup(*row_group, *reader, buffer_manager);
				}
			}
		}
	}

	// Gives every index whose clusters are on the heap (built or changed by a sync) a temporary chain as its home.
	void WriteTemporaryChains(VSSBlockChainReader &reader, BufferManager &buffer_manager) {
		for (auto &row_group : row_groups) {
			if (!row_group->cluster_cache) {
				WriteTemporaryChain(*row_group, reader, buffer_manager);
			}
		}
	}

	// With each chain's segments, so that a range of a chain can be read without walking it (ReadRange).
	void AddRowGroupEntries(VSSDirectory &directory) const {
		for (const auto &row_group : row_groups) {
			directory.row_groups.push_back({row_group->row_start, row_group->row_end, row_group->persisted_chain});
		}
	}

	// The allocator was reset: nothing is persisted anymore.
	void ResetPersistedChains() {
		orphaned_chains.clear();
		for (auto &row_group : row_groups) {
			row_group->persisted_chain = VSSBlockChain();
			row_group->is_dirty = true;
		}
	}

	// Compares the mirror of the table's row group [start, end) with the row group's committed `row_ids` (ascending).
	// Deletes the rows the mirror holds that the table no longer has, and passes the runs of rows the mirror does not
	// hold to `stage` (all of them, without a mirror).
	void ReconcileRowGroup(const row_t start, const row_t end, const row_t *const row_ids, const idx_t count,
	                       const std::function<void(row_t, row_t)> &stage) {
		const auto row_group_idx = LookupRowGroup(start);
		const auto index = row_group_idx.IsValid() ? row_groups[row_group_idx.GetIndex()]->index.get() : nullptr;
		const auto is_indexed = [&](const row_t row_id) {
			return index && index->Contains(static_cast<size_t>(row_id));
		};
		bool in_missing_run = false;
		row_t missing_run_start = 0;
		idx_t next = 0;
		for (row_t row_id = start; row_id < end; row_id++) {
			const bool in_table = next < count && row_ids[next] == row_id;
			if (in_table) {
				next++;
			}
			const bool indexed = is_indexed(row_id);
			if (!in_table && indexed) {
				DeleteRow(row_id);
			}
			const bool missing = in_table && !indexed;
			if (missing && !in_missing_run) {
				in_missing_run = true;
				missing_run_start = row_id;
			} else if (!missing && in_missing_run) {
				in_missing_run = false;
				stage(missing_run_start, row_id);
			}
		}
		if (in_missing_run) {
			stage(missing_run_start, end);
		}
	}

	// Loads a persisted row group after the ones already loaded: entries come in the directory's (row_start) order.
	// With page_clusters, only its resident data: its clusters stay in the chain until searches need them. Returns its
	// self-managed memory size.
	uint64_t LoadRowGroup(VSSBlockChainReader &reader, const VSSDirectory::RowGroupEntry &entry,
	                      BufferManager &buffer_manager, const bool page_clusters) {
		auto row_group = make_uniq<PDXRowGroup>();
		row_group->row_start = entry.row_start;
		row_group->row_end = entry.row_end;
		row_group->persisted_chain = entry.chain;
		if (page_clusters) {
			PageRowGroup(*row_group, reader, buffer_manager);
		} else {
			MaterializeRowGroup(*row_group, reader, buffer_manager);
		}
		row_group->is_dirty = false;
		const auto self_managed_bytes = row_group->GetSelfManagedMemorySizeInBytes();
		row_groups.push_back(std::move(row_group));
		return self_managed_bytes;
	}

	idx_t GetTotalNumClusters() const {
		idx_t total = 0;
		for (const auto &row_group : row_groups) {
			if (row_group->index) {
				total += row_group->index->GetNumClusters();
			}
		}
		return total;
	}

	idx_t GetNumRowGroups() const {
		return row_groups.size();
	}

	uint64_t GetInMemorySizeInBytes() const override {
		// The rotation matrix, the pruner's copy of it and the SAQ model.
		uint64_t in_memory_size_in_bytes =
		    sizeof(*this) + 2 * static_cast<uint64_t>(GetNumDimensions()) * GetNumDimensions() * sizeof(float) +
		    (saq_model ? saq_model->GetInMemorySizeInBytes() : 0);
		for (const auto &row_group : row_groups) {
			in_memory_size_in_bytes += row_group->GetInMemorySizeInBytes();
		}
		return in_memory_size_in_bytes;
	}

	uint64_t GetSelfManagedMemorySizeInBytes() const override {
		// The rotation matrix and the SAQ model. The pruner's copy of the rotation comes from the memory allocator.
		uint64_t self_managed_bytes = sizeof(*this) +
		                              static_cast<uint64_t>(GetNumDimensions()) * GetNumDimensions() * sizeof(float) +
		                              (saq_model ? saq_model->GetInMemorySizeInBytes() : 0);
		for (const auto &row_group : row_groups) {
			self_managed_bytes += row_group->GetSelfManagedMemorySizeInBytes();
		}
		return self_managed_bytes;
	}

private:
	PDX::PDXIndexConfig MakeRowGroupIndexConfig(const row_t row_start, const idx_t num_embeddings, const bool is_closed,
	                                            PDX::ParallelExecutor *const executor) const {
		auto config =
		    MakePDXIndexConfig(ComputeNumClustersForRowGroup(num_embeddings), executor, static_cast<idx_t>(row_start));
		if (is_closed) {
			config.expect_appends = false;
		} else {
			config.max_cluster_size = GetMaxClusterSize();
		}
		return config;
	}

	unique_ptr<PDX::PDXIndex<Q>> MakeIVFIndex(const PDX::PDXIndexConfig &config) const {
		if constexpr (Q == PDX::SAQ) {
			if (!saq_model) {
				throw InternalException(
				    "DuckIR VSS: a row group of this u4 index is being clustered, but the index's SAQ "
				    "model hasn't been fitted. If you see this, please report this bug.");
			}
			return make_uniq<PDX::PDXIndex<Q>>(config, *pruner, *saq_model);
		} else {
			return make_uniq<PDX::PDXIndex<Q>>(config, *pruner);
		}
	}

	unique_ptr<PDX::IPDXIndex> BuildRowGroupIndex(const PDXRowGroup &row_group, const row_t *const row_ids,
	                                              const float *const embeddings, const idx_t num_embeddings,
	                                              PDX::ParallelExecutor *const executor) const {
		const auto config = MakeRowGroupIndexConfig(row_group.row_start, num_embeddings, row_group.is_closed, executor);
		std::vector<size_t> ids(num_embeddings);
		for (idx_t i = 0; i < num_embeddings; i++) {
			ids[i] = static_cast<size_t>(row_ids[i]);
		}

		if (num_embeddings < MIN_EMBEDDINGS_FOR_CLUSTERING) {
			auto flat_index = make_uniq<PDX::FlatIndex>(config, *pruner);
			flat_index->BuildIndex(ids.data(), embeddings, num_embeddings);
			return std::move(flat_index);
		}
		auto ivf_index = MakeIVFIndex(config);
		ivf_index->BuildIndex(ids.data(), embeddings, num_embeddings);
		return std::move(ivf_index);
	}

	void PromoteToIVF(PDXRowGroup &row_group, const PDX::FlatIndex &flat_index, PDX::ParallelExecutor *const executor) {
		const auto row_ids = flat_index.GetRowIds();
		const auto embeddings = flat_index.GetEmbeddings();
		auto ivf_index =
		    MakeIVFIndex(MakeRowGroupIndexConfig(row_group.row_start, row_ids.size(), row_group.is_closed, executor));
		ivf_index->BuildIndex(row_ids.data(), embeddings.get(), row_ids.size());
		// The executor ends with the build: later appends and deletes run serially.
		ivf_index->SetExecutor(nullptr);
		row_group.index = std::move(ivf_index);
	}

	// Moves an IVF index on the heap into a new temporary chain, and pages it from there. A Flat index stays on the
	// heap: paging would load it fully again.
	void WriteTemporaryChain(PDXRowGroup &row_group, VSSBlockChainReader &reader, BufferManager &buffer_manager) {
		if (dynamic_cast<PDX::FlatIndex *>(row_group.index.get())) {
			return;
		}
		VSSTemporaryChainWriter writer(buffer_manager);
		VSSChainOutputStream out(writer);
		row_group.index->SaveToStream(out);
		row_group.temporary_chain = writer.Finish();
		row_group.num_unwritten_deletes = 0;
		PageRowGroup(row_group, reader, buffer_manager);
	}

	// Loads the row group's index from its home with only its resident data: searches read its clusters from the home
	// through a new cluster cache. A Flat index is read fully.
	void PageRowGroup(PDXRowGroup &row_group, VSSBlockChainReader &reader, BufferManager &buffer_manager) {
		auto cluster_cache = make_uniq<VSSClusterCache>(reader, buffer_manager);
		std::unique_ptr<PDX::IPDXIndex> index;
		if (!row_group.temporary_chain.blocks.empty()) {
			VSSTemporaryChainReader temporary_reader(buffer_manager, row_group.temporary_chain);
			VSSChainInputStream in(temporary_reader);
			index = PDX::LoadPDXIndexFromStream(in, *pruner, cluster_cache.get(), &memory_allocator);
			// IVF index? The reader stopped where the non-resident data starts.
			if (temporary_reader.GetBytesRead() < row_group.temporary_chain.num_bytes) {
				cluster_cache->Bind(*index, row_group.temporary_chain, temporary_reader.GetBytesRead());
			} else { // Flat index? The reader stopped at the end of the chain.
				cluster_cache.reset();
			}
		} else {
			reader.Open(row_group.persisted_chain);
			VSSChainInputStream in(reader);
			index = PDX::LoadPDXIndexFromStream(in, *pruner, cluster_cache.get(), &memory_allocator);
			if (reader.GetBytesRead() < row_group.persisted_chain.num_bytes) {
				cluster_cache->Bind(*index, row_group.persisted_chain, reader.GetBytesRead());
				reader.Close();
			} else {
				cluster_cache.reset();
				reader.Finish();
			}
		}
		// The old index points to the old cache: it goes first.
		row_group.index = std::move(index);
		row_group.cluster_cache = std::move(cluster_cache);
	}

	// Loads all of the row group's index from its home, so that it can be changed. The index in memory is its home now.
	void MaterializeRowGroup(PDXRowGroup &row_group, VSSBlockChainReader &reader, BufferManager &buffer_manager) {
		// The resident index points to its cache, so it is destroyed first.
		auto resident_cache = std::move(row_group.cluster_cache);
		auto resident_index = std::move(row_group.index);
		try {
			if (!row_group.temporary_chain.blocks.empty()) {
				VSSTemporaryChainReader temporary_reader(buffer_manager, row_group.temporary_chain);
				VSSChainInputStream in(temporary_reader);
				row_group.index = PDX::LoadPDXIndexFromStream(in, *pruner, nullptr, &memory_allocator);
			} else {
				reader.Open(row_group.persisted_chain);
				VSSChainInputStream in(reader);
				row_group.index = PDX::LoadPDXIndexFromStream(in, *pruner, nullptr, &memory_allocator);
				reader.Finish();
			}
			// A row group loaded with only its resident data took deletes as tombstones since its home was written.
			// These were never reflected on the index home. Therefore, we must replay them.
			if (resident_index) {
				for (row_t row_id = row_group.row_start; row_id < row_group.row_end; row_id++) {
					const auto id = static_cast<size_t>(row_id);
					if (row_group.index->Contains(id) && !resident_index->Contains(id)) {
						row_group.index->Delete(id);
					}
				}
			}
			if (row_group.is_closed) {
				row_group.index->ShrinkToFit();
			}
		} catch (...) {
			// A failed load leaves the row group as it was: paged from its home.
			reader.Close();
			row_group.index = std::move(resident_index);
			row_group.cluster_cache = std::move(resident_cache);
			throw;
		}
		row_group.temporary_chain = VSSTemporaryChain();
	}
};

using VSSWrapperF32 = VSSWrapperParallel<PDX::F32>;
using VSSWrapperU8 = VSSWrapperParallel<PDX::U8>;
using VSSWrapperSAQ = VSSWrapperParallel<PDX::SAQ>;

} // namespace duckdb

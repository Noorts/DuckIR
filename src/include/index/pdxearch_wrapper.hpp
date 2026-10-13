#pragma once

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
#include <ostream>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/indexes/flat.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/ivf_searcher.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "index/pdxearch_index_utils.hpp"
#include "index/pdxearch_storage.hpp"

namespace duckdb {

class PDXearchWrapper {
public:
	static constexpr PDX::DistanceMetric DEFAULT_DISTANCE_METRIC = PDX::DistanceMetric::L2SQ;
	static constexpr PDX::Quantization DEFAULT_QUANTIZATION = PDX::Quantization::U8;
	static constexpr int32_t DEFAULT_N_PROBE = 24; // 5% of the data in a full search

private:
	const uint32_t num_dimensions;
	// Whether the embeddings (both the query embeddings and those stored in the
	// index) are normalized. The PDXearch kernel only implements Euclidean
	// distance (L2SQ). To compute the cosine and inner product distances we
	// normalize and then use L2SQ.
	const bool is_normalized;

	// The fields below are index options that can be set during index creation:
	// `CREATE INDEX ON t USING PDXearch(vec) WITH (n_probe = 10, seed = 42)`.
	// See `create/pdxearch_index_create_plan.cpp` for the validation logic, and
	// `pdxearch_index.cpp` for the usage.
	const PDX::DistanceMetric distance_metric;
	const PDX::Quantization quantization;
	// Between 16 and 128 is common. Setting this to 0 will probe all clusters.
	// Can be set at index build time using the 'n_probe' index option, else
	// uses `DEFAULT_N_PROBE`. At query time the 'pdxearch_n_probe' runtime
	// setting will take precedence over the n_probe saved here.
	const uint32_t n_probe;
	// Seed that currently affects the random rotation matrix generation.
	const int32_t seed;

protected:
	const unique_ptr<float[]> rotation_matrix;

public:
	PDXearchWrapper(PDX::Quantization quantization, PDX::DistanceMetric distance_metric, uint32_t num_dimensions,
	                uint32_t n_probe, int32_t seed, unique_ptr<float[]> rotation_matrix_p = nullptr)
	    : num_dimensions(num_dimensions), is_normalized(PDX::DistanceMetricRequiresNormalization(distance_metric)),
	      distance_metric(distance_metric), quantization(quantization), n_probe(n_probe), seed(seed),
	      rotation_matrix(rotation_matrix_p ? std::move(rotation_matrix_p)
	                                        : GenerateRandomRotationMatrix(num_dimensions, seed)) {
	}
	virtual ~PDXearchWrapper() = default;

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
	float *GetRotationMatrix() const {
		return rotation_matrix.get();
	}

	// An approximate lower bound of the index's size in memory.
	virtual uint64_t GetInMemorySizeInBytes() const = 0;
};

struct PDXearchRowGroupBounds {
	row_t row_start;
	idx_t count;
};

struct PDXRowGroup {
	row_t row_start;
	row_t row_end;
	std::unique_ptr<PDX::ADSamplingPruner> pruner;
	std::unique_ptr<PDX::IPDXIndex> index;
	// Held while the index is built, so that rows of this row group arriving in a later batch wait for it.
	std::mutex mutex;
	// Where the index was last persisted. A checkpoint rewrites only the row groups that changed since.
	PDXearchBlockChain persisted_chain;
	bool is_dirty = true;

	uint64_t GetInMemorySizeInBytes() const {
		return sizeof(*this) + (index ? index->GetInMemorySizeInBytes() : 0);
	}
};

// The PDXearchWrapper for the parallel implementation. The parallel implementation uses a separate index for each row
// group. This allows the creation of the index and searching in it to be parallelized at the row group level.
template <PDX::Quantization Q>
class PDXearchWrapperParallel : public PDXearchWrapper {
private:
	const idx_t row_group_size;
	// Sorted by row_start. Structural changes only happen during the index build and under the index's exclusive lock,
	// so scans read it without locking.
	std::vector<unique_ptr<PDXRowGroup>> row_groups;
	// Lock for the above vector structure.
	std::mutex row_groups_mutex;
	// The persisted chains of removed row groups, freed at the next checkpoint.
	std::vector<PDXearchBlockChain> orphaned_chains;

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

	PDXearchWrapperParallel(PDX::DistanceMetric distance_metric, uint32_t num_dimensions, uint32_t n_probe,
	                        int32_t seed, idx_t row_group_size, unique_ptr<float[]> rotation_matrix = nullptr)
	    : PDXearchWrapper(Q, distance_metric, num_dimensions, n_probe, seed, std::move(rotation_matrix)),
	      row_group_size(row_group_size) {
	}

	idx_t GetRowGroupSize() const {
		return row_group_size;
	}

	// Builds the index of the row group [row_start, row_start + count) from its (non-NULL) rows. Rows of this row
	// group that arrive in a later batch are appended to the index that was already built.
	void SetUpIndexForRowGroup(const row_t *const row_ids, const float *const embeddings, const idx_t num_embeddings,
	                           const row_t row_start, const idx_t count) {
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
				new_row_group->pruner = make_uniq<PDX::ADSamplingPruner>(GetNumDimensions(), rotation_matrix.get());
				row_group = new_row_group.get();
				build_lock = std::unique_lock<std::mutex>(row_group->mutex);
				row_groups.insert(it, std::move(new_row_group));
			}
		}

		if (build_lock.owns_lock()) {
			row_group->index = BuildRowGroupIndex(*row_group, row_ids, embeddings, num_embeddings);
			return;
		}
		// Rare path
		// Only reached when DuckDB splits a row group over several scan tasks (e.g. PRAGMA verify_parallelism hands
		// out one vector per task): the first batch built the index above, the later ones are appended to it.
		const std::lock_guard<std::mutex> lock(row_group->mutex);
		for (idx_t i = 0; i < num_embeddings; i++) {
			row_group->index->Append(static_cast<size_t>(row_ids[i]), embeddings + i * GetNumDimensions());
		}
		row_group->is_dirty = true;
	}

	unique_ptr<PDX::IIterativeSearch>
	BeginSearchForRowGroup(const idx_t row_group_idx, const float *const preprocessed_query_embedding,
	                       const idx_t limit, PDX::TopKHeap &top_k_heap,
	                       const std::vector<size_t> *const passing_row_ids,
	                       const std::vector<uint32_t> *const clusters_access_order = nullptr) {
		auto search_cursor = row_groups[row_group_idx]->index->BeginIterativeSearch(
		    preprocessed_query_embedding, static_cast<uint32_t>(limit), top_k_heap, passing_row_ids,
		    /*is_query_transformed=*/true, clusters_access_order);
		return unique_ptr<PDX::IIterativeSearch>(search_cursor.release());
	}

	// The row group's clusters, nearest to the query first (searches of one query share it).
	std::vector<uint32_t> GetClustersAccessOrderForRowGroup(const idx_t row_group_idx,
	                                                        const float *const preprocessed_query_embedding) {
		return row_groups[row_group_idx]->index->GetClustersAccessOrder(preprocessed_query_embedding,
		                                                                /*is_query_transformed=*/true);
	}

	// Maintenance, one writer at a time (the index's exclusive lock). The row belongs to the DuckDB row group that
	// `row_groups[row_group_idx]` mirrors; the row group's end grows with it.
	// If the rowgroup is a Flat index and it has enough embeddings, it is promoted to an IVF index.
	void AppendRow(const idx_t row_group_idx, const row_t row_id, const float *const transformed_embedding) {
		auto &row_group = *row_groups[row_group_idx];
		D_ASSERT(row_id >= row_group.row_start);
		// Rare: a rebuild of this row group (e.g., a checkpoint merge) ran earlier in the same
		// sync and fetched all its committed rows, including (some) staged ones. The staged range now
		// brings them here a second time.
		if (row_group.index->Contains(static_cast<size_t>(row_id))) {
			return;
		}
		row_group.index->Append(static_cast<size_t>(row_id), transformed_embedding);
		row_group.row_end = MaxValue<row_t>(row_group.row_end, row_id + 1);
		row_group.is_dirty = true;
		auto flat_index = dynamic_cast<PDX::FlatIndex *>(row_group.index.get());
		if (flat_index && flat_index->GetClusterSize(0) >= MIN_EMBEDDINGS_FOR_CLUSTERING) {
			PromoteToIVF(row_group, *flat_index);
		}
	}

	void DeleteRow(const row_t row_id) {
		const auto row_group_idx = LookupRowGroup(row_id);
		if (!row_group_idx.IsValid()) {
			return;
		}
		auto &row_group = *row_groups[row_group_idx.GetIndex()];
		// Rows the index does not hold (NULL embeddings, deletes replayed from the WAL) leave the row group clean.
		if (row_group.index->Delete(static_cast<size_t>(row_id))) {
			row_group.is_dirty = true;
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

	std::vector<PDXearchRowGroupBounds>
	FindStaleRowGroups(const std::vector<PDXearchRowGroupBounds> &physical_row_groups) const {
		std::vector<PDXearchRowGroupBounds> stale;
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
	// each row group, so the allocator never holds more than one rewritten row group in memory.
	void PersistDirtyRowGroups(FixedSizeAllocator &allocator, const std::function<void()> &write_partial_blocks) {
		for (auto &chain : orphaned_chains) {
			chain.Free(allocator);
		}
		orphaned_chains.clear();
		for (auto &row_group : row_groups) {
			if (!row_group->is_dirty) {
				continue;
			}
			row_group->persisted_chain.Free(allocator);
			PDXearchBlockChainWriter writer(allocator);
			std::ostream out(&writer);
			row_group->index->SaveToStream(out);
			row_group->persisted_chain = writer.Finish();
			row_group->is_dirty = false;
			write_partial_blocks();
		}
	}

	// The directory only stores where each chain starts and how long it is.
	void AddRowGroupEntries(PDXearchDirectory &directory) const {
		for (const auto &row_group : row_groups) {
			const auto &chain = row_group->persisted_chain;
			directory.row_groups.push_back(
			    {row_group->row_start, row_group->row_end, PDXearchBlockChain {chain.head, chain.num_bytes, {}}});
		}
	}

	// The allocator was reset: nothing is persisted anymore.
	void ResetPersistedChains() {
		orphaned_chains.clear();
		for (auto &row_group : row_groups) {
			row_group->persisted_chain = PDXearchBlockChain();
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

	// Loads the persisted row groups, in the directory's (row_start) order.
	void LoadRowGroups(PDXearchBlockChainReader &reader, const vector<PDXearchDirectory::RowGroupEntry> &entries) {
		for (const auto &entry : entries) {
			auto row_group = make_uniq<PDXRowGroup>();
			row_group->row_start = entry.row_start;
			row_group->row_end = entry.row_end;
			row_group->pruner = make_uniq<PDX::ADSamplingPruner>(GetNumDimensions(), rotation_matrix.get());
			reader.Open(entry.chain);
			std::istream in(&reader);
			row_group->index = PDX::LoadPDXIndexFromStream(in, *row_group->pruner);
			row_group->persisted_chain = reader.Finish();
			row_group->is_dirty = false;
			row_groups.push_back(std::move(row_group));
		}
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
		uint64_t in_memory_size_in_bytes = sizeof(*this);
		for (const auto &row_group : row_groups) {
			in_memory_size_in_bytes += row_group->GetInMemorySizeInBytes();
		}
		return in_memory_size_in_bytes;
	}

private:
	PDX::PDXIndexConfig MakeIndexConfig(const row_t row_start, const idx_t num_embeddings) const {
		PDX::PDXIndexConfig config;
		config.num_dimensions = GetNumDimensions();
		config.distance_metric = GetDistanceMetric();
		config.seed = static_cast<uint32_t>(GetSeed());
		config.num_clusters = static_cast<uint32_t>(ComputeNumClustersForRowGroup(num_embeddings));
		config.kmeans_iters = 8;
		config.hierarchical_indexing = true;
		config.n_threads = 1;
		config.is_data_transformed = true;
		config.base_row_id = static_cast<size_t>(row_start);
		return config;
	}

	unique_ptr<PDX::IPDXIndex> BuildRowGroupIndex(const PDXRowGroup &row_group, const row_t *const row_ids,
	                                              const float *const embeddings, const idx_t num_embeddings) const {
		const auto config = MakeIndexConfig(row_group.row_start, num_embeddings);
		std::vector<size_t> ids(num_embeddings);
		for (idx_t i = 0; i < num_embeddings; i++) {
			ids[i] = static_cast<size_t>(row_ids[i]);
		}

		if (num_embeddings < MIN_EMBEDDINGS_FOR_CLUSTERING) {
			auto flat_index = make_uniq<PDX::FlatIndex>(config, *row_group.pruner);
			flat_index->BuildIndex(ids.data(), embeddings, num_embeddings);
			return std::move(flat_index);
		}
		auto ivf_index = make_uniq<PDX::PDXIndex<Q>>(config, *row_group.pruner);
		ivf_index->BuildIndex(ids.data(), embeddings, num_embeddings);
		return std::move(ivf_index);
	}

	void PromoteToIVF(PDXRowGroup &row_group, const PDX::FlatIndex &flat_index) {
		const auto row_ids = flat_index.GetRowIds();
		const auto embeddings = flat_index.GetEmbeddings();
		auto ivf_index =
		    make_uniq<PDX::PDXIndex<Q>>(MakeIndexConfig(row_group.row_start, row_ids.size()), *row_group.pruner);
		ivf_index->BuildIndex(row_ids.data(), embeddings.get(), row_ids.size());
		row_group.index = std::move(ivf_index);
	}
};

using PDXearchWrapperF32 = PDXearchWrapperParallel<PDX::F32>;
using PDXearchWrapperU8 = PDXearchWrapperParallel<PDX::U8>;

} // namespace duckdb

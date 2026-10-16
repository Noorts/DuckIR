#pragma once

#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include <atomic>

#include "pdx/common.hpp"
#include "index/vss_storage.hpp"
#include "index/vss_wrapper.hpp"

namespace duckdb {

struct VSSRowRange {
	row_t start; // [start, end)
	row_t end;
};

struct VSSIndexStats {
	string metric;
	string quantization;
	int64_t num_dimensions;
	int64_t n_probe;
	int64_t seed;
	bool is_normalized;
	int64_t approximate_lower_bound_memory_usage_bytes;
	int64_t cluster_acquires;
	int64_t cluster_cache_misses;
	int64_t cluster_bytes_fetched;
	int64_t blocks_read;
};

class VSSIndex : public BoundIndex {
public:
	static constexpr const char *TYPE_NAME = "DUCKIR_VSS";

	static const case_insensitive_map_t<PDX::DistanceMetric> DISTANCE_METRIC_MAP;
	static const case_insensitive_map_t<PDX::Quantization> QUANTIZATION_MAP;

private:
	// Null only when the index was not loaded from storage and paging is off.
	unique_ptr<VSSBlockChainReader> storage_reader;
	bool cluster_paging = true;
	bool record_paging_counters = false;
	bool cache_tiers = false;
	unique_ptr<VSSWrapper> vss_wrapper;
	unique_ptr<EmbeddingPreprocessor> embedding_preprocessor;

	// Row ids committed to the table whose embeddings are not in the index yet
	std::vector<VSSRowRange> unindexed_row_ranges;
	std::atomic<bool> has_unindexed_rows {false};
	// Set when the index is loaded from storage: its first sync reconciles it with the table (ReconcileWithTable).
	bool needs_reconciliation = false;
	// Why a persisted index failed to load when DuckDB bound it, raised on first use instead: DuckDB's
	// TableIndexList::Bind is not exception-safe, and a bind that throws leaves the next one spinning forever.
	ErrorData bind_error;
	IndexStorageInfo persisted_storage_info;

	BufferManager &buffer_manager;
	std::atomic<idx_t> reserved_memory_bytes {0};
	// Reserves `delta` more bytes, or frees `-delta`. Throws DuckDB's out-of-memory error when nothing more can be
	// evicted.
	void UpdateReservedMemory(int64_t delta);

	void AppendRows(idx_t row_group_idx, const row_t *row_ids, float *transformed_embeddings, idx_t count,
	                PDX::ParallelExecutor *executor);
	void DeleteRow(row_t row_id);

	// Returns the committed rows of [start, end) whose embedding is not NULL,
	// We transform them (random rotation) and return how many were written to `embeddings`.
	// `rows_returned` counts every committed row the table returned, NULL embeddings included.
	idx_t FetchRows(DataTable &table, row_t start, row_t end, row_t *row_ids, float *embeddings, idx_t &rows_returned,
	                PDX::ParallelExecutor *executor);

	// The committed rows of [start, end), ascending, written to `row_ids`. Scans only the row id column, so it reads no
	// column data.
	idx_t FetchCommittedRowIds(DataTable &table, row_t start, row_t end, row_t *row_ids);

	// The first sync after the index was loaded from storage. The index can lack rows the table has (a crash before the
	// first checkpoint, inserts in WAL replays DuckDB dropped, #26112) and hold rows the table deleted (deletes in
	// those replays).
	void ReconcileWithTable(DataTable &table);
	void ReconcileRowGroup(row_t start, row_t end, const row_t *row_ids, idx_t count,
	                       const std::function<void(row_t, row_t)> &stage);

	// Find row groups whose mirrors are stale:
	// - DuckDB merged two or more row groups into one
	// - DuckDB dropped a rowgroup (e.g., VACUUM, CHECKPOINT merging)
	vector<VSSRowGroupBounds> FindStaleRowGroups(DataTable &table) const;

	void RemoveRowGroupsOverlapping(row_t start, row_t end);

	// Adds the rows of one commit to unindexed_row_ranges, which needs to stay sorted and disjoint.
	// A range that overlaps staged ones (DuckDB replaying a commit from the WAL) is merged with them.
	void StageUnindexedRange(VSSRowRange range);

	void RemoveUnindexedRow(row_t row_id);

	// A table is in sync if:
	// - No unindexed rows
	// - Every mirrored row group matches its DuckDB row group.
	bool IsInSyncWithTable(DataTable &table) const;

	// The index's storage: the directory chain points to the rotation chain and to each row group's chain.
	unique_ptr<FixedSizeAllocator> allocator;
	VSSBlockChain directory_chain;
	VSSBlockChain rotation_chain;

	// Writes the allocator's dirty buffers to disk, which releases their memory.
	void WritePartialBlocks(QueryContext context);
	// Frees the current directory chain and writes `directory` into a new one.
	void WriteDirectory(const VSSDirectory &directory);
	IndexStorageInfo MakeStorageInfo() const;
	void PersistDirtyRowGroups(const std::function<void()> &write_partial_blocks);
	void WriteTemporaryChains();
	void AddRowGroupEntries(VSSDirectory &directory) const;
	void ResetPersistedChains();
	// Reads the directory and the rotation of a persisted index. The returned reader
	// then reads the row groups' chains.
	unique_ptr<VSSBlockChainReader> OpenStorage(const IndexStorageInfo &info, uint32_t num_dimensions,
	                                            VSSDirectory &directory, unique_ptr<float[]> &rotation_matrix);
	void LoadRowGroups(VSSBlockChainReader &reader, const VSSDirectory &directory, bool page_clusters);
	static IndexStorageInfo CopyStorageInfo(const IndexStorageInfo &info);

	unique_ptr<ExpressionMatcher> function_matcher;

	// A readers-writers (shared-exclusive; exclusive prioritized) lock that
	// lets any number of index scans run concurrently (shared) while a
	// maintenance operation excludes them all (exclusive). Scans only read
	// index state; their per-query state lives in the PDX::IIterativeSearch
	// cursors they own. DuckDB's IndexLock is also acquired implicitly in
	// BoundIndex, but this is insufficient to guard 1+ index scans from the
	// maintenance operations. Hence, we add this rwlock.
	//
	// See the pull request this was introduced in for more context.
	mutable StorageLock rwlock;

public:
	VSSIndex(const Identifier &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	         TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	         AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	         const IndexStorageInfo &info = IndexStorageInfo());
	~VSSIndex() override;

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	/******************************************************************
	 * Index creation and search methods specific to the parallel implementation
	 ******************************************************************/

	// Syncs the index with the table if it is out of date, then returns the shared lock a search holds while it runs.
	unique_ptr<StorageLockKey> SyncAndLockForSearch(ClientContext &context, DataTable &table);

	bool HasUnindexedRows() const {
		return has_unindexed_rows;
	}

	// Detect and rebuild rowgroups whose mirrors no longer match the table rowgroups.
	// Append unindexed rows to the index, and retire them from the unindexed list
	// Rows whose commit is still in flight stay unindexed.
	void SyncWithTable(ClientContext &context, DataTable &table);

	idx_t GetRowGroupSize() const;

	// The DuckDB row group that holds row_id. False if the row is not in the table (yet).
	bool TryGetPhysicalRowGroup(DataTable &table, row_t row_id, VSSRowGroupBounds &result) const;

	vector<VSSRowGroupBounds> GetPhysicalRowGroups(DataTable &table) const;

	// Position of the index's row group that holds row_id (the row_group_idx of the methods below).
	optional_idx LookupRowGroup(row_t row_id) const;

	// The row ids the index's row group at row_group_idx covers.
	VSSRowRange GetRowGroupRange(idx_t row_group_idx) const;

	void SetUpIndexForRowGroup(const row_t *row_ids, const float *embeddings, idx_t num_embeddings, row_t row_start,
	                           idx_t count, PDX::ParallelExecutor *executor);
	uint64_t EstimateBuildHeapBytes(idx_t num_embeddings) const;

	// !`passing_row_ids` are size_t because they go straight into PDX's IPDXIndex::BeginIterativeSearch.
	// `clusters_access_order`: the row group's GetClustersAccessOrderForRowGroup for this query (nullptr: rank).
	unique_ptr<PDX::IIterativeSearch>
	BeginSearchForRowGroup(idx_t row_group_idx, const float *preprocessed_query_embedding, idx_t limit,
	                       PDX::TopKHeap &top_k_heap, const std::vector<size_t> *passing_row_ids,
	                       const std::vector<uint32_t> *clusters_access_order = nullptr);
	std::vector<uint32_t> GetClustersAccessOrderForRowGroup(idx_t row_group_idx,
	                                                        const float *preprocessed_query_embedding);

	/******************************************************************
	 * Index maintenance
	 ******************************************************************/

	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) override;

	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;

	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) override;

	void ResetStorage(IndexLock &lock) override;

	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;

	void Vacuum(IndexLock &state) override;

	void Verify(IndexLock &state) override;

	string ToString(IndexLock &state, bool display_ascii) override;

	void VerifyAllocations(IndexLock &state) override;

	void VerifyBuffers(IndexLock &state) override;

	idx_t GetInMemorySize(IndexLock &state) const override;

	unique_ptr<VSSIndexStats> GetStats(const ClientContext &context) const;

	/******************************************************************
	 * Index persistence
	 ******************************************************************/

	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;

	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;

	/******************************************************************
	 * Misc.
	 ******************************************************************/

	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) const override {
		return "Constraint violation in DuckIR VSS index";
	}

	bool TryMatchDistanceFunction(const unique_ptr<Expression> &expr, vector<reference<Expression>> &bindings) const;

	bool TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const;

	unique_ptr<ExpressionMatcher> MakeFunctionMatcher(const VSSWrapper &vss_wrapper);

	/******************************************************************
	 * Getters
	 ******************************************************************/

	// Total cluster count summed across all (initialized) row groups. Used for EXPLAIN output.
	idx_t GetTotalNumClusters() const {
		if (vss_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<VSSWrapperU8 *>(vss_wrapper.get())->GetTotalNumClusters();
		}
		return static_cast<VSSWrapperF32 *>(vss_wrapper.get())->GetTotalNumClusters();
	}

	// The clusters a full row group is built with. This is not an upper bound, SPFresh can grow a row group past it.
	static constexpr idx_t GetNumClustersForFullRowGroup() {
		return VSSWrapperF32::ComputeNumClustersForRowGroup(DEFAULT_ROW_GROUP_SIZE);
	}

	idx_t GetNumRowGroups() const {
		if (vss_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<VSSWrapperU8 *>(vss_wrapper.get())->GetNumRowGroups();
		}
		return static_cast<VSSWrapperF32 *>(vss_wrapper.get())->GetNumRowGroups();
	}

	const PDX::IPDXIndex &GetRowGroupIndex(const idx_t row_group_idx) const {
		if (vss_wrapper->GetQuantization() == PDX::U8) {
			return static_cast<VSSWrapperU8 *>(vss_wrapper.get())->GetRowGroupIndex(row_group_idx);
		}
		return static_cast<VSSWrapperF32 *>(vss_wrapper.get())->GetRowGroupIndex(row_group_idx);
	}

	// The most clusters a row group has: the bound n_probe is clamped to.
	idx_t GetMaxNumClusters() const {
		idx_t max_num_clusters = 0;
		for (idx_t row_group_idx = 0; row_group_idx < GetNumRowGroups(); row_group_idx++) {
			max_num_clusters = MaxValue<idx_t>(max_num_clusters, GetRowGroupIndex(row_group_idx).GetNumClusters());
		}
		return max_num_clusters;
	}

	string GetQuantization() const;

	idx_t GetNumDimensions() const {
		return vss_wrapper->GetNumDimensions();
	}

	string GetDistanceMetric() const;

	bool IsNormalized() const {
		return vss_wrapper->IsNormalized();
	}

	// N_probe precedence: runtime setting (duckir_vss_n_probe) > index setting (n_probe) > default.
	idx_t GetEffectiveNProbe(const ClientContext &context) const {
		auto current_n_probe = static_cast<idx_t>(vss_wrapper->GetNProbe());

		Value vss_n_probe_opt;
		if (context.TryGetCurrentSetting("duckir_vss_n_probe", vss_n_probe_opt)) {
			if (!vss_n_probe_opt.IsNull() && vss_n_probe_opt.type() == LogicalType::INTEGER) {
				auto val = vss_n_probe_opt.GetValue<int32_t>();
				if (val >= 0) {
					current_n_probe = static_cast<idx_t>(val);
				}
			}
		}

		return current_n_probe;
	}

	float *GetRotationMatrix() const {
		return vss_wrapper->GetRotationMatrix();
	}

	PDX::PDXIndexConfig MakePDXIndexConfig(const idx_t num_clusters, PDX::ParallelExecutor *const executor,
	                                       const idx_t base_row_id) const {
		return vss_wrapper->MakePDXIndexConfig(num_clusters, executor, base_row_id);
	}
};

// The table columns a search fetches by rowid, for EXPLAIN: one name per line, like a SEQ_SCAN's projections.
string ColumnNamesToString(const TableCatalogEntry &table, const vector<ColumnIndex> &column_ids);

} // namespace duckdb

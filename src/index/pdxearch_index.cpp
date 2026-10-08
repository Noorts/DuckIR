#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction_data.hpp"
#include "pdx/common.hpp"
#include "pdx/utils.hpp"
#include "index/pdxearch_index.hpp"

#include "index/pdxearch_module.hpp"

#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

PDXearchIndex::PDXearchIndex(const string &name, IndexConstraintType index_constraint_type,
                             const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                             const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                             const case_insensitive_map_t<Value> &index_creation_options,
                             const IndexStorageInfo &persistence_info)
    : BoundIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db),
      buffer_manager(BufferManager::GetBufferManager(db)) {
	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("PDXearch indexes do not support unique or primary key constraints");
	}

	// We only support one ARRAY column
	D_ASSERT(logical_types.size() == 1);
	const auto &embedding_type = logical_types[0];
	D_ASSERT(embedding_type.id() == LogicalTypeId::ARRAY);

	const auto num_dimensions = ArrayType::GetSize(embedding_type);

	// DuckDB v1.5.6 does not persist the WITH options in its catalog, so a
	// persisted index reads them from its storage info (MakeStorageInfo).
	const auto &options = persistence_info.IsValid() ? persistence_info.options : index_creation_options;

	// Try to get the vector metric from the options, this parameter should be verified during binding.
	auto dist_metric = PDXearchWrapper::DEFAULT_DISTANCE_METRIC;
	const auto dist_metric_opt = options.find("metric");
	if (dist_metric_opt != options.end()) {
		const auto dist_metric_val =
		    PDXearchIndex::DISTANCE_METRIC_MAP.find(dist_metric_opt->second.GetValue<string>());
		if (dist_metric_val != PDXearchIndex::DISTANCE_METRIC_MAP.end()) {
			dist_metric = dist_metric_val->second;
		}
	}

	auto quantization = PDXearchWrapper::DEFAULT_QUANTIZATION;
	const auto quantization_opt = options.find("quantization");
	if (quantization_opt != options.end()) {
		const auto quantization_val = PDXearchIndex::QUANTIZATION_MAP.find(quantization_opt->second.GetValue<string>());
		if (quantization_val != PDXearchIndex::QUANTIZATION_MAP.end()) {
			quantization = quantization_val->second;
		}
	}

	auto n_probe = PDXearchWrapper::DEFAULT_N_PROBE;
	const auto n_probe_opt = options.find("n_probe");
	if (n_probe_opt != options.end()) {
		n_probe = n_probe_opt->second.GetValue<int32_t>();
	}

	// TODO: Confirm the static cast is sound.
	auto seed = static_cast<int32_t>(std::random_device {}());
	const auto seed_opt = options.find("seed");
	if (seed_opt != options.end()) {
		seed = seed_opt->second.GetValue<int32_t>();
	}

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	allocator = make_uniq<FixedSizeAllocator>(PDXearchBlockChain::GetSegmentSize(block_manager), block_manager);
	// A persisted index keeps its rotation. Its row groups are loaded once the wrapper exists.
	PDXearchDirectory directory;
	unique_ptr<float[]> rotation_matrix;
	if (persistence_info.IsValid()) {
		storage_reader =
		    OpenStorage(persistence_info, static_cast<uint32_t>(num_dimensions), directory, rotation_matrix);
	}

	const idx_t row_group_size = table_io_manager.GetRowGroupSize();
	if (quantization == PDX::Quantization::F32) {
		D_ASSERT(ArrayType::GetChildType(embedding_type).id() == LogicalTypeId::FLOAT);

		pdxearch_wrapper = make_uniq<PDXearchWrapperF32>(dist_metric, num_dimensions, n_probe, seed, row_group_size,
		                                                 std::move(rotation_matrix));
	} else if (quantization == PDX::Quantization::U8) {
		pdxearch_wrapper = make_uniq<PDXearchWrapperU8>(dist_metric, num_dimensions, n_probe, seed, row_group_size,
		                                                std::move(rotation_matrix));
	} else {
		throw InternalException("Unsupported quantization: %s", quantization);
	}
	// The rotation matrix and the pruner, before any row group.
	UpdateReservedMemory(static_cast<int64_t>(pdxearch_wrapper->GetInMemorySizeInBytes()));
	Value cluster_paging_setting;
	db.GetDatabase().TryGetCurrentSetting("pdxearch_cluster_paging", cluster_paging_setting);
	cluster_paging = cluster_paging_setting.IsNull() || cluster_paging_setting.GetValue<bool>();
	Value paging_counters_setting;
	db.GetDatabase().TryGetCurrentSetting("pdxearch_paging_counters", paging_counters_setting);
	record_paging_counters = !paging_counters_setting.IsNull() && paging_counters_setting.GetValue<bool>();
	Value cache_tiers_setting;
	db.GetDatabase().TryGetCurrentSetting("pdxearch_cache_tiers", cache_tiers_setting);
	cache_tiers = cache_tiers_setting.IsNull() || cache_tiers_setting.GetValue<bool>();
	// An index created in this session pages its row groups from their temporary chains.
	if (cluster_paging && !storage_reader) {
		storage_reader =
		    make_uniq<PDXearchBlockChainReader>(table_io_manager.GetIndexBlockManager(), allocator->GetInfo());
	}
	if (storage_reader) {
		storage_reader->counters.enabled = record_paging_counters;
		storage_reader->cache_tiers = cache_tiers;
	}
	if (persistence_info.IsValid()) {
		// The destructor does not run when the constructor throws (e.g., out of memory while loading).
		try {
			LoadRowGroups(*storage_reader, directory, cluster_paging);
		} catch (...) {
			UpdateReservedMemory(-static_cast<int64_t>(reserved_memory_bytes.load()));
			throw;
		}
		needs_reconciliation = true;
	}

	function_matcher = MakeFunctionMatcher(*pdxearch_wrapper.get());
	embedding_preprocessor = make_uniq<EmbeddingPreprocessor>(num_dimensions, pdxearch_wrapper->GetRotationMatrix());
}

// Indexes are destroyed before DuckDB's buffer manager (DatabaseInstance::~DatabaseInstance).
PDXearchIndex::~PDXearchIndex() {
	UpdateReservedMemory(-static_cast<int64_t>(reserved_memory_bytes.load()));
}

void PDXearchIndex::UpdateReservedMemory(const int64_t delta) {
	if (delta > 0) {
		buffer_manager.ReserveMemory(static_cast<idx_t>(delta));
		reserved_memory_bytes += static_cast<idx_t>(delta);
	} else if (delta < 0) {
		buffer_manager.FreeReservedMemory(static_cast<idx_t>(-delta));
		reserved_memory_bytes -= static_cast<idx_t>(-delta);
	}
}

/******************************************************************
 * Index creation and search methods specific to the parallel implementation
 ******************************************************************/

bool PDXearchIndex::IsInSyncWithTable(DataTable &table) const {
	return !needs_reconciliation && !has_unindexed_rows && FindStaleRowGroups(table).empty();
}

unique_ptr<StorageLockKey> PDXearchIndex::SyncAndLockForSearch(DataTable &table) {
	bool in_sync;
	{
		auto _lock = rwlock.GetSharedLock();
		in_sync = IsInSyncWithTable(table);
	}
	if (!in_sync) {
		SyncWithTable(table);
	}
	return rwlock.GetSharedLock();
}

idx_t PDXearchIndex::GetRowGroupSize() const {
	return table_io_manager.GetRowGroupSize();
}

bool PDXearchIndex::TryGetPhysicalRowGroup(DataTable &table, const row_t row_id, PDXearchRowGroupBounds &result) const {
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	idx_t segment_index;
	if (!row_groups->TryGetSegmentIndex(lock, static_cast<idx_t>(row_id), segment_index)) {
		return false;
	}
	auto node = row_groups->GetSegmentByIndex(lock, static_cast<int64_t>(segment_index));
	result = {static_cast<row_t>(node->GetRowStart()), node->GetCount()};
	return true;
}

vector<PDXearchRowGroupBounds> PDXearchIndex::GetPhysicalRowGroups(DataTable &table) const {
	vector<PDXearchRowGroupBounds> result;
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	auto lock = row_groups->Lock();
	for (auto node = row_groups->GetRootSegment(lock); node; node = row_groups->GetNextSegment(lock, *node)) {
		result.push_back({static_cast<row_t>(node->GetRowStart()), node->GetCount()});
	}
	return result;
}

optional_idx PDXearchIndex::LookupRowGroup(const row_t row_id) const {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->LookupRowGroup(row_id);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->LookupRowGroup(row_id);
}

PDXearchRowRange PDXearchIndex::GetRowGroupRange(const idx_t row_group_idx) const {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		const auto &row_group = static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->GetRowGroup(row_group_idx);
		return {row_group.row_start, row_group.row_end};
	}
	const auto &row_group = static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->GetRowGroup(row_group_idx);
	return {row_group.row_start, row_group.row_end};
}

void PDXearchIndex::AppendRow(const idx_t row_group_idx, const row_t row_id, const float *const transformed_embedding) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->AppendRow(row_group_idx, row_id, transformed_embedding, storage_reader.get(), buffer_manager);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->AppendRow(row_group_idx, row_id, transformed_embedding, storage_reader.get(), buffer_manager);
	}
}

void PDXearchIndex::DeleteRow(const row_t row_id) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->DeleteRow(row_id);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->DeleteRow(row_id);
	}
}

vector<PDXearchRowGroupBounds> PDXearchIndex::FindStaleRowGroups(DataTable &table) const {
	const auto physical_row_groups = GetPhysicalRowGroups(table);
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->FindStaleRowGroups(physical_row_groups);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->FindStaleRowGroups(physical_row_groups);
}

void PDXearchIndex::RemoveRowGroupsOverlapping(const row_t start, const row_t end) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->RemoveRowGroupsOverlapping(start, end);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->RemoveRowGroupsOverlapping(start, end);
	}
}

idx_t PDXearchIndex::FetchRows(DataTable &table, const row_t start, const row_t end, row_t *const row_ids,
                               float *const embeddings, idx_t &rows_returned) {
	rows_returned = 0;
	// Fetches every committed row, whatever the calling transaction's snapshot. The mirror
	// holds the committed state, and rows a query must not see are dropped when it fetches its results.
	const TransactionData committed(MAX_TRANSACTION_ID, DuckTransactionManager::Get(db).GetLastCommit() + 1);
	const vector<StorageIndex> fetch_column_ids {StorageIndex(GetColumnIds()[0]), StorageIndex()};
	DataChunk fetched;
	fetched.Initialize(Allocator::Get(db), {logical_types[0], LogicalType::ROW_TYPE});
	Vector fetch_row_ids(LogicalType::ROW_TYPE, STANDARD_VECTOR_SIZE);
	const auto fetch_row_ids_data = FlatVector::GetData<row_t>(fetch_row_ids);
	ColumnFetchState fetch_state;
	const auto num_dimensions = GetNumDimensions();
	auto raw_embeddings = make_uniq_array<float>(STANDARD_VECTOR_SIZE * num_dimensions);

	idx_t count = 0;
	for (row_t batch_start = start; batch_start < end; batch_start += STANDARD_VECTOR_SIZE) {
		const idx_t batch_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, static_cast<idx_t>(end - batch_start));
		for (idx_t i = 0; i < batch_size; i++) {
			fetch_row_ids_data[i] = batch_start + static_cast<row_t>(i);
		}
		fetched.Reset();
		// Deleted rows are gone, NULL embeddings are NULL.
		table.GetRowGroupCollection()->Fetch(committed, fetched, fetch_column_ids, fetch_row_ids, batch_size,
		                                     fetch_state);
		const idx_t fetched_count = fetched.size();
		rows_returned += fetched_count;
		if (fetched_count == 0) {
			continue;
		}
		auto &embedding_column = fetched.data[0];
		embedding_column.Flatten(fetched_count);
		fetched.data[1].Flatten(fetched_count);
		const auto &validity = FlatVector::Validity(embedding_column);
		const auto fetched_embeddings = FlatVector::GetData<float>(ArrayVector::GetEntry(embedding_column));
		const auto fetched_row_ids = FlatVector::GetData<row_t>(fetched.data[1]);
		idx_t batch_count = 0;
		for (idx_t i = 0; i < fetched_count; i++) {
			if (!validity.RowIsValid(i)) {
				continue;
			}
			memcpy(raw_embeddings.get() + batch_count * num_dimensions, fetched_embeddings + i * num_dimensions,
			       num_dimensions * sizeof(float));
			row_ids[count + batch_count] = fetched_row_ids[i];
			batch_count++;
		}
		if (batch_count > 0) {
			embedding_preprocessor->PreprocessEmbeddings(raw_embeddings.get(), embeddings + count * num_dimensions,
			                                             batch_count, IsNormalized());
			count += batch_count;
		}
	}
	return count;
}

idx_t PDXearchIndex::FetchCommittedRowIds(DataTable &table, const row_t start, const row_t end, row_t *const row_ids) {
	const vector<StorageIndex> column_ids {StorageIndex()};
	TableScanState scan_state;
	scan_state.Initialize(column_ids);
	table.GetRowGroupCollection()->InitializeScanWithOffset(QueryContext(), scan_state.table_state, column_ids,
	                                                        static_cast<idx_t>(start), static_cast<idx_t>(end));
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(db), {LogicalType::ROW_TYPE});
	idx_t count = 0;
	while (true) {
		chunk.Reset();
		scan_state.table_state.Scan(chunk, TableScanType::TABLE_SCAN_COMMITTED_ROWS);
		if (chunk.size() == 0) {
			break;
		}
		chunk.data[0].Flatten(chunk.size());
		memcpy(row_ids + count, FlatVector::GetData<row_t>(chunk.data[0]), chunk.size() * sizeof(row_t));
		count += chunk.size();
	}
	return count;
}

// One DuckDB row group of row ids at a time
void PDXearchIndex::ReconcileWithTable(DataTable &table) {
	auto row_ids = make_uniq_array<row_t>(GetRowGroupSize());
	const auto stage = [&](const row_t start, const row_t end) {
		StageUnindexedRange({start, end});
	};
	for (const auto &physical : GetPhysicalRowGroups(table)) {
		if (physical.count == 0) {
			continue;
		}
		const row_t end = physical.row_start + static_cast<row_t>(physical.count);
		const idx_t count = FetchCommittedRowIds(table, physical.row_start, end, row_ids.get());
		ReconcileRowGroup(physical.row_start, end, row_ids.get(), count, stage);
	}
}

void PDXearchIndex::ReconcileRowGroup(const row_t start, const row_t end, const row_t *const row_ids, const idx_t count,
                                      const std::function<void(row_t, row_t)> &stage) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->ReconcileRowGroup(start, end, row_ids, count, stage);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->ReconcileRowGroup(start, end, row_ids, count, stage);
	}
}

void PDXearchIndex::SyncWithTable(DataTable &table) {
	auto _lock = rwlock.GetExclusiveLock();

	// One DuckDB row group of transformed embeddings at a time, like the create sink, in a DuckDB buffer of the rows
	// fetched.
	const auto num_dimensions = GetNumDimensions();
	const auto allocate_embeddings = [&](const idx_t num_rows) {
		return buffer_manager.Allocate(MemoryTag::EXTENSION, num_rows * num_dimensions * sizeof(float),
		                               /*can_destroy=*/false);
	};
	auto row_ids = make_uniq_array<row_t>(GetRowGroupSize());

	// Row groups a checkpoint merged or dropped: their mirrors go, and the ones still in the table are rebuilt.
	idx_t rows_returned;
	for (const auto &stale : FindStaleRowGroups(table)) {
		RemoveRowGroupsOverlapping(stale.row_start, stale.row_start + static_cast<row_t>(stale.count));
		PDXearchRowGroupBounds bounds;
		if (!TryGetPhysicalRowGroup(table, stale.row_start, bounds)) {
			continue;
		}
		auto embeddings_buffer = allocate_embeddings(bounds.count);
		const auto embeddings = reinterpret_cast<float *>(embeddings_buffer.Ptr());
		const idx_t count = FetchRows(table, bounds.row_start, bounds.row_start + static_cast<row_t>(bounds.count),
		                              row_ids.get(), embeddings, rows_returned);
		if (count > 0) {
			SetUpIndexForRowGroup(row_ids.get(), embeddings, count, bounds.row_start, bounds.count, /*n_threads=*/1);
		}
	}

	// Only happens on the first query after the first index bind
	if (needs_reconciliation) {
		ReconcileWithTable(table);
		needs_reconciliation = false;
	}

	// Unindexed rows, gathered per DuckDB row group: a row group without a mirror is clustered once from all of its
	// rows. Each range is retired on its own because rows of one commit become visible together.
	std::vector<PDXearchRowRange> still_unindexed;
	idx_t range_idx = 0;
	while (range_idx < unindexed_row_ranges.size()) {
		PDXearchRowGroupBounds bounds;
		if (!TryGetPhysicalRowGroup(table, unindexed_row_ranges[range_idx].start, bounds)) {
			// The rows are not in the table yet: their commit is still in flight.
			still_unindexed.push_back(unindexed_row_ranges[range_idx++]);
			continue;
		}
		const row_t bounds_end = bounds.row_start + static_cast<row_t>(bounds.count);
		idx_t num_unindexed_rows = 0;
		for (idx_t i = range_idx; i < unindexed_row_ranges.size() && unindexed_row_ranges[i].start < bounds_end; i++) {
			const auto &range = unindexed_row_ranges[i];
			num_unindexed_rows += static_cast<idx_t>(MinValue<row_t>(range.end, bounds_end) - range.start);
		}
		auto embeddings_buffer = allocate_embeddings(num_unindexed_rows);
		const auto embeddings = reinterpret_cast<float *>(embeddings_buffer.Ptr());
		idx_t count = 0;
		while (range_idx < unindexed_row_ranges.size() && unindexed_row_ranges[range_idx].start < bounds_end) {
			auto &range = unindexed_row_ranges[range_idx];
			const row_t sub_end = MinValue<row_t>(range.end, bounds_end);
			const idx_t range_count = FetchRows(table, range.start, sub_end, row_ids.get() + count,
			                                    embeddings + count * num_dimensions, rows_returned);
			if (rows_returned == 0) {
				// Not committed yet: a later sync indexes them.
				still_unindexed.push_back({range.start, sub_end});
			}
			count += range_count;
			if (sub_end < range.end) {
				range.start = sub_end;
				break;
			}
			range_idx++;
		}
		if (count == 0) {
			continue;
		}
		const auto row_group_idx = LookupRowGroup(bounds.row_start);
		if (row_group_idx.IsValid()) {
			// The appends load a paged row group fully: the end of the sync frees what it does not keep.
			if (cluster_paging) {
				const auto range = GetRowGroupRange(row_group_idx.GetIndex());
				UpdateReservedMemory(
				    static_cast<int64_t>(EstimateBuildHeapBytes(static_cast<idx_t>(range.end - range.start))));
			}
			for (idx_t i = 0; i < count; i++) {
				AppendRow(row_group_idx.GetIndex(), row_ids[i], embeddings + i * num_dimensions);
			}
		} else {
			SetUpIndexForRowGroup(row_ids.get(), embeddings, count, bounds.row_start, bounds.count, /*n_threads=*/1);
		}
	}
	unindexed_row_ranges = std::move(still_unindexed);
	has_unindexed_rows = !unindexed_row_ranges.empty();

	// The appends loaded their row groups fully: they go back to temporary chains.
	if (cluster_paging) {
		WriteTemporaryChains();
	}

	// The sync's appends, deletes and removed row groups changed the size without reserving it. Reserving the
	// difference between the size now and what is reserved makes the two equal (it frees when the index shrank).
	const auto in_memory_size = static_cast<int64_t>(pdxearch_wrapper->GetInMemorySizeInBytes());
	UpdateReservedMemory(in_memory_size - static_cast<int64_t>(reserved_memory_bytes.load()));
}

// Called by CREATE INDEX's threads in parallel: each reserves its build's memory while it runs, then the memory its row
// group grew by.
void PDXearchIndex::SetUpIndexForRowGroup(const row_t *const row_ids, const float *const vectors,
                                          const idx_t num_vectors, const row_t row_start, const idx_t count,
                                          const idx_t n_threads) {
	// Without paging, the row groups stay on the heap.
	const auto reader = cluster_paging ? storage_reader.get() : nullptr;
	const auto build_heap_bytes = static_cast<int64_t>(EstimateBuildHeapBytes(count));
	UpdateReservedMemory(build_heap_bytes);
	int64_t memory_growth_bytes;
	try {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			memory_growth_bytes = static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
			                          ->SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_start, count,
			                                                  n_threads, reader, buffer_manager);
		} else {
			memory_growth_bytes = static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
			                          ->SetUpIndexForRowGroup(row_ids, vectors, num_vectors, row_start, count,
			                                                  n_threads, reader, buffer_manager);
		}
	} catch (...) {
		UpdateReservedMemory(-build_heap_bytes);
		throw;
	}
	UpdateReservedMemory(memory_growth_bytes - build_heap_bytes);
}

uint64_t PDXearchIndex::EstimateBuildHeapBytes(const idx_t num_embeddings) const {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->EstimateBuildHeapBytes(num_embeddings);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->EstimateBuildHeapBytes(num_embeddings);
}

unique_ptr<PDX::IIterativeSearch> PDXearchIndex::BeginSearchForRowGroup(
    const idx_t row_group_idx, const float *const preprocessed_query, const idx_t limit, PDX::TopKHeap &top_k_heap,
    const std::vector<size_t> *const passing_row_ids, const std::vector<uint32_t> *const clusters_access_order) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->BeginSearchForRowGroup(row_group_idx, preprocessed_query, limit, top_k_heap, passing_row_ids,
		                             clusters_access_order);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
	    ->BeginSearchForRowGroup(row_group_idx, preprocessed_query, limit, top_k_heap, passing_row_ids,
	                             clusters_access_order);
}

std::vector<uint32_t> PDXearchIndex::GetClustersAccessOrderForRowGroup(const idx_t row_group_idx,
                                                                       const float *const preprocessed_query) {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		return static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
		    ->GetClustersAccessOrderForRowGroup(row_group_idx, preprocessed_query);
	}
	return static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
	    ->GetClustersAccessOrderForRowGroup(row_group_idx, preprocessed_query);
}

/******************************************************************
 * Index maintenance
 ******************************************************************/

ErrorData PDXearchIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	// Only the row ids are needed, so the index expressions are not evaluated.
	return Insert(lock, entries, row_ids);
}

// Called by DuckDB at commit with the final row ids, before the rows enter the table's row groups. Which row group
// they land in is not knowable here, so only the ids are recorded; SyncWithTable indexes them once the table knows.
ErrorData PDXearchIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	auto _lock = rwlock.GetExclusiveLock();

	const idx_t count = data.size();
	if (count == 0) {
		return ErrorData();
	}
	row_ids.Flatten(count);
	const auto row_id_data = FlatVector::GetData<row_t>(row_ids);
	// Determine consecutive ranges of row ids.
	// For example: `1, 2, 3, 5, 6 --> [1, 3] + [5, 6]`
	PDXearchRowRange range {row_id_data[0], row_id_data[0] + 1};
	for (idx_t i = 1; i < count; i++) {
		if (row_id_data[i] == range.end) {
			range.end++;
			continue;
		}
		StageUnindexedRange(range);
		range = {row_id_data[i], row_id_data[i] + 1};
	}
	StageUnindexedRange(range);
	has_unindexed_rows = true;
	return ErrorData();
}

// Called by DuckDB at commit. Row ids that are not in the index (never indexed, or reverted before they were) are
// skipped.
void PDXearchIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_ids) {
	auto _lock = rwlock.GetExclusiveLock();

	const idx_t count = entries.size();
	row_ids.Flatten(count);
	const auto row_id_data = FlatVector::GetData<row_t>(row_ids);
	for (idx_t i = 0; i < count; i++) {
		// PDX Delete is idempotent: if the row was never indexed, it is a no-op
		DeleteRow(row_id_data[i]);
		RemoveUnindexedRow(row_id_data[i]);
	}
	has_unindexed_rows = !unindexed_row_ranges.empty();
}

void PDXearchIndex::StageUnindexedRange(const PDXearchRowRange range) {
	auto &ranges = unindexed_row_ranges;
	// The first staged range that starts after this one.
	auto it = std::upper_bound(ranges.begin(), ranges.end(), range.start,
	                           [](row_t start, const PDXearchRowRange &staged) { return start < staged.start; });
	if (it != ranges.begin() && std::prev(it)->end > range.start) {
		--it;
		it->end = MaxValue<row_t>(it->end, range.end);
	} else {
		it = ranges.insert(it, range);
	}
	// Absorb the following ranges this one now overlaps.
	auto next = std::next(it);
	while (next != ranges.end() && next->start < it->end) {
		it->end = MaxValue<row_t>(it->end, next->end);
		next = ranges.erase(next);
	}
}

// A deleted row that was never indexed has nothing left to index: take it out of its staged range.
void PDXearchIndex::RemoveUnindexedRow(const row_t row_id) {
	// Find staged range [start, end) that contains row_id, if any
	auto it = std::upper_bound(unindexed_row_ranges.begin(), unindexed_row_ranges.end(), row_id,
	                           [](row_t id, const PDXearchRowRange &range) { return id < range.start; });
	if (it == unindexed_row_ranges.begin()) {
		return;
	}
	--it;
	// row_id is not staged if it falls in a gap
	if (row_id >= it->end) {
		return;
	}
	// row_id is the first of the range, we just shrink the range by moving the start up one
	if (it->start == row_id) {
		it->start++;
	} else if (it->end == row_id + 1) { // idem (with the tail)
		it->end--;
	} else {
		// row_id is in the middle of the range, we split it into two ranges
		const PDXearchRowRange tail {row_id + 1, it->end};
		it->end = row_id;
		unindexed_row_ranges.insert(it + 1, tail);
		return;
	}
	// If the range is now empty, remove it
	if (it->start == it->end) {
		unindexed_row_ranges.erase(it);
	}
}

// Called by DROP INDEX.
// Dropping the allocator's buffers marks their blocks modified; DuckDB frees them at the next checkpoint.
void PDXearchIndex::ResetStorage(IndexLock &lock) {
	auto _lock = rwlock.GetExclusiveLock();

	allocator->Reset();
	directory_chain = PDXearchBlockChain();
	rotation_chain = PDXearchBlockChain();
	ResetPersistedChains();
}

bool PDXearchIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("PDXearchIndex::MergeIndexes() not implemented");
}

void PDXearchIndex::Vacuum(IndexLock &state) {
}

void PDXearchIndex::Verify(IndexLock &state) {
	throw NotImplementedException("PDXearchIndex::Verify() not implemented");
}

string PDXearchIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("PDXearchIndex::ToString() not implemented");
}

void PDXearchIndex::VerifyAllocations(IndexLock &state) {
	throw NotImplementedException("PDXearchIndex::VerifyAllocations() not implemented");
}

// Called in debug builds: at most one empty buffer.
void PDXearchIndex::VerifyBuffers(IndexLock &state) {
	allocator->VerifyBuffers();
}

idx_t PDXearchIndex::GetInMemorySize(IndexLock &state) {
	auto _lock = rwlock.GetSharedLock();

	// The allocator only holds the buffers that were not written to disk yet.
	return pdxearch_wrapper->GetInMemorySizeInBytes() + allocator->GetInMemorySize();
}

unique_ptr<PDXearchIndexStats> PDXearchIndex::GetStats(const ClientContext &context) const {
	auto result = make_uniq<PDXearchIndexStats>();

	// Make sure to sync any changes made here to `pdxearch_index_info_function.cpp`.
	result->metric = GetDistanceMetric();
	result->quantization = GetQuantization();
	result->num_dimensions = static_cast<int64_t>(GetNumDimensions());
	result->n_probe = static_cast<int64_t>(pdxearch_wrapper->GetNProbe());
	result->seed = static_cast<int64_t>(pdxearch_wrapper->GetSeed());
	result->is_normalized = IsNormalized();
	result->approximate_lower_bound_memory_usage_bytes =
	    static_cast<int64_t>(pdxearch_wrapper->GetInMemorySizeInBytes());
	if (storage_reader) {
		const auto &counters = storage_reader->counters;
		result->cluster_acquires = static_cast<int64_t>(counters.cluster_acquires.load());
		result->cluster_cache_misses = static_cast<int64_t>(counters.cluster_cache_misses.load());
		result->cluster_bytes_fetched = static_cast<int64_t>(counters.cluster_bytes_fetched.load());
		result->blocks_read = static_cast<int64_t>(counters.blocks_read.load());
	}

	return result;
}

/******************************************************************
 * Index persistence
 ******************************************************************/

unique_ptr<PDXearchBlockChainReader> PDXearchIndex::OpenStorage(const IndexStorageInfo &info,
                                                                const uint32_t num_dimensions,
                                                                PDXearchDirectory &directory,
                                                                unique_ptr<float[]> &rotation_matrix) {
	const auto version = info.options.find("storage_version");
	if (version == info.options.end() || version->second.GetValue<uint32_t>() != PDXEARCH_STORAGE_VERSION) {
		throw NotImplementedException("The PDXearch index \"%s\" was persisted in a storage format this version of the "
		                              "extension cannot read. Update the extension, or DROP INDEX and create it again.",
		                              name);
	}
	const auto &allocator_info = info.allocator_infos[0];
	if (allocator_info.segment_size != allocator->GetSegmentSize()) {
		throw SerializationException("The PDXearch index \"%s\" has segments of %llu bytes instead of %llu", name,
		                             allocator_info.segment_size, allocator->GetSegmentSize());
	}
	// Registers the persisted blocks without reading them: the next checkpoints keep or free them.
	allocator->Init(allocator_info);

	auto reader = make_uniq<PDXearchBlockChainReader>(table_io_manager.GetIndexBlockManager(), allocator_info);
	std::istream in(reader.get());
	PDXearchBlockChain root;
	root.head.Set(info.root);
	root.num_bytes = info.options.at("directory_bytes").GetValue<idx_t>();
	reader->Open(root);
	directory = PDXearchDirectory::Read(in);
	directory_chain = reader->Finish();
	if (directory.num_dimensions != num_dimensions) {
		throw SerializationException("The PDXearch index \"%s\" was persisted with %u dimensions instead of %u", name,
		                             directory.num_dimensions, num_dimensions);
	}
	// A directory written to the WAL has no rotation.
	if (directory.rotation.num_bytes > 0) {
		const size_t num_values = static_cast<size_t>(num_dimensions) * num_dimensions;
		rotation_matrix = make_uniq_array<float>(num_values);
		reader->Open(directory.rotation);
		PDX::StreamReader rotation_reader {in};
		rotation_reader.Read(rotation_matrix.get(), sizeof(float) * num_values);
		rotation_chain = reader->Finish();
	}
	return reader;
}

// Reserves each row group's memory as it loads, so DuckDB evicts other data while the index grows. A row group loaded
// fully is reserved before it is read.
void PDXearchIndex::LoadRowGroups(PDXearchBlockChainReader &reader, const PDXearchDirectory &directory,
                                  const bool page_clusters) {
	for (const auto &entry : directory.row_groups) {
		int64_t load_bytes = 0;
		if (!page_clusters) {
			load_bytes =
			    static_cast<int64_t>(EstimateBuildHeapBytes(static_cast<idx_t>(entry.row_end - entry.row_start)));
			UpdateReservedMemory(load_bytes);
		}
		uint64_t in_memory_size;
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			in_memory_size = static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
			                     ->LoadRowGroup(reader, entry, buffer_manager, page_clusters);
		} else {
			in_memory_size = static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
			                     ->LoadRowGroup(reader, entry, buffer_manager, page_clusters);
		}
		UpdateReservedMemory(static_cast<int64_t>(in_memory_size) - load_bytes);
	}
}

// A rewritten row group can be loaded fully, one at a time. A checkpoint must not fail: without the memory, it goes on
// unreserved.
void PDXearchIndex::PersistDirtyRowGroups(const std::function<void()> &write_partial_blocks) {
	auto materialize_bytes = static_cast<int64_t>(EstimateBuildHeapBytes(GetRowGroupSize()));
	try {
		UpdateReservedMemory(materialize_bytes);
	} catch (OutOfMemoryException &) {
		materialize_bytes = 0;
	}
	try {
		if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
			static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())
			    ->PersistDirtyRowGroups(*allocator, write_partial_blocks, storage_reader.get(), buffer_manager,
			                            cluster_paging);
		} else {
			static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
			    ->PersistDirtyRowGroups(*allocator, write_partial_blocks, storage_reader.get(), buffer_manager,
			                            cluster_paging);
		}
	} catch (...) {
		UpdateReservedMemory(-materialize_bytes);
		throw;
	}
	UpdateReservedMemory(-materialize_bytes);
}

void PDXearchIndex::WriteTemporaryChains() {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->WriteTemporaryChains(*storage_reader, buffer_manager);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())
		    ->WriteTemporaryChains(*storage_reader, buffer_manager);
	}
}

void PDXearchIndex::AddRowGroupEntries(PDXearchDirectory &directory) const {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->AddRowGroupEntries(directory);
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->AddRowGroupEntries(directory);
	}
}

void PDXearchIndex::ResetPersistedChains() {
	if (pdxearch_wrapper->GetQuantization() == PDX::U8) {
		static_cast<PDXearchWrapperU8 *>(pdxearch_wrapper.get())->ResetPersistedChains();
	} else {
		static_cast<PDXearchWrapperF32 *>(pdxearch_wrapper.get())->ResetPersistedChains();
	}
}

// Like ART::WritePartialBlocks. Flushing every time keeps a later New() from landing in a buffer whose block is not on
// disk yet. Empty buffers go first: DuckDB asserts that the buffers it serializes are not empty.
void PDXearchIndex::WritePartialBlocks(QueryContext context) {
	allocator->RemoveEmptyBuffers();
	PartialBlockManager partial_block_manager(context, table_io_manager.GetIndexBlockManager(),
	                                          PartialBlockType::FULL_CHECKPOINT);
	allocator->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
}

void PDXearchIndex::WriteDirectory(const PDXearchDirectory &directory) {
	directory_chain.Free(*allocator);
	PDXearchBlockChainWriter writer(*allocator);
	std::ostream out(&writer);
	directory.Write(out);
	directory_chain = writer.Finish();
}

IndexStorageInfo PDXearchIndex::MakeStorageInfo() const {
	IndexStorageInfo info(name);
	info.root = directory_chain.head.Get();
	info.options.emplace("storage_version", Value::UINTEGER(PDXEARCH_STORAGE_VERSION));
	info.options.emplace("directory_bytes", Value::UBIGINT(directory_chain.num_bytes));
	info.options.emplace("metric", Value(GetDistanceMetric()));
	info.options.emplace("quantization", Value(GetQuantization()));
	info.options.emplace("n_probe", Value::INTEGER(static_cast<int32_t>(pdxearch_wrapper->GetNProbe())));
	info.options.emplace("seed", Value::INTEGER(pdxearch_wrapper->GetSeed()));
	info.allocator_infos.push_back(allocator->GetInfo());
	return info;
}

// Called at every checkpoint, also when nothing changed: only the dirty row groups and the directory are rewritten.
IndexStorageInfo PDXearchIndex::SerializeToDisk(QueryContext context,
                                                const case_insensitive_map_t<Value> &serialization_options) {
	auto _lock = rwlock.GetExclusiveLock();

	PersistDirtyRowGroups([&]() { WritePartialBlocks(context); });
	// Written once: the rotation never changes.
	if (rotation_chain.segments.empty()) {
		const auto num_dimensions = GetNumDimensions();
		PDXearchBlockChainWriter writer(*allocator);
		std::ostream out(&writer);
		out.write(reinterpret_cast<const char *>(GetRotationMatrix()),
		          static_cast<std::streamsize>(sizeof(float) * num_dimensions * num_dimensions));
		rotation_chain = writer.Finish();
	}

	PDXearchDirectory directory;
	directory.num_dimensions = static_cast<uint32_t>(GetNumDimensions());
	directory.rotation = PDXearchBlockChain {rotation_chain.head, rotation_chain.num_bytes, {}};
	AddRowGroupEntries(directory);
	WriteDirectory(directory);
	WritePartialBlocks(context);
	// The checkpoint can have moved live segments to other blocks.
	if (storage_reader) {
		storage_reader->UpdateBlockPointers(allocator->GetInfo());
	}
	// The rewritten row groups were paged again: free what they no longer use. Reserving more could throw, which a
	// checkpoint must not: the next sync reserves any growth.
	const auto in_memory_size = static_cast<int64_t>(pdxearch_wrapper->GetInMemorySizeInBytes());
	const auto reserved = static_cast<int64_t>(reserved_memory_bytes.load());
	if (in_memory_size < reserved) {
		UpdateReservedMemory(in_memory_size - reserved);
	}
	return MakeStorageInfo();
}

// Called when CREATE INDEX commits. The WAL only gets an empty directory and the options: if the process stops before
// the next checkpoint, replay binds an empty index and its first sync indexes the table. Logging the whole index would
// keep it in memory twice until that checkpoint (2x the index size).
IndexStorageInfo PDXearchIndex::SerializeToWAL(const case_insensitive_map_t<Value> &serialization_options) {
	auto _lock = rwlock.GetExclusiveLock();

	PDXearchDirectory directory;
	directory.num_dimensions = static_cast<uint32_t>(GetNumDimensions());
	WriteDirectory(directory);
	// Sets the buffers' allocation sizes, which GetInfo reads.
	auto buffers = allocator->InitSerializationToWAL();
	auto info = MakeStorageInfo();
	info.buffers.push_back(std::move(buffers));
	return info;
}

/******************************************************************
 * Misc.
 ******************************************************************/

bool PDXearchIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                             vector<reference<Expression>> &bindings) const {
	return function_matcher->Match(*expr, bindings);
}

static void TryBindIndexExpressionInternal(Expression &expr, idx_t table_idx, const vector<column_t> &index_columns,
                                           const vector<ColumnIndex> &table_columns, bool &success, bool &found) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();

		// Rewrite the column reference to fit in the current set of bound column ids
		ref.binding.table_index = table_idx;

		const auto referenced_column = index_columns[ref.binding.column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == referenced_column) {
				ref.binding.column_index = i;
				return;
			}
		}
		success = false;
	}

	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		TryBindIndexExpressionInternal(child, table_idx, index_columns, table_columns, success, found);
	});
}

bool PDXearchIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr_ptr = unbound_expressions.back()->Copy();

	auto &expr = *expr_ptr;
	auto &index_columns = GetColumnIds();
	auto &table_columns = get.GetColumnIds();

	auto success = true;
	auto found = false;

	TryBindIndexExpressionInternal(expr, get.table_index, index_columns, table_columns, success, found);

	if (success && found) {
		result = std::move(expr_ptr);
		return true;
	}
	return false;
}

string PDXearchIndex::GetQuantization() const {
	switch (pdxearch_wrapper->GetQuantization()) {
	case PDX::Quantization::F32:
		return "f32";
	case PDX::Quantization::U8:
		return "u8";
	default:
		throw InternalException("Unknown quantization");
	}
}

const case_insensitive_map_t<PDX::Quantization> PDXearchIndex::QUANTIZATION_MAP = {
    {"f32", PDX::Quantization::F32},
    {"u8", PDX::Quantization::U8},
};

string PDXearchIndex::GetDistanceMetric() const {
	switch (pdxearch_wrapper->GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		return "l2sq";
	case PDX::DistanceMetric::COSINE:
		return "cosine";
	// case PDX::DistanceMetric::IP:
	// 	return "ip";
	default:
		throw InternalException("Unknown distance metric");
	}
}

const case_insensitive_map_t<PDX::DistanceMetric> PDXearchIndex::DISTANCE_METRIC_MAP = {
    {"l2sq", PDX::DistanceMetric::L2SQ}, {"cosine", PDX::DistanceMetric::COSINE},
    // {"ip", PDX::DistanceMetric::IP},
};

unique_ptr<ExpressionMatcher> PDXearchIndex::MakeFunctionMatcher(const PDXearchWrapper &pdxearch_wrapper) {
	unordered_set<string> distance_functions;

	switch (pdxearch_wrapper.GetDistanceMetric()) {
	case PDX::DistanceMetric::L2SQ:
		distance_functions = {"array_distance", "<->"};
		break;
	case PDX::DistanceMetric::COSINE:
		distance_functions = {"array_cosine_distance", "<=>"};
		break;
	// case PDX::DistanceMetric::IP:
	// 	distance_functions = {"array_negative_inner_product", "<#>"};
	//  break;
	default:
		throw NotImplementedException("Unknown distance metric");
	}

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto array_type = LogicalType::ARRAY(LogicalType::FLOAT, pdxearch_wrapper.GetNumDimensions());

	auto lhs_matcher = make_uniq<ExpressionMatcher>();
	lhs_matcher->type = make_uniq<SetTypesMatcher>(vector<LogicalType>({array_type, LogicalType::BLOB}));
	matcher->matchers.push_back(std::move(lhs_matcher));

	auto rhs_matcher = make_uniq<ExpressionMatcher>();
	rhs_matcher->type = make_uniq<SetTypesMatcher>(vector<LogicalType>({array_type, LogicalType::BLOB}));
	matcher->matchers.push_back(std::move(rhs_matcher));

	return std::move(matcher);
}

string ColumnNamesToString(const TableCatalogEntry &table, const vector<ColumnIndex> &column_ids) {
	string result;
	for (auto &column_id : column_ids) {
		if (!result.empty()) {
			result += "\n";
		}
		if (column_id.IsRowIdColumn()) {
			result += "rowid";
		} else {
			result += table.GetColumn(LogicalIndex(column_id.GetPrimaryIndex())).Name();
		}
	}
	return result;
}

void PDXearchModule::RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;

	index_type.name = PDXearchIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<PDXearchIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                    input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = PDXearchIndex::CreatePlan;

	db.config.AddExtensionOption("pdxearch_n_probe",
	                             "override the n_probe parameter when scanning PDXearch indexes (default: " +
	                                 to_string(PDXearchWrapper::DEFAULT_N_PROBE) + ", must be >= 0)",
	                             LogicalType::INTEGER, Value());
	db.config.AddExtensionOption("pdxearch_rank_clusters_across_row_groups",
	                             "LATERAL index joins probe clusters nearest to the "
	                             "query across all row groups (default: true); false probes the n_probe nearest of "
	                             "each row group",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true));
	db.config.AddExtensionOption(
	    "pdxearch_bruteforce_rows_per_cluster",
	    "filtered LATERAL index joins search the passing rows by brute force, without the index, when they number "
	    "at most this many per cluster of the row groups they are in, and at most pdxearch_bruteforce_max_rows "
	    "(default: 2, 0 disables it)",
	    LogicalType::DOUBLE, Value::DOUBLE(2.0));
	db.config.AddExtensionOption("pdxearch_bruteforce_max_rows",
	                             "the most passing rows a filtered LATERAL index join searches by brute force "
	                             "(default: 100000, 0 disables it)",
	                             LogicalType::UBIGINT, Value::UBIGINT(100000));
	db.config.AddExtensionOption(
	    "pdxearch_experimental_on_the_fly_indexing",
	    "filtered LATERAL index joins whose passing rows are too many for a brute-force search build indexes over only "
	    "the passing rows and search them instead of the row groups: 'always', 'never' or 'auto' (when there are at "
	    "most pdxearch_on_the_fly_indexing_threshold passing rows per query, queries as estimated; default: never)",
	    LogicalType::VARCHAR, Value("never"), [](ClientContext &, SetScope, Value &parameter) {
		    const auto mode = StringUtil::Lower(parameter.ToString());
		    if (mode != "always" && mode != "never" && mode != "auto") {
			    throw InvalidInputException(
			        "pdxearch_experimental_on_the_fly_indexing must be 'always', 'never' or 'auto', not '%s'",
			        parameter.ToString());
		    }
		    parameter = Value(mode);
	    });
	db.config.AddExtensionOption(
	    "pdxearch_on_the_fly_indexing_threshold",
	    "with pdxearch_experimental_on_the_fly_indexing = 'auto', the indexes over the passing rows are built when "
	    "there are at most this many passing rows per query, queries as estimated (default: 20)",
	    LogicalType::UBIGINT, Value::UBIGINT(20));
	db.config.AddExtensionOption("pdxearch_on_the_fly_indexing_max_row_groups",
	                             "the most consecutive row groups whose passing rows go into one index built on the "
	                             "fly (default: 0, no limit: the memory left decides)",
	                             LogicalType::UBIGINT, Value::UBIGINT(0));
	db.config.AddExtensionOption("pdxearch_cluster_paging",
	                             "indexes loaded from a database file read their clusters from it as searches need "
	                             "them, so they can be larger than memory_limit; false loads them fully when they "
	                             "load (default: true)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true), nullptr, SetScope::GLOBAL);
	db.config.AddExtensionOption("pdxearch_paging_counters",
	                             "indexes loaded with it on count their cluster cache's acquires, misses and bytes "
	                             "read, and the blocks read, in pdxearch_index_info (default: false)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(false), nullptr, SetScope::GLOBAL);
	db.config.AddExtensionOption("pdxearch_cache_tiers",
	                             "in indexes loaded with it on, the clusters searches read most often are the last "
	                             "DuckDB evicts from memory (default: true)",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(true), nullptr, SetScope::GLOBAL);

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}

} // namespace duckdb

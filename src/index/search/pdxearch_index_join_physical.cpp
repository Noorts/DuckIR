#include "index/search/pdxearch_index_join_physical.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

#include "index/pdxearch_blob_codec.hpp"
#include "index/pdxearch_index.hpp"

#include "pdx/ivf_searcher.hpp"

#include <algorithm>
#include <numeric>

namespace duckdb {

PhysicalPDXearchIndexJoin::PhysicalPDXearchIndexJoin(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                     DuckTableEntry &table, Index &index, idx_t limit,
                                                     idx_t query_column, const vector<ColumnIndex> &column_ids,
                                                     idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalPDXearchIndexJoin::TYPE, std::move(types), estimated_cardinality),
      table(table), index(index), limit(limit), query_column(query_column), column_ids(column_ids) {
	D_ASSERT(limit <= STANDARD_VECTOR_SIZE);
	fetch_column_ids.reserve(column_ids.size());
	for (auto &id : column_ids) {
		if (id.IsRowIdColumn()) {
			fetch_column_ids.emplace_back();
		} else {
			auto &col = table.GetColumn(LogicalIndex(id.GetPrimaryIndex()));
			fetch_column_ids.emplace_back(col.StorageOid());
		}
	}
}

// pdxearch_on_the_fly_indexing: whether a filtered join builds indexes over the passing rows.
enum class OnTheFlyIndexMode : uint8_t { NEVER, ALWAYS, AUTO };

// The part of the search that is the same for every query, set up once per operator.
class PDXearchIndexJoinSearch {
public:
	PDXearchIndexJoinSearch(ClientContext &context, const PhysicalPDXearchIndexJoin &op)
	    : search_lock(op.index.Cast<PDXearchIndex>().SyncAndLockForSearch(op.table.GetStorage())),
	      index(op.index.Cast<PDXearchIndex>()), limit(op.limit),
	      preprocessor(index.GetNumDimensions(), index.GetRotationMatrix()) {
		auto n_probe = index.GetEffectiveNProbe(context);
		const idx_t num_clusters_for_full_row_group = PDXearchIndex::GetNumClustersForFullRowGroup();
		clusters_to_probe_on_first_iteration =
		    (n_probe == 0 || n_probe > num_clusters_for_full_row_group) ? num_clusters_for_full_row_group : n_probe;
		Value rank_clusters_across_row_groups_setting;
		if (context.TryGetCurrentSetting("pdxearch_rank_clusters_across_row_groups",
		                                 rank_clusters_across_row_groups_setting) &&
		    !rank_clusters_across_row_groups_setting.IsNull()) {
			rank_clusters_across_row_groups = rank_clusters_across_row_groups_setting.GetValue<bool>();
		}
		Value max_passing_rows_per_cluster_for_flat_search_setting;
		if (context.TryGetCurrentSetting("pdxearch_bruteforce_rows_per_cluster",
		                                 max_passing_rows_per_cluster_for_flat_search_setting) &&
		    !max_passing_rows_per_cluster_for_flat_search_setting.IsNull()) {
			max_passing_rows_per_cluster_for_flat_search =
			    max_passing_rows_per_cluster_for_flat_search_setting.GetValue<double>();
		}
		Value max_passing_rows_for_flat_search_setting;
		if (context.TryGetCurrentSetting("pdxearch_bruteforce_max_rows", max_passing_rows_for_flat_search_setting) &&
		    !max_passing_rows_for_flat_search_setting.IsNull()) {
			max_passing_rows_for_flat_search = max_passing_rows_for_flat_search_setting.GetValue<uint64_t>();
		}
		Value on_the_fly_index_setting;
		if (context.TryGetCurrentSetting("pdxearch_on_the_fly_indexing", on_the_fly_index_setting) &&
		    !on_the_fly_index_setting.IsNull()) {
			const auto mode = on_the_fly_index_setting.ToString();
			on_the_fly_index = mode == "always" ? OnTheFlyIndexMode::ALWAYS
			                   : mode == "auto" ? OnTheFlyIndexMode::AUTO
			                                    : OnTheFlyIndexMode::NEVER;
		}
		Value min_queries_per_passing_row_for_on_the_fly_index_setting;
		if (context.TryGetCurrentSetting("pdxearch_on_the_fly_indexing_threshold",
		                                 min_queries_per_passing_row_for_on_the_fly_index_setting) &&
		    !min_queries_per_passing_row_for_on_the_fly_index_setting.IsNull()) {
			min_queries_per_passing_row_for_on_the_fly_index =
			    min_queries_per_passing_row_for_on_the_fly_index_setting.GetValue<double>();
		}
		Value max_row_groups_per_on_the_fly_index_setting;
		if (context.TryGetCurrentSetting("pdxearch_on_the_fly_indexing_max_row_groups",
		                                 max_row_groups_per_on_the_fly_index_setting) &&
		    !max_row_groups_per_on_the_fly_index_setting.IsNull()) {
			max_row_groups_per_on_the_fly_index = max_row_groups_per_on_the_fly_index_setting.GetValue<uint64_t>();
		}
	}

	// Held for the duration of execution to serialize searches against index maintenance.
	// Taken once per operator: a thread must not take it again while it holds it.
	unique_ptr<StorageLockKey> search_lock;

	PDXearchIndex &index;
	// The number of nearest rows per query (`k`).
	const idx_t limit;
	// Shared by all threads: it holds a copy of the rotation matrix, and preprocessing a query only reads it.
	const EmbeddingPreprocessor preprocessor;

	idx_t clusters_to_probe_on_first_iteration {0};
	static constexpr idx_t CLUSTERS_TO_PROBE_PER_FOLLOW_UP_ITERATION = 5;
	bool rank_clusters_across_row_groups {true};
	double max_passing_rows_per_cluster_for_flat_search {2.0};
	idx_t max_passing_rows_for_flat_search {100000};
	OnTheFlyIndexMode on_the_fly_index {OnTheFlyIndexMode::NEVER};
	double min_queries_per_passing_row_for_on_the_fly_index {0.05};
	idx_t max_row_groups_per_on_the_fly_index {0};
	// The build's peak memory beyond the indexes, in embeddings of the largest window: its gathered embeddings (1) and
	// the k-means scratch (0.1).
	static constexpr double OTF_INDEX_BUILD_PEAK_MEMORY_FACTOR = 1.1;

	bool is_filtered {false};
	vector<std::vector<size_t>> passing_row_ids_per_row_group;
	vector<std::unique_ptr<PDX::PredicateEvaluator>> shared_predicate_evaluators;
	std::unique_ptr<PDX::ADSamplingPruner> passing_rows_otf_indexes_pruner;
	vector<std::unique_ptr<PDX::IPDXIndex>> passing_rows_otf_indexes;
	vector<row_t> passing_rows_otf_indexes_row_ids;
	idx_t passing_rows_otf_indexes_clusters_to_probe {0};
};

// Holds the search state when the table is filtered (the sink runs first).
class PDXearchIndexJoinGlobalSinkState : public GlobalSinkState {
public:
	PDXearchIndexJoinGlobalSinkState(ClientContext &context, const PhysicalPDXearchIndexJoin &op)
	    : search(context, op) {
		search.is_filtered = true;
		search.passing_row_ids_per_row_group.resize(search.index.GetNumRowGroups());
	}

	PDXearchIndexJoinSearch search;
};

// Holds the search state when the table is not filtered (there is no sink).
class PDXearchIndexJoinGlobalOperatorState : public GlobalOperatorState {
public:
	PDXearchIndexJoinGlobalOperatorState(ClientContext &context, const PhysicalPDXearchIndexJoin &op) {
		if (!op.IsSink()) {
			search = make_uniq<PDXearchIndexJoinSearch>(context, op);
		}
	}

	unique_ptr<PDXearchIndexJoinSearch> search;
};

unique_ptr<GlobalOperatorState> PhysicalPDXearchIndexJoin::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<PDXearchIndexJoinGlobalOperatorState>(context, *this);
}

unique_ptr<GlobalSinkState> PhysicalPDXearchIndexJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<PDXearchIndexJoinGlobalSinkState>(context, *this);
}

// ------------------------------
// On filtered search, Sink only collects the rowids of the rows that pass the predicate.
// At this point, the query rows have not come into the pipeline yet.
// ------------------------------

class PDXearchIndexJoinLocalSinkState : public LocalSinkState {
public:
	explicit PDXearchIndexJoinLocalSinkState(const PDXearchIndex &index)
	    : passing_row_ids_per_row_group(index.GetNumRowGroups()) {
	}
	idx_t current_row_group_id {0};
	PDXearchRowRange current_row_group_range {0, 0};
	vector<std::vector<size_t>> passing_row_ids_per_row_group;
};

unique_ptr<LocalSinkState> PhysicalPDXearchIndexJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<PDXearchIndexJoinLocalSinkState>(index.Cast<PDXearchIndex>());
}

SinkResultType PhysicalPDXearchIndexJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSinkInput &input) const {
	auto &l_sink = input.local_state.Cast<PDXearchIndexJoinLocalSinkState>();
	auto &pdxearch_index = index.Cast<PDXearchIndex>();

	D_ASSERT(chunk.ColumnCount() == 1);
	D_ASSERT(chunk.data[0].GetType() == LogicalType::ROW_TYPE);

	chunk.data[0].Flatten(chunk.size());
	const auto row_ids = FlatVector::GetData<row_t>(chunk.data[0]);
	// No search starts here (the queries come later), so the rows only go to their row group's list
	for (idx_t i = 0; i < chunk.size(); i++) {
		const row_t row_id = row_ids[i];
		if (row_id < l_sink.current_row_group_range.start || row_id >= l_sink.current_row_group_range.end) {
			const auto row_group_idx = pdxearch_index.LookupRowGroup(row_id);
			if (!row_group_idx.IsValid()) {
				// Rows without an embedding are not in the index.
				continue;
			}
			l_sink.current_row_group_id = row_group_idx.GetIndex();
			l_sink.current_row_group_range = pdxearch_index.GetRowGroupRange(l_sink.current_row_group_id);
		}
		l_sink.passing_row_ids_per_row_group[l_sink.current_row_group_id].push_back(static_cast<size_t>(row_id));
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalPDXearchIndexJoin::Combine(ExecutionContext &context,
                                                         OperatorSinkCombineInput &input) const {
	auto &g_sink = input.global_state.Cast<PDXearchIndexJoinGlobalSinkState>();
	auto &l_sink = input.local_state.Cast<PDXearchIndexJoinLocalSinkState>();

	const auto guard = g_sink.Lock();
	for (idx_t row_group_idx = 0; row_group_idx < l_sink.passing_row_ids_per_row_group.size(); row_group_idx++) {
		auto &local_row_ids = l_sink.passing_row_ids_per_row_group[row_group_idx];
		auto &global_row_ids = g_sink.search.passing_row_ids_per_row_group[row_group_idx];
		global_row_ids.insert(global_row_ids.end(), local_row_ids.begin(), local_row_ids.end());
	}
	return SinkCombineResultType::FINISHED;
}

// The passing rows of the largest on-the-fly index when the row groups with passing rows are split into windows of
// row_groups_per_window consecutive ones.
static idx_t CountMaxPassingRowsPerOtfIndex(const vector<std::vector<size_t>> &passing_row_ids_per_row_group,
                                            const idx_t row_groups_per_window) {
	idx_t max_rows = 0;
	idx_t window_rows = 0;
	idx_t window_row_groups = 0;
	for (const auto &row_ids : passing_row_ids_per_row_group) {
		if (row_ids.empty()) {
			continue;
		}
		window_rows += row_ids.size();
		if (++window_row_groups == row_groups_per_window) {
			max_rows = MaxValue<idx_t>(max_rows, window_rows);
			window_rows = 0;
			window_row_groups = 0;
		}
	}
	// The last window can hold fewer row groups.
	return MaxValue<idx_t>(max_rows, window_rows);
}

// Builds the indexes that the queries search instead of the row groups: the row groups with passing rows are split
// into windows of row_groups_per_window consecutive ones, and the passing rows of each window are gathered from their
// row groups' indexes into one index (Flat when build_flat or when too few to cluster; else an IVF with PDX's default
// number of clusters). One window after another, so that only one window's gathered embeddings are held at a time.
// The rows are numbered by position in passing_rows_otf_indexes_row_ids.
static void GatherPassingRowsIntoOtfIndexes(PDXearchIndexJoinSearch &search, const idx_t row_groups_per_window,
                                            const bool build_flat, const idx_t n_threads) {
	const idx_t num_dimensions = search.index.GetNumDimensions();
	auto &passing_row_ids_per_row_group = search.passing_row_ids_per_row_group;
	// Rows without an embedding are not in the index.
	vector<idx_t> row_groups_with_passing_rows;
	idx_t num_rows = 0;
	for (idx_t row_group_idx = 0; row_group_idx < passing_row_ids_per_row_group.size(); row_group_idx++) {
		const auto &row_group_index = search.index.GetRowGroupIndex(row_group_idx);
		auto &row_ids = passing_row_ids_per_row_group[row_group_idx];
		row_ids.erase(std::remove_if(row_ids.begin(), row_ids.end(),
		                             [&](const size_t row_id) { return !row_group_index.Contains(row_id); }),
		              row_ids.end());
		if (!row_ids.empty()) {
			row_groups_with_passing_rows.push_back(row_group_idx);
			num_rows += row_ids.size();
		}
	}
	if (row_groups_with_passing_rows.empty()) {
		return;
	}
	const idx_t max_rows_per_window =
	    CountMaxPassingRowsPerOtfIndex(passing_row_ids_per_row_group, row_groups_per_window);
	auto embeddings = make_uniq_array_uninitialized<float>(max_rows_per_window * num_dimensions);
	std::vector<size_t> positions(max_rows_per_window);
	PDX::PDXIndexConfig config;
	config.num_dimensions = static_cast<uint32_t>(num_dimensions);
	config.distance_metric = PDX::DistanceMetric::L2SQ;
	config.is_data_transformed = true;
	// As the row groups' indexes.
	config.kmeans_iters = 8;
	config.hierarchical_indexing = true;
	// The indexes set PDX's process-wide thread count from their config.
	config.n_threads = static_cast<uint32_t>(n_threads);
	search.passing_rows_otf_indexes_pruner =
	    std::make_unique<PDX::ADSamplingPruner>(num_dimensions, search.index.GetRotationMatrix());
	// A window of C clusters probes n_probe x sqrt(C / (clusters of a full row group)) of them, which keeps the recall
	// of n_probe in the row groups; n_probe 0 (or all of a row group's clusters) probes all of them.
	const idx_t num_clusters_for_full_row_group = PDXearchIndex::GetNumClustersForFullRowGroup();
	const bool probe_all_clusters = search.clusters_to_probe_on_first_iteration >= num_clusters_for_full_row_group;
	const double clusters_to_probe_per_sqrt_cluster = static_cast<double>(search.clusters_to_probe_on_first_iteration) /
	                                                  std::sqrt(static_cast<double>(num_clusters_for_full_row_group));
	auto &otf_row_ids = search.passing_rows_otf_indexes_row_ids;
	otf_row_ids.reserve(num_rows);
	for (idx_t first = 0; first < row_groups_with_passing_rows.size(); first += row_groups_per_window) {
		const idx_t end = MinValue<idx_t>(first + row_groups_per_window, row_groups_with_passing_rows.size());
		const idx_t window_offset = otf_row_ids.size();
		for (idx_t i = first; i < end; i++) {
			const auto &row_ids = passing_row_ids_per_row_group[row_groups_with_passing_rows[i]];
			search.index.GetRowGroupIndex(row_groups_with_passing_rows[i])
			    .GetEmbeddingsFromIndexByRowIds(row_ids, embeddings.get() +
			                                                 (otf_row_ids.size() - window_offset) * num_dimensions);
			otf_row_ids.insert(otf_row_ids.end(), row_ids.begin(), row_ids.end());
		}
		const idx_t window_rows = otf_row_ids.size() - window_offset;
		std::iota(positions.begin(), positions.begin() + static_cast<std::ptrdiff_t>(window_rows), window_offset);
		config.base_row_id = window_offset;
		std::unique_ptr<PDX::IPDXIndex> otf_index;
		if (build_flat || window_rows < PDXearchWrapperF32::MIN_EMBEDDINGS_FOR_CLUSTERING) {
			auto flat_index = std::make_unique<PDX::FlatIndex>(config, *search.passing_rows_otf_indexes_pruner);
			flat_index->BuildIndex(positions.data(), embeddings.get(), window_rows);
			otf_index = std::move(flat_index);
		} else {
			auto ivf_index = std::make_unique<PDX::PDXIndexF32>(config, *search.passing_rows_otf_indexes_pruner);
			ivf_index->BuildIndex(positions.data(), embeddings.get(), window_rows);
			otf_index = std::move(ivf_index);
		}
		const idx_t num_clusters = otf_index->GetNumClusters();
		search.passing_rows_otf_indexes_clusters_to_probe +=
		    probe_all_clusters
		        ? num_clusters
		        : MinValue<idx_t>(num_clusters,
		                          static_cast<idx_t>(std::ceil(clusters_to_probe_per_sqrt_cluster *
		                                                       std::sqrt(static_cast<double>(num_clusters)))));
		search.passing_rows_otf_indexes.push_back(std::move(otf_index));
	}
}

// The most consecutive row groups per on-the-fly index (at most pdxearch_on_the_fly_indexing_max_row_groups) whose
// build fits in the memory DuckDB has left: the indexes of all windows stay, and the window being built also holds its
// gathered embeddings and the k-means scratch. 0 when not even one row group per index fits.
static idx_t ChooseRowGroupsPerOtfIndexWithinMemoryBudget(ClientContext &context, const PDXearchIndexJoinSearch &search,
                                                          const idx_t num_passing_rows) {
	auto &buffer_manager = BufferManager::GetBufferManager(context);
	// DuckDB does not count the row groups' indexes as used memory.
	const idx_t used_memory = buffer_manager.GetUsedMemory() + search.index.GetInMemorySizeInBytesWithoutLocking();
	const idx_t max_memory = buffer_manager.GetMaxMemory();
	if (used_memory >= max_memory) {
		return 0;
	}
	const double memory_budget = static_cast<double>(max_memory - used_memory);
	const double embedding_size = static_cast<double>(search.index.GetNumDimensions() * sizeof(float));
	const auto &passing_row_ids_per_row_group = search.passing_row_ids_per_row_group;
	auto row_groups_per_window =
	    static_cast<idx_t>(std::count_if(passing_row_ids_per_row_group.begin(), passing_row_ids_per_row_group.end(),
	                                     [](const std::vector<size_t> &row_ids) { return !row_ids.empty(); }));
	if (search.max_row_groups_per_on_the_fly_index > 0) {
		row_groups_per_window = MinValue<idx_t>(row_groups_per_window, search.max_row_groups_per_on_the_fly_index);
	}
	// Halving: each check reads every row group.
	while (row_groups_per_window > 0) {
		const auto max_rows_per_window =
		    static_cast<double>(CountMaxPassingRowsPerOtfIndex(passing_row_ids_per_row_group, row_groups_per_window));
		const double peak_memory = (static_cast<double>(num_passing_rows) +
		                            PDXearchIndexJoinSearch::OTF_INDEX_BUILD_PEAK_MEMORY_FACTOR * max_rows_per_window) *
		                           embedding_size;
		if (peak_memory <= memory_budget) {
			return row_groups_per_window;
		}
		row_groups_per_window = row_groups_per_window == 1 ? 0 : (row_groups_per_window + 1) / 2;
	}
	return 0;
}

SinkFinalizeType PhysicalPDXearchIndexJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                     OperatorSinkFinalizeInput &input) const {
	auto &search = input.global_state.Cast<PDXearchIndexJoinGlobalSinkState>().search;
	const auto &passing_row_ids_per_row_group = search.passing_row_ids_per_row_group;
	const bool any_row_passes = std::any_of(passing_row_ids_per_row_group.begin(), passing_row_ids_per_row_group.end(),
	                                        [](const std::vector<size_t> &row_ids) { return !row_ids.empty(); });
	search.shared_predicate_evaluators.resize(passing_row_ids_per_row_group.size());
	// Without rows no query has neighbours.
	if (!any_row_passes) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	// Few passing rows are searched faster exhaustively than through the clusters of their row groups. The cap bounds
	// the Flat index's memory and per-query scan, which the ratio alone lets grow with the number of row groups.
	idx_t num_passing_rows = 0;
	idx_t num_clusters_of_row_groups_with_passing_rows = 0;
	for (idx_t row_group_idx = 0; row_group_idx < passing_row_ids_per_row_group.size(); row_group_idx++) {
		if (!passing_row_ids_per_row_group[row_group_idx].empty()) {
			num_passing_rows += passing_row_ids_per_row_group[row_group_idx].size();
			num_clusters_of_row_groups_with_passing_rows +=
			    search.index.GetRowGroupIndex(row_group_idx).GetNumClusters();
		}
	}
	if (num_passing_rows <= search.max_passing_rows_for_flat_search &&
	    static_cast<double>(num_passing_rows) <=
	        search.max_passing_rows_per_cluster_for_flat_search *
	            static_cast<double>(num_clusters_of_row_groups_with_passing_rows)) {
		GatherPassingRowsIntoOtfIndexes(search, passing_row_ids_per_row_group.size(), /*build_flat=*/true,
		                                /*n_threads=*/1);
		return search.passing_rows_otf_indexes.empty() ? SinkFinalizeType::NO_OUTPUT_POSSIBLE : SinkFinalizeType::READY;
	}
	// Many passing rows: indexes over only them, searched instead of the row groups, when their build fits in memory.
	// 'auto' builds them only for enough queries (as estimated) to pay for the build.
	const bool build_otf_indexes =
	    search.on_the_fly_index == OnTheFlyIndexMode::ALWAYS ||
	    (search.on_the_fly_index == OnTheFlyIndexMode::AUTO &&
	     static_cast<double>(children[0].get().estimated_cardinality) >=
	         search.min_queries_per_passing_row_for_on_the_fly_index * static_cast<double>(num_passing_rows));
	if (build_otf_indexes) {
		const idx_t row_groups_per_otf_index =
		    ChooseRowGroupsPerOtfIndexWithinMemoryBudget(context, search, num_passing_rows);
		if (row_groups_per_otf_index > 0) {
			GatherPassingRowsIntoOtfIndexes(search, row_groups_per_otf_index, /*build_flat=*/false,
			                                static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads()));
			return search.passing_rows_otf_indexes.empty() ? SinkFinalizeType::NO_OUTPUT_POSSIBLE
			                                               : SinkFinalizeType::READY;
		}
	}
	// Each row group's filter is the same for every query: built once here.
	for (idx_t row_group_idx = 0; row_group_idx < passing_row_ids_per_row_group.size(); row_group_idx++) {
		if (!passing_row_ids_per_row_group[row_group_idx].empty()) {
			search.shared_predicate_evaluators[row_group_idx] =
			    search.index.GetRowGroupIndex(row_group_idx)
			        .CreateSharedPredicateEvaluator(passing_row_ids_per_row_group[row_group_idx]);
		}
	}
	return SinkFinalizeType::READY;
}

// ------------------------------
// Operator: one search per query row.
// ------------------------------

class PDXearchIndexJoinOperatorState : public OperatorState {
public:
	PDXearchIndexJoinOperatorState(ClientContext &context, const PhysicalPDXearchIndexJoin &op)
	    : num_dimensions(op.index.Cast<PDXearchIndex>().GetNumDimensions()),
	      raw_query(make_uniq_array<float>(num_dimensions)), query(make_uniq_array<float>(num_dimensions)),
	      row_ids(LogicalType::ROW_TYPE) {
		// The operator's last columns are the fetched table columns.
		const vector<LogicalType> fetch_types(op.types.end() - static_cast<std::ptrdiff_t>(op.fetch_column_ids.size()),
		                                      op.types.end());
		fetched.Initialize(Allocator::Get(context), fetch_types);
	}

	const idx_t num_dimensions;
	const unique_ptr<float[]> raw_query;
	const unique_ptr<float[]> query;
	PDX::TopKHeap heap;
	std::vector<unique_ptr<PDX::IIterativeSearch>> search_cursors;
	// Scratch of ProbeNearestClustersAcrossRowGroups, reused by every query.
	std::vector<float> centroid_distances;
	std::vector<uint32_t> clusters_access_order;
	std::vector<std::pair<float, idx_t>> clusters_ranked_across_row_groups;

	// The input chunk being processed: its query column, and the next row to search.
	bool input_started {false};
	UnifiedVectorFormat query_format;
	idx_t next_input_row {0};

	// The row ids of the current query's nearest rows, and those rows as fetched.
	Vector row_ids;
	DataChunk fetched;
	ColumnFetchState fetch_state;
};

unique_ptr<OperatorState> PhysicalPDXearchIndexJoin::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<PDXearchIndexJoinOperatorState>(context.client, *this);
}

// Copies the query vector of an input row into raw_query. False when the query is NULL.
static bool TryReadQuery(const PhysicalPDXearchIndexJoin &op, DataChunk &input, const idx_t row,
                         PDXearchIndexJoinOperatorState &state) {
	auto &query_vector = input.data[op.query_column];
	const auto query_idx = state.query_format.sel->get_index(row);
	if (!state.query_format.validity.RowIsValid(query_idx)) {
		return false;
	}
	const auto num_dimensions = state.num_dimensions;
	if (query_vector.GetType().id() == LogicalTypeId::BLOB) {
		const auto &blob = UnifiedVectorFormat::GetData<string_t>(state.query_format)[query_idx];
		if (BlobDimensionCount(blob.GetSize()) != num_dimensions) {
			// The error of the distance function, which gets the same query for each row we return.
			throw InvalidInputException("BLOB dimension count (%llu) does not match array size (%llu)",
			                            BlobDimensionCount(blob.GetSize()), num_dimensions);
		}
		DecodeBlobToFloatArray(const_data_ptr_cast(blob.GetData()), blob.GetSize(), state.raw_query.get());
		return true;
	}
	auto &query_elements = ArrayVector::GetEntry(query_vector);
	const auto query_data = FlatVector::GetData<float>(query_elements);
	const auto &query_validity = FlatVector::Validity(query_elements);
	const auto offset = query_idx * num_dimensions;
	// TODO: This is a bit inefficient, especially with high dimensionality.
	// Maybe an optimistic memcpy and then a validity check?
	for (idx_t i = 0; i < num_dimensions; i++) {
		// A NULL element is searched as 0: the distance function raises its error for the rows we return.
		state.raw_query[i] = query_validity.RowIsValid(offset + i) ? query_data[offset + i] : 0;
	}
	return true;
}

// Starts a cursor on index (on the shared evaluator when there is one) that queues its clusters nearest to the query
// first, and adds the clusters it queues to the ranking across indexes. The cursor stays in state.search_cursors.
static void BeginSearchAndRankClusters(const PDXearchIndexJoinSearch &search, PDXearchIndexJoinOperatorState &state,
                                       const PDX::IPDXIndex &index, const PDX::PredicateEvaluator *evaluator) {
	const uint32_t num_clusters = index.GetNumClusters();
	auto &distances = state.centroid_distances;
	distances.resize(num_clusters);
	index.GetDistancesToCentroids(state.query.get(), /*is_query_transformed=*/true, distances.data());
	auto &clusters_access_order = state.clusters_access_order;
	clusters_access_order.resize(num_clusters);
	std::iota(clusters_access_order.begin(), clusters_access_order.end(), 0);
	std::sort(clusters_access_order.begin(), clusters_access_order.end(),
	          [&distances](const uint32_t a, const uint32_t b) { return distances[a] < distances[b]; });
	// The cursor copies the order, so the scratch vectors serve the next index.
	auto search_cursor =
	    evaluator
	        ? index.BeginIterativeSearchWithSharedEvaluator(state.query.get(), static_cast<uint32_t>(search.limit),
	                                                        state.heap, *evaluator, /*is_query_transformed=*/true,
	                                                        &clusters_access_order)
	        : index.BeginIterativeSearch(state.query.get(), static_cast<uint32_t>(search.limit), state.heap, nullptr,
	                                     /*is_query_transformed=*/true, &clusters_access_order);
	const idx_t cursor_idx = state.search_cursors.size();
	state.search_cursors.push_back(unique_ptr<PDX::IIterativeSearch>(search_cursor.release()));
	// The clusters the cursor queues: not empty and, when filtered, with passing rows.
	for (uint32_t cluster_id = 0; cluster_id < num_clusters; cluster_id++) {
		if (index.GetClusterSize(cluster_id) > 0 && (!evaluator || evaluator->n_passing_tuples[cluster_id] > 0)) {
			state.clusters_ranked_across_row_groups.emplace_back(distances[cluster_id], cursor_idx);
		}
	}
}

// Probes the ranked clusters nearest to the query first, at most total_n_probe_budget of them, so that the shared heap
// tightens early.
static void ProbeNearestRankedClusters(PDXearchIndexJoinOperatorState &state, const idx_t total_n_probe_budget) {
	auto &clusters_ranked = state.clusters_ranked_across_row_groups;
	const idx_t num_clusters_to_probe = MinValue<idx_t>(total_n_probe_budget, clusters_ranked.size());
	std::partial_sort(clusters_ranked.begin(),
	                  clusters_ranked.begin() + static_cast<std::ptrdiff_t>(num_clusters_to_probe),
	                  clusters_ranked.end());
	// Each cursor queues its clusters nearest first, so its next cluster is the one listed here.
	for (idx_t i = 0; i < num_clusters_to_probe; i++) {
		state.search_cursors[clusters_ranked[i].second]->Next(1);
	}
}

// The first round of a query's search: the n_probe x (row groups searched) clusters nearest to the query across all
// row groups (when filtered, only clusters with passing rows), probed nearest first.
static void ProbeNearestClustersAcrossRowGroups(const PDXearchIndexJoinSearch &search,
                                                PDXearchIndexJoinOperatorState &state) {
	state.clusters_ranked_across_row_groups.clear();
	idx_t num_row_groups_searched = 0;
	for (idx_t row_group_idx = 0; row_group_idx < search.index.GetNumRowGroups(); row_group_idx++) {
		const PDX::PredicateEvaluator *evaluator = nullptr;
		if (search.is_filtered) {
			evaluator = search.shared_predicate_evaluators[row_group_idx].get();
			if (!evaluator) {
				continue;
			}
		}
		BeginSearchAndRankClusters(search, state, search.index.GetRowGroupIndex(row_group_idx), evaluator);
		num_row_groups_searched++;
	}
	ProbeNearestRankedClusters(state, search.clusters_to_probe_on_first_iteration * num_row_groups_searched);
}

// The search of a query over the indexes built on the fly over the passing rows (every row in them passes): their
// passing_rows_otf_indexes_clusters_to_probe clusters nearest to the query across all of them, probed nearest first.
static void ProbeNearestClustersAcrossPassingRowsOtfIndexes(const PDXearchIndexJoinSearch &search,
                                                            PDXearchIndexJoinOperatorState &state) {
	state.clusters_ranked_across_row_groups.clear();
	for (const auto &otf_index : search.passing_rows_otf_indexes) {
		BeginSearchAndRankClusters(search, state, *otf_index, nullptr);
	}
	ProbeNearestRankedClusters(state, search.passing_rows_otf_indexes_clusters_to_probe);
}

// The first round with each row group searched alone: its own n_probe nearest clusters (when filtered, with passing
// rows). Used when pdxearch_rank_clusters_across_row_groups is false.
static void ProbeNearestClustersPerRowGroup(const PDXearchIndexJoinSearch &search,
                                            PDXearchIndexJoinOperatorState &state) {
	for (idx_t row_group_idx = 0; row_group_idx < search.index.GetNumRowGroups(); row_group_idx++) {
		unique_ptr<PDX::IIterativeSearch> search_cursor;
		if (!search.is_filtered) {
			search_cursor = search.index.BeginSearchForRowGroup(row_group_idx, state.query.get(), search.limit,
			                                                    state.heap, nullptr);
		} else {
			const auto *evaluator = search.shared_predicate_evaluators[row_group_idx].get();
			if (!evaluator) {
				continue;
			}
			search_cursor = unique_ptr<PDX::IIterativeSearch>(
			    search.index.GetRowGroupIndex(row_group_idx)
			        .BeginIterativeSearchWithSharedEvaluator(state.query.get(), static_cast<uint32_t>(search.limit),
			                                                 state.heap, *evaluator, /*is_query_transformed=*/true)
			        .release());
		}
		search_cursor->Next(search.clusters_to_probe_on_first_iteration);
		state.search_cursors.push_back(std::move(search_cursor));
	}
}

// Searches the index for the query in raw_query, and writes the row ids of its nearest rows (nearest first) to
// row_ids. Returns their number, at most K.
static idx_t SearchQuery(const PDXearchIndexJoinSearch &search, PDXearchIndexJoinOperatorState &state) {
	// Normalizes raw_query in place for the cosine distance: our copy, not the input the operators above read.
	search.preprocessor.PreprocessEmbedding(state.raw_query.get(), state.query.get(), search.index.IsNormalized());

	state.heap.heap = PDX::Heap();
	state.search_cursors.clear();
	if (!search.passing_rows_otf_indexes.empty()) {
		ProbeNearestClustersAcrossPassingRowsOtfIndexes(search, state);
	} else if (search.rank_clusters_across_row_groups) {
		ProbeNearestClustersAcrossRowGroups(search, state);
	} else {
		ProbeNearestClustersPerRowGroup(search, state);
	}
	if (search.is_filtered) {
		// Then more clusters until the heap holds K rows or no cluster with passing rows is left.
		while (state.heap.heap.size() < search.limit) {
			bool probed = false;
			for (auto &search_cursor : state.search_cursors) {
				if (!search_cursor->Done()) {
					search_cursor->Next(PDXearchIndexJoinSearch::CLUSTERS_TO_PROBE_PER_FOLLOW_UP_ITERATION);
					probed = true;
				}
			}
			if (!probed) {
				break;
			}
		}
	}

	const auto nearest = PDX::BuildResultSetFromHeap(search.limit, state.heap.heap);
	const auto row_ids = FlatVector::GetData<row_t>(state.row_ids);
	for (size_t i = 0; i < nearest.size(); i++) {
		row_ids[i] = search.passing_rows_otf_indexes.empty()
		                 ? nearest[i].index
		                 : search.passing_rows_otf_indexes_row_ids[nearest[i].index];
	}
	return nearest.size();
}

// Called once per input chunk (up to 2048 query vectors)
OperatorResultType PhysicalPDXearchIndexJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                      GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<PDXearchIndexJoinOperatorState>();
	const auto &search = IsSink() ? sink_state->Cast<PDXearchIndexJoinGlobalSinkState>().search
	                              : *gstate.Cast<PDXearchIndexJoinGlobalOperatorState>().search;
	if (!search.is_filtered && search.index.GetNumRowGroups() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	if (!state.input_started) {
		input.data[query_column].ToUnifiedFormat(input.size(), state.query_format);
		state.next_input_row = 0;
		state.input_started = true;
	}

	// One query per call: the nearest rows of the next input row's query (at most K, which fits in one chunk), with the
	// input row's columns. Fetch skips the rows this transaction cannot see, so a query can have
	// fewer rows; when none are left, the next input row is searched.
	// TODO: Search the queries of a chunk together once PDX has a bulk search.
	auto &transaction = DuckTransaction::Get(context.client, table.catalog);
	while (state.next_input_row < input.size()) {
		const auto row = state.next_input_row++;
		if (!TryReadQuery(*this, input, row, state)) {
			continue;
		}
		const auto num_nearest = SearchQuery(search, state);
		if (num_nearest == 0) {
			continue;
		}
		state.fetched.Reset();
		table.GetStorage().Fetch(transaction, state.fetched, fetch_column_ids, state.row_ids, num_nearest,
		                         state.fetch_state);
		if (state.fetched.size() == 0) {
			continue;
		}

		// The input row's columns, then the fetched table columns.
		const auto num_query_columns = input.ColumnCount();
		for (idx_t col = 0; col < num_query_columns; col++) {
			ConstantVector::Reference(chunk.data[col], input.data[col], row, input.size());
		}
		for (idx_t col = 0; col < state.fetched.ColumnCount(); col++) {
			chunk.data[num_query_columns + col].Reference(state.fetched.data[col]);
		}
		chunk.SetCardinality(state.fetched.size());
		break;
	}

	if (state.next_input_row < input.size()) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	state.input_started = false;
	return OperatorResultType::NEED_MORE_INPUT;
}

// ------------------------------
// Pipeline construction
// ------------------------------

void PhysicalPDXearchIndexJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	if (!IsSink()) {
		// Not filtered: an operator in the pipeline of the query rows.
		PhysicalOperator::BuildPipelines(current, meta_pipeline);
		return;
	}
	// Filtered: built like a join (PhysicalJoin::BuildJoinPipelines, which casts to PhysicalJoin). The query rows'
	// pipeline is the probe side, and a child pipeline sinks the rowids (the build side), which runs first.
	op_state.reset();
	sink_state.reset();

	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, *this);

	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this, MetaPipelineType::JOIN_BUILD);
	child_meta_pipeline.Build(children[1].get());

	children[0].get().BuildPipelines(current, meta_pipeline);
}

vector<const_reference<PhysicalOperator>> PhysicalPDXearchIndexJoin::GetSources() const {
	return children[0].get().GetSources();
}

// Defines this operator's details shown in the query plan.
InsertionOrderPreservingMap<string> PhysicalPDXearchIndexJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Table"] = table.name;
	result["PDXearch Index"] = index.GetIndexName();
	result["Projections"] = ColumnNamesToString(table, column_ids);
	result["K"] = StringUtil::Format("%llu", limit);
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb

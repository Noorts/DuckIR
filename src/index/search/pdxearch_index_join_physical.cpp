#include "index/search/pdxearch_index_join_physical.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
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
		if (context.TryGetCurrentSetting("pdxearch_max_passing_rows_per_cluster_for_flat_search",
		                                 max_passing_rows_per_cluster_for_flat_search_setting) &&
		    !max_passing_rows_per_cluster_for_flat_search_setting.IsNull()) {
			max_passing_rows_per_cluster_for_flat_search =
			    max_passing_rows_per_cluster_for_flat_search_setting.GetValue<double>();
		}
		Value max_passing_rows_for_flat_search_setting;
		if (context.TryGetCurrentSetting("pdxearch_max_passing_rows_for_flat_search",
		                                 max_passing_rows_for_flat_search_setting) &&
		    !max_passing_rows_for_flat_search_setting.IsNull()) {
			max_passing_rows_for_flat_search = max_passing_rows_for_flat_search_setting.GetValue<uint64_t>();
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

	bool is_filtered {false};
	vector<std::vector<size_t>> passing_row_ids_per_row_group;
	vector<std::unique_ptr<PDX::PredicateEvaluator>> shared_predicate_evaluators;
	std::unique_ptr<PDX::ADSamplingPruner> passing_rows_flat_index_pruner;
	std::unique_ptr<PDX::FlatIndex> passing_rows_flat_index;
	vector<row_t> passing_rows_flat_row_ids;
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

// Gathers the embeddings of the passing rows from their row groups' indexes into one Flat index, which every query
// then searches exhaustively instead of the row groups. Its rows are numbered by position in passing_rows_flat_row_ids.
static void GatherPassingRowsIntoFlatIndex(PDXearchIndexJoinSearch &search) {
	const idx_t num_dimensions = search.index.GetNumDimensions();
	auto &passing_row_ids_per_row_group = search.passing_row_ids_per_row_group;
	// Rows without an embedding are not in the index.
	idx_t num_rows = 0;
	for (idx_t row_group_idx = 0; row_group_idx < passing_row_ids_per_row_group.size(); row_group_idx++) {
		const auto &row_group_index = search.index.GetRowGroupIndex(row_group_idx);
		auto &row_ids = passing_row_ids_per_row_group[row_group_idx];
		row_ids.erase(std::remove_if(row_ids.begin(), row_ids.end(),
		                             [&](const size_t row_id) { return !row_group_index.Contains(row_id); }),
		              row_ids.end());
		num_rows += row_ids.size();
	}
	if (num_rows == 0) {
		return;
	}
	auto embeddings = make_uniq_array_uninitialized<float>(num_rows * num_dimensions);
	auto &flat_row_ids = search.passing_rows_flat_row_ids;
	flat_row_ids.reserve(num_rows);
	for (idx_t row_group_idx = 0; row_group_idx < passing_row_ids_per_row_group.size(); row_group_idx++) {
		const auto &row_ids = passing_row_ids_per_row_group[row_group_idx];
		search.index.GetRowGroupIndex(row_group_idx)
		    .GetEmbeddingsFromIndexByRowIds(row_ids, embeddings.get() + flat_row_ids.size() * num_dimensions);
		flat_row_ids.insert(flat_row_ids.end(), row_ids.begin(), row_ids.end());
	}
	PDX::PDXIndexConfig config;
	config.num_dimensions = static_cast<uint32_t>(num_dimensions);
	config.distance_metric = PDX::DistanceMetric::L2SQ;
	config.is_data_transformed = true;
	// FlatIndex sets PDX's process-wide thread count from its config.
	config.n_threads = 1;
	search.passing_rows_flat_index_pruner =
	    std::make_unique<PDX::ADSamplingPruner>(num_dimensions, search.index.GetRotationMatrix());
	search.passing_rows_flat_index = std::make_unique<PDX::FlatIndex>(config, *search.passing_rows_flat_index_pruner);
	std::vector<size_t> positions(num_rows);
	std::iota(positions.begin(), positions.end(), 0);
	search.passing_rows_flat_index->BuildIndex(positions.data(), embeddings.get(), num_rows);
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
		GatherPassingRowsIntoFlatIndex(search);
		return search.passing_rows_flat_index ? SinkFinalizeType::READY : SinkFinalizeType::NO_OUTPUT_POSSIBLE;
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

// The first round of a query's search: the n_probe x (row groups searched) clusters nearest to the query across all
// row groups (when filtered, only clusters with passing rows), probed nearest first so that the shared heap tightens
// early. Each row group's cursor starts from its own ranking and stays in state.search_cursors.
static void ProbeNearestClustersAcrossRowGroups(const PDXearchIndexJoinSearch &search,
                                                PDXearchIndexJoinOperatorState &state) {
	auto &clusters_ranked = state.clusters_ranked_across_row_groups;
	clusters_ranked.clear();
	idx_t num_row_groups_searched = 0;
	for (idx_t row_group_idx = 0; row_group_idx < search.index.GetNumRowGroups(); row_group_idx++) {
		const PDX::PredicateEvaluator *evaluator = nullptr;
		if (search.is_filtered) {
			evaluator = search.shared_predicate_evaluators[row_group_idx].get();
			if (!evaluator) {
				continue;
			}
		}
		const auto &row_group_index = search.index.GetRowGroupIndex(row_group_idx);
		const uint32_t num_clusters = row_group_index.GetNumClusters();
		auto &distances = state.centroid_distances;
		distances.resize(num_clusters);
		row_group_index.GetDistancesToCentroids(state.query.get(), /*is_query_transformed=*/true, distances.data());
		auto &clusters_access_order = state.clusters_access_order;
		clusters_access_order.resize(num_clusters);
		std::iota(clusters_access_order.begin(), clusters_access_order.end(), 0);
		std::sort(clusters_access_order.begin(), clusters_access_order.end(),
		          [&distances](const uint32_t a, const uint32_t b) { return distances[a] < distances[b]; });
		// The cursor copies the order, so the scratch vectors serve the next row group.
		auto search_cursor =
		    evaluator ? row_group_index.BeginIterativeSearchWithSharedEvaluator(
		                    state.query.get(), static_cast<uint32_t>(search.limit), state.heap, *evaluator,
		                    /*is_query_transformed=*/true, &clusters_access_order)
		              : row_group_index.BeginIterativeSearch(state.query.get(), static_cast<uint32_t>(search.limit),
		                                                     state.heap, nullptr, /*is_query_transformed=*/true,
		                                                     &clusters_access_order);
		const idx_t cursor_idx = state.search_cursors.size();
		state.search_cursors.push_back(unique_ptr<PDX::IIterativeSearch>(search_cursor.release()));
		// The clusters the cursor queues: not empty and, when filtered, with passing rows.
		for (uint32_t cluster_id = 0; cluster_id < num_clusters; cluster_id++) {
			if (row_group_index.GetClusterSize(cluster_id) > 0 &&
			    (!evaluator || evaluator->n_passing_tuples[cluster_id] > 0)) {
				clusters_ranked.emplace_back(distances[cluster_id], cursor_idx);
			}
		}
		num_row_groups_searched++;
	}
	const idx_t total_n_probe_budget =
	    MinValue<idx_t>(search.clusters_to_probe_on_first_iteration * num_row_groups_searched, clusters_ranked.size());
	std::partial_sort(clusters_ranked.begin(),
	                  clusters_ranked.begin() + static_cast<std::ptrdiff_t>(total_n_probe_budget),
	                  clusters_ranked.end());
	// Each cursor queues its clusters nearest first, so its next cluster is the one listed here.
	for (idx_t i = 0; i < total_n_probe_budget; i++) {
		state.search_cursors[clusters_ranked[i].second]->Next(1);
	}
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
	if (search.passing_rows_flat_index) {
		// Its one cluster holds every passing row.
		search.passing_rows_flat_index
		    ->BeginIterativeSearch(state.query.get(), static_cast<uint32_t>(search.limit), state.heap, nullptr,
		                           /*is_query_transformed=*/true)
		    ->Next(1);
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
		row_ids[i] =
		    search.passing_rows_flat_index ? search.passing_rows_flat_row_ids[nearest[i].index] : nearest[i].index;
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

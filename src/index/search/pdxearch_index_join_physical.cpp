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

	bool is_filtered {false};
	vector<std::vector<size_t>> passing_row_ids; // per row group
};

// Holds the search state when the table is filtered (the sink runs first).
class PDXearchIndexJoinGlobalSinkState : public GlobalSinkState {
public:
	PDXearchIndexJoinGlobalSinkState(ClientContext &context, const PhysicalPDXearchIndexJoin &op)
	    : search(context, op) {
		search.is_filtered = true;
		search.passing_row_ids.resize(search.index.GetNumRowGroups());
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
	explicit PDXearchIndexJoinLocalSinkState(const PDXearchIndex &index) : passing_row_ids(index.GetNumRowGroups()) {
	}
	idx_t current_row_group_id {0};
	PDXearchRowRange current_row_group_range {0, 0};
	vector<std::vector<size_t>> passing_row_ids;
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
		l_sink.passing_row_ids[l_sink.current_row_group_id].push_back(static_cast<size_t>(row_id));
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalPDXearchIndexJoin::Combine(ExecutionContext &context,
                                                         OperatorSinkCombineInput &input) const {
	auto &g_sink = input.global_state.Cast<PDXearchIndexJoinGlobalSinkState>();
	auto &l_sink = input.local_state.Cast<PDXearchIndexJoinLocalSinkState>();

	const auto guard = g_sink.Lock();
	for (idx_t row_group_idx = 0; row_group_idx < l_sink.passing_row_ids.size(); row_group_idx++) {
		auto &local_row_ids = l_sink.passing_row_ids[row_group_idx];
		auto &global_row_ids = g_sink.search.passing_row_ids[row_group_idx];
		global_row_ids.insert(global_row_ids.end(), local_row_ids.begin(), local_row_ids.end());
	}
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalPDXearchIndexJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                     OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<PDXearchIndexJoinGlobalSinkState>();
	const auto &passing_row_ids = g_sink.search.passing_row_ids;
	const bool any_row_passes = std::any_of(passing_row_ids.begin(), passing_row_ids.end(),
	                                        [](const std::vector<size_t> &row_ids) { return !row_ids.empty(); });
	// Without rows no query has neighbours, and DuckDB skips the pipeline of the query rows.
	return any_row_passes ? SinkFinalizeType::READY : SinkFinalizeType::NO_OUTPUT_POSSIBLE;
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

// Searches the index for the query in raw_query, and writes the row ids of its nearest rows (nearest first) to
// row_ids. Returns their number, at most K.
static idx_t SearchQuery(const PDXearchIndexJoinSearch &search, PDXearchIndexJoinOperatorState &state) {
	// Normalizes raw_query in place for the cosine distance: our copy, not the input the operators above read.
	search.preprocessor.PreprocessEmbedding(state.raw_query.get(), state.query.get(), search.index.IsNormalized());

	state.heap.heap = PDX::Heap();
	state.search_cursors.clear();
	if (!search.is_filtered) {
		// As in the index scan: the n_probe nearest clusters of every row group.
		for (idx_t row_group_idx = 0; row_group_idx < search.index.GetNumRowGroups(); row_group_idx++) {
			auto search_cursor = search.index.BeginSearchForRowGroup(row_group_idx, state.query.get(), search.limit,
			                                                         state.heap, nullptr);
			search_cursor->Next(search.clusters_to_probe_on_first_iteration);
		}
	} else {
		// As in the filtered scan: the n_probe nearest clusters with passing rows of every row group that has passing
		// rows, then more clusters until the heap holds K rows or no cluster with passing rows is left.
		for (idx_t row_group_idx = 0; row_group_idx < search.passing_row_ids.size(); row_group_idx++) {
			if (search.passing_row_ids[row_group_idx].empty()) {
				continue;
			}
			auto search_cursor = search.index.BeginSearchForRowGroup(
			    row_group_idx, state.query.get(), search.limit, state.heap, &search.passing_row_ids[row_group_idx]);
			search_cursor->Next(search.clusters_to_probe_on_first_iteration);
			state.search_cursors.push_back(std::move(search_cursor));
		}
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
		row_ids[i] = nearest[i].index;
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

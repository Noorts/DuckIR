#include "index/search/pdxearch_index_scan_physical.hpp"
#include "duckdb/parallel/event.hpp"
#include "index/pdxearch_index.hpp"
#include "index/pdxearch_index_utils.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/execution/executor.hpp"

#include "pdx/ivf_searcher.hpp"

namespace duckdb {

PhysicalPDXearchIndexScan::PhysicalPDXearchIndexScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                     unique_ptr<PDXearchIndexScanBindData> bind_data,
                                                     vector<ColumnIndex> column_ids, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalPDXearchIndexScan::TYPE, std::move(types), estimated_cardinality),
      bind_data(std::move(bind_data)), column_ids(std::move(column_ids)) {
}

class PDXearchScanGlobalSourceState : public GlobalSourceState {
public:
	PDXearchScanGlobalSourceState(ClientContext &context, const PhysicalPDXearchIndexScan &op,
	                              const PDXearchIndexScanBindData &bind_data,
	                              const vector<ColumnIndex> &operator_column_ids)
	    : search_lock(bind_data.index.Cast<PDXearchIndex>().SyncAndLockForSearch(bind_data.table.GetStorage())),
	      context(context), op(op), limit(bind_data.limit), index(bind_data.index.Cast<PDXearchIndex>()),
	      preprocessed_query_embedding(make_uniq_array<float>(index.GetNumDimensions())), search_started(false),
	      search_completed(false), pdxearch_row_ids(nullptr), pdxearch_row_ids_idx(0) {
		// Preprocess the query embedding.
		const EmbeddingPreprocessor embedding_preprocessor =
		    EmbeddingPreprocessor(index.GetNumDimensions(), index.GetRotationMatrix());
		embedding_preprocessor.PreprocessEmbedding(bind_data.query_embedding.get(), preprocessed_query_embedding.get(),
		                                           index.IsNormalized());

		// Determine number of clusters to probe per row group. The cluster count varies per row group, but a single
		// upper bound is used here. We use the count for a full row group: for full row groups it matches exactly, and
		// for smaller row groups the per-row-group searcher clamps internally.
		const auto n_probe = index.GetEffectiveNProbe(context);
		const auto num_clusters_for_full_row_group = PDXearchIndex::GetNumClustersForFullRowGroup();
		num_clusters_to_probe_per_row_group =
		    (n_probe == 0 || n_probe > num_clusters_for_full_row_group) ? num_clusters_for_full_row_group : n_probe;

		// Set up column IDs for fetching data from storage.
		column_ids.reserve(operator_column_ids.size());
		for (auto &id : operator_column_ids) {
			StorageIndex storage_id;
			if (id.IsRowIdColumn()) {
				storage_id = StorageIndex();
			} else {
				auto &col = bind_data.table.GetColumn(LogicalIndex(id.GetPrimaryIndex()));
				storage_id = StorageIndex(col.StorageOid());
			}
			column_ids.emplace_back(storage_id);
		}

		// Initialize the storage scan state.
		local_storage_state.Initialize(column_ids, context, nullptr);
		auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
		local_storage.InitializeScan(bind_data.table.GetStorage(), local_storage_state.local_state, nullptr);
	}

	// Held for the duration of execution to serialize searches against index
	// maintenance. Declared first so it is constructed first and destroyed
	// last, ensuring the lock is held while all other members (which reference
	// index state) are torn down.
	unique_ptr<StorageLockKey> search_lock;

	ClientContext &context;
	const PhysicalPDXearchIndexScan &op;
	// The limit (the K in KNN search).
	const idx_t limit;
	PDXearchIndex &index;

	const unique_ptr<float[]> preprocessed_query_embedding;

	// Top-k heap shared by the searches of all row groups (and threads).
	PDX::TopKHeap top_k_heap {/*thread_safe=*/true};

	idx_t num_clusters_to_probe_per_row_group {0};

	// For Source blocking coordination.
	std::atomic<bool> search_started;
	std::atomic<bool> search_completed;

	// Row ids of the final result of the search. For these rows the projected columns are fetched from local storage
	// and emitted through the Source interface.
	std::unique_ptr<std::vector<row_t>> pdxearch_row_ids;
	//! Current position in pdxearch_rowids when emitting fetched rows. Used in case the operator has to emit multiple
	//! chunks of results.
	idx_t pdxearch_row_ids_idx;

	// For fetching the result rows from storage.
	vector<StorageIndex> column_ids;
	TableScanState local_storage_state;
	ColumnFetchState fetch_state;
};

class PDXearchScanSearchTask : public ExecutorTask {
public:
	PDXearchScanSearchTask(shared_ptr<Event> event_p, ClientContext &context, PDXearchScanGlobalSourceState &g_state_p,
	                       const PhysicalOperator &op_p, idx_t row_group_id_p)
	    : ExecutorTask(context, std::move(event_p), op_p), g_state(g_state_p), row_group_id(row_group_id_p) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		auto &index = g_state.index;

		auto search_cursor = index.BeginSearchForRowGroup(row_group_id, g_state.preprocessed_query_embedding.get(),
		                                                  g_state.limit, g_state.top_k_heap, nullptr);
		search_cursor->Next(g_state.num_clusters_to_probe_per_row_group);

		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

	string TaskType() const override {
		return "PDXearchScanSearchTask";
	}

private:
	PDXearchScanGlobalSourceState &g_state;
	idx_t row_group_id;
};

class PDXearchScanSearchEvent : public Event {
public:
	PDXearchScanSearchEvent(Executor &executor_p, PDXearchScanGlobalSourceState &g_state_p)
	    : Event(executor_p), g_state(g_state_p) {
	}

	PDXearchScanGlobalSourceState &g_state;

public:
	void Schedule() override {
		auto &context = GetClientContext();
		vector<shared_ptr<Task>> tasks;
		for (idx_t row_group_id = 0; row_group_id < g_state.index.GetNumRowGroups(); row_group_id++) {
			tasks.push_back(
			    make_uniq<PDXearchScanSearchTask>(shared_from_this(), context, g_state, g_state.op, row_group_id));
		}
		if (tasks.empty()) {
			// An index without row groups
			Finish();
			return;
		}
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		// Store PDXearch result into the source state.
		const auto result_rowids = PDX::BuildResultSetFromHeap(g_state.limit, g_state.top_k_heap.heap);
		g_state.pdxearch_row_ids = make_uniq<std::vector<row_t>>(result_rowids.size());
		for (size_t i = 0; i < result_rowids.size(); i++) {
			(*g_state.pdxearch_row_ids)[i] = result_rowids[i].index;
		}

		g_state.search_completed = true;

		const auto guard = g_state.Lock();
		g_state.UnblockTasks(guard);
	}
};

class PDXearchScanLocalSourceState : public LocalSourceState {};

unique_ptr<GlobalSourceState> PhysicalPDXearchIndexScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PDXearchScanGlobalSourceState>(context, *this, *bind_data, column_ids);
}

unique_ptr<LocalSourceState> PhysicalPDXearchIndexScan::GetLocalSourceState(ExecutionContext &context,
                                                                            GlobalSourceState &g_state) const {
	return make_uniq<PDXearchScanLocalSourceState>();
}

SourceResultType PhysicalPDXearchIndexScan::GetDataInternal(ExecutionContext &context, DataChunk &output_chunk,
                                                            OperatorSourceInput &input) const {
	auto &g_state = input.global_state.Cast<PDXearchScanGlobalSourceState>();

	// Trigger parallel search of all row groups.
	if (!g_state.search_started.exchange(true)) {
		auto &executor = Executor::Get(context.client);
		auto event = make_shared_ptr<PDXearchScanSearchEvent>(executor, g_state);
		executor.AddEvent(event);
		event->Schedule();
	}

	// Block until the search is done.
	const auto guard = g_state.Lock();
	if (!g_state.search_completed) {
		return g_state.BlockSource(guard, input.interrupt_state);
	}

	// Search is done, now fetch and emit results. Note that if K > STANDARD_VECTOR_SIZE, then GetData will be called
	// multiple times and multiple chunks of results will be emitted.
	D_ASSERT(g_state.pdxearch_row_ids);
	D_ASSERT(g_state.pdxearch_row_ids_idx <= g_state.pdxearch_row_ids->size());

	const idx_t num_results_to_emit =
	    MinValue<idx_t>(STANDARD_VECTOR_SIZE, g_state.pdxearch_row_ids->size() - g_state.pdxearch_row_ids_idx);

	if (num_results_to_emit == 0) {
		return SourceResultType::FINISHED;
	}

	// Create vector of row ids that are part of the current output chunk.
	Vector row_ids_vector(LogicalType::ROW_TYPE, num_results_to_emit);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vector);
	for (idx_t i = 0; i < num_results_to_emit; i++) {
		row_ids_data[i] = g_state.pdxearch_row_ids->at(g_state.pdxearch_row_ids_idx + i);
	}
	g_state.pdxearch_row_ids_idx += num_results_to_emit;

	// Fetch the data (visible to the transaction) from storage.
	auto &transaction = DuckTransaction::Get(context.client, bind_data->table.catalog);
	bind_data->table.GetStorage().Fetch(transaction, output_chunk, g_state.column_ids, row_ids_vector,
	                                    num_results_to_emit, g_state.fetch_state);

	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalPDXearchIndexScan::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Table"] = bind_data->table.name;
	result["PDXearch Index"] = bind_data->index.GetIndexName();
	result["Projections"] = ColumnNamesToString(bind_data->table, column_ids);
	result["Total Clusters"] = StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetTotalNumClusters());
	result["Row Groups"] = StringUtil::Format("%zu", bind_data->index.Cast<PDXearchIndex>().GetNumRowGroups());
	const idx_t index_in_memory_size = bind_data->index.Cast<BoundIndex>().GetInMemorySize();
	result["Index Size"] = ConvertBytesToHumanReadableString(index_in_memory_size);
	SetEstimatedCardinality(result, estimated_cardinality);

	return result;
}

} // namespace duckdb

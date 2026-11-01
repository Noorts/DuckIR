#include "index/create/vss_index_create.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"

#include "index/vss_index.hpp"
#include "index/vss_parallel_executor.hpp"

namespace duckdb {

PhysicalCreateVSSIndex::PhysicalCreateVSSIndex(PhysicalPlan &physical_plan, const vector<LogicalType> &types_p,
                                               TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                               unique_ptr<CreateIndexInfo> info,
                                               vector<unique_ptr<Expression>> unbound_expressions,
                                               idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types_p, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {
	for (auto &virtual_column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(virtual_column_id)).index);
	}
}

// Row groups are built as concurrently as DuckDB's memory manager grants memory for, and at least one at a time. A
// thread that starts a row group takes one of `max_concurrent_builds` slots, or waits until another thread finishes
// building and gives its slot back.
class CreateVSSIndexGlobalSinkState : public GlobalSinkState {
public:
	CreateVSSIndexGlobalSinkState(const PhysicalCreateVSSIndex &op, ClientContext &context)
	    : global_index(make_uniq<VSSIndex>(op.info->GetIndexName(), op.info->constraint_type, op.storage_ids,
	                                       TableIOManager::Get(op.table.GetStorage()), op.unbound_expressions,
	                                       op.table.GetStorage().db, op.info->options, IndexStorageInfo())),
	      num_dimensions(ArrayType::GetSize(op.unbound_expressions[0]->GetReturnType())) {
		global_index->Cast<VSSIndex>().FitSAQModelForCreate(context, op.table.GetStorage());
		const auto &index = global_index->Cast<VSSIndex>();
		const idx_t row_group_size = index.GetRowGroupSize();
		const idx_t build_bytes =
		    row_group_size * num_dimensions * sizeof(float) + index.EstimateBuildPeakBytes(row_group_size);
		const auto num_threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
		// SuperKMeans sizes its k-means scratch from an executor's NumWorkers(). This executor runs nothing: it only
		// asks for the scratch of all the threads, which the concurrent builds split between them.
		const DuckDBParallelExecutor all_threads(context, num_threads);
		const idx_t kmeans_tiles_bytes =
		    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::ScratchSize(
		        all_threads) *
		    sizeof(float);
		// Note that for a plain table scan, op.estimated_cardinality is exact.
		const idx_t num_row_groups =
		    MaxValue<idx_t>(1, (op.estimated_cardinality + row_group_size - 1) / row_group_size);
		memory_state = TemporaryMemoryManager::Get(context).Register(context);
		memory_state->SetMinimumReservation(build_bytes + kmeans_tiles_bytes);
		memory_state->SetRemainingSizeAndUpdateReservation(
		    context, MinValue(num_threads, num_row_groups) * build_bytes + kmeans_tiles_bytes);
		max_concurrent_builds = MaxValue<idx_t>(1, (memory_state->GetReservation() - kmeans_tiles_bytes) / build_bytes);
		default_threads_per_build = MaxValue<idx_t>(1, num_threads / max_concurrent_builds);
	}

	unique_ptr<BoundIndex> global_index;
	const idx_t num_dimensions;
	unique_ptr<TemporaryMemoryState> memory_state;
	idx_t max_concurrent_builds = 1;
	// Under Lock().
	idx_t running_builds = 0;
	idx_t default_threads_per_build = 1;
};

unique_ptr<GlobalSinkState> PhysicalCreateVSSIndex::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<CreateVSSIndexGlobalSinkState>(*this, context);
}

class CreateVSSIndexLocalSinkState : public LocalSinkState {
public:
	explicit CreateVSSIndexLocalSinkState(const PhysicalCreateVSSIndex &op, ClientContext &context,
	                                      CreateVSSIndexGlobalSinkState &g_sink) {
		const auto row_group_size = g_sink.global_index->Cast<VSSIndex>().GetRowGroupSize();
		row_group_row_ids.resize(row_group_size);
	}

	// The DuckDB row group whose rows are currently buffered.
	VSSRowGroupBounds row_group {0, 0};
	bool row_group_is_closed {false};
	bool has_row_group {false};
	// Number of embeddings currently buffered in the row group.
	idx_t row_group_embeddings_count {0};
	BufferHandle row_group_embeddings;
	// Row IDs of the embeddings currently buffered in the row group.
	std::vector<row_t> row_group_row_ids;
};

// Builds the buffered row group, then frees its buffer and gives its build slot to the threads waiting for one.
static void FlushRowGroup(CreateVSSIndexGlobalSinkState &g_sink, CreateVSSIndexLocalSinkState &l_sink,
                          ClientContext &context) {
	if (!l_sink.has_row_group) {
		return;
	}
	if (l_sink.row_group_embeddings_count > 0) {
		DuckDBParallelExecutor executor(context, g_sink.default_threads_per_build);
		g_sink.global_index->Cast<VSSIndex>().SetUpIndexForRowGroup(
		    l_sink.row_group_row_ids.data(), reinterpret_cast<float *>(l_sink.row_group_embeddings.GetDataMutable()),
		    l_sink.row_group_embeddings_count, l_sink.row_group.row_start, l_sink.row_group.count,
		    l_sink.row_group_is_closed, &executor);
		l_sink.row_group_embeddings_count = 0;
	}
	l_sink.row_group_embeddings.Destroy();
	l_sink.has_row_group = false;
	annotated_lock_guard<annotated_mutex> guard(g_sink.lock);
	g_sink.running_builds--;
	g_sink.UnblockTasks();
}

unique_ptr<LocalSinkState> PhysicalCreateVSSIndex::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<CreateVSSIndexLocalSinkState>(*this, context.client,
	                                               sink_state->Cast<CreateVSSIndexGlobalSinkState>());
}

SinkResultType PhysicalCreateVSSIndex::Sink(ExecutionContext &context, DataChunk &input_chunk,
                                            OperatorSinkInput &input) const {
	auto &g_sink = input.global_state.Cast<CreateVSSIndexGlobalSinkState>();
	auto &l_sink = input.local_state.Cast<CreateVSSIndexLocalSinkState>();
	auto &vss_index = g_sink.global_index->Cast<VSSIndex>();

	// Early exit if the chunk is empty.
	if (input_chunk.size() == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	// Validate input chunk structure.
	D_ASSERT(input_chunk.ColumnCount() == 2);
	auto &embedding_column = input_chunk.data[0];
	auto &row_id_column = input_chunk.data[1];
	D_ASSERT(embedding_column.GetType().id() == LogicalTypeId::ARRAY);
	D_ASSERT(ArrayType::GetSize(embedding_column.GetType()) == g_sink.num_dimensions);
	D_ASSERT(row_id_column.GetType() == LogicalType::ROW_TYPE);

	// Chunks arrive with a selection vector when the scan skipped deleted rows or the filter dropped NULL embeddings.
	const idx_t num_embeddings = input_chunk.size();
	embedding_column.Flatten();
	row_id_column.Flatten();
	const auto row_id_data = FlatVector::GetData<row_t>(row_id_column);

	// A chunk never spans two DuckDB row groups, so the first row id tells which one this chunk belongs to.
	VSSRowGroupBounds row_group;
	if (!vss_index.TryGetPhysicalRowGroup(table.GetStorage(), row_id_data[0], row_group)) {
		throw InternalException("DuckIR VSS: row id %lld is not in any row group of the table", row_id_data[0]);
	}
	D_ASSERT(!l_sink.has_row_group || l_sink.row_group.row_start <= row_group.row_start);

	// If we detect a new row group, then finalize the previous row group and prepare to process the new one.
	if (l_sink.has_row_group && row_group.row_start != l_sink.row_group.row_start) {
		FlushRowGroup(g_sink, l_sink, context.client);
	}
	if (!l_sink.has_row_group) {
		// Without a free build slot, DuckDB runs this chunk again once a build gives its slot back.
		{
			annotated_lock_guard<annotated_mutex> guard(g_sink.lock);
			if (g_sink.running_builds == g_sink.max_concurrent_builds) {
				return g_sink.BlockSink(input.interrupt_state);
			}
			g_sink.running_builds++;
		}
		l_sink.row_group_embeddings =
		    BufferManager::GetBufferManager(context.client)
		        .Allocate(MemoryTag::EXTENSION, row_group.count * g_sink.num_dimensions * sizeof(float),
		                  /*can_destroy=*/false);
		l_sink.row_group = row_group;
		l_sink.row_group_is_closed = vss_index.IsClosedRowGroup(table.GetStorage(), row_group);
		l_sink.has_row_group = true;
	}

	// Preprocess and accumulate the embeddings into the row group's buffer.
	D_ASSERT(l_sink.row_group_embeddings_count + num_embeddings <= row_group.count);

	vss_index.PreprocessEmbeddings(FlatVector::GetDataMutable<float>(ArrayVector::GetChildMutable(embedding_column)),
	                               reinterpret_cast<float *>(l_sink.row_group_embeddings.GetDataMutable()) +
	                                   (l_sink.row_group_embeddings_count * g_sink.num_dimensions),
	                               num_embeddings);

	memcpy(l_sink.row_group_row_ids.data() + l_sink.row_group_embeddings_count, row_id_data,
	       num_embeddings * sizeof(row_t));

	l_sink.row_group_embeddings_count += num_embeddings;

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreateVSSIndex::Combine(ExecutionContext &context,
                                                      OperatorSinkCombineInput &input) const {
	auto &l_sink = input.local_state.Cast<CreateVSSIndexLocalSinkState>();
	auto &g_sink = input.global_state.Cast<CreateVSSIndexGlobalSinkState>();

	// Finalize this thread's last row group.
	FlushRowGroup(g_sink, l_sink, context.client);

	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalCreateVSSIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  OperatorSinkFinalizeInput &input) const {
	auto &g_sink = input.global_state.Cast<CreateVSSIndexGlobalSinkState>();

	auto &storage = table.GetStorage();
	if (!storage.IsMainTable()) {
		throw TransactionException(
		    "Transaction conflict: cannot add an index to a table that has been altered or dropped");
	}

	auto &schema = table.schema;
	info->column_ids = storage_ids;

	// Ensure that the index does not yet exist in the catalog.
	auto entry = schema.GetEntry(schema.GetCatalogTransaction(context), CatalogType::INDEX_ENTRY, info->GetIndexName());
	if (entry) {
		if (info->on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
			throw CatalogException("Index with name \"%s\" already exists!", info->GetIndexName());
		}
		// IF NOT EXISTS on existing index. We are done.
		return SinkFinalizeType::READY;
	}

	auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(context), *info, table).get();
	D_ASSERT(index_entry);
	auto &index = index_entry->Cast<DuckIndexEntry>();

	index.initial_index_size = g_sink.global_index->GetInMemorySize();

	// Add the index to the storage.
	storage.AddIndex(std::move(g_sink.global_index), index.oid);
	return SinkFinalizeType::READY;
}

} // namespace duckdb

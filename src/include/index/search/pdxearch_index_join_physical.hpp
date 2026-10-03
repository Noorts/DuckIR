#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/storage/storage_index.hpp"

namespace duckdb {

class Index;
class DuckTableEntry;

// Runs the LogicalPDXearchIndexJoin. When the table is filtered (children[1]), it is a sink first: it collects the
// rowids that pass the predicate, per row group. Then it is an operator over the query rows (children[0]): it searches
// the index for each query, one query after another, fetches the K nearest rows, and emits them with the query row.
class PhysicalPDXearchIndexJoin : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

public:
	PhysicalPDXearchIndexJoin(PhysicalPlan &physical_plan, vector<LogicalType> types, DuckTableEntry &table,
	                          Index &index, idx_t limit, idx_t query_column, const vector<ColumnIndex> &column_ids,
	                          idx_t estimated_cardinality);

	DuckTableEntry &table;
	Index &index;
	// The number of nearest rows per query (`k`).
	// At most STANDARD_VECTOR_SIZE (the rows of a query fit in one output chunk)
	const idx_t limit;
	// The query vector's column in the query rows.
	const idx_t query_column;
	// The table columns fetched by rowid for the results.
	vector<ColumnIndex> column_ids;
	vector<StorageIndex> fetch_column_ids;

public:
	string GetName() const override {
		return "PDXEARCH_INDEX_JOIN";
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// Operator interface: the query rows.
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	bool ParallelOperator() const override {
		return true;
	}

	// Sink interface: the rowids of the table's rows that pass the predicate, when the table is filtered.
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	bool IsSink() const override {
		return children.size() == 2;
	}
	bool ParallelSink() const override {
		return true;
	}

	// Pipeline construction: like a join's, with the rowids as the build side.
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;
};

} // namespace duckdb

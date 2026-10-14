#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/storage/table/scan_state.hpp"

namespace duckdb {

class Index;
class DuckTableEntry;

struct VSSIndexPhysicalScanBindData {
	VSSIndexPhysicalScanBindData(DuckTableEntry &table, Index &index, idx_t limit,
	                             unsafe_unique_array<float> query_embedding)
	    : table(table), index(index), limit(limit), query_embedding(std::move(query_embedding)) {
	}

	DuckTableEntry &table;
	Index &index;
	// The limit (the K in KNN search).
	const idx_t limit;
	unsafe_unique_array<float> query_embedding;
};

class PhysicalVSSIndexFilteredScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

public:
	PhysicalVSSIndexFilteredScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                             unique_ptr<VSSIndexPhysicalScanBindData> bind_data, vector<ColumnIndex> column_ids,
	                             idx_t estimated_cardinality);

	string GetName() const override {
		return "DUCKIR_VSS_INDEX_FILT_SCAN";
	};

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// Sink interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	// Note: PrepareFinalize is not in use.
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}

	// Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return false;
	}
	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::FIXED_ORDER;
	}

private:
	unique_ptr<VSSIndexPhysicalScanBindData> bind_data;

	// IDs of the columns to fetch from local storage for the (at most) top K
	// results. This depends on the column IDs of the original table scan.
	vector<ColumnIndex> column_ids;
};

} // namespace duckdb

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/storage/table/scan_state.hpp"

namespace duckdb {

class Index;
class DuckTableEntry;

struct VSSIndexScanBindData {
	VSSIndexScanBindData(DuckTableEntry &table, Index &index, idx_t limit, unsafe_unique_array<float> query_embedding)
	    : table(table), index(index), limit(limit), query_embedding(std::move(query_embedding)) {
	}

	DuckTableEntry &table;
	Index &index;
	// The limit (the K in KNN search).
	const idx_t limit;
	unsafe_unique_array<float> query_embedding;
};

class PhysicalVSSIndexScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

public:
	PhysicalVSSIndexScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                     unique_ptr<VSSIndexScanBindData> bind_data, vector<ColumnIndex> column_ids,
	                     idx_t estimated_cardinality);

	string GetName() const override {
		return "DUCKIR_VSS_INDEX_SCAN";
	};

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &g_source) const override;
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
	unique_ptr<VSSIndexScanBindData> bind_data;

	// IDs of the columns to fetch from local storage for the (at most) top K
	// results. This depends on the column IDs of the original table scan.
	vector<ColumnIndex> column_ids;
};

} // namespace duckdb

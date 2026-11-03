#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

#include "index/search/vss_search_guard.hpp"

namespace duckdb {

class DuckTableEntry;

class LogicalVSSIndexScan : public LogicalExtensionOperator {
public:
	LogicalVSSIndexScan(DuckTableEntry &table, VSSIndexPlanInfo index_info, idx_t limit,
	                    unsafe_unique_array<float> query_embedding, vector<ColumnIndex> column_ids,
	                    vector<ColumnBinding> column_bindings)
	    : LogicalExtensionOperator(), column_ids(std::move(column_ids)), column_bindings(std::move(column_bindings)),
	      table(table), index_info(std::move(index_info)), limit(limit), query_embedding(std::move(query_embedding)) {
	}

	// The table columns fetched by rowid for the results, and the bindings they are emitted under: those the operator
	// above reads.
	vector<ColumnIndex> column_ids;
	vector<ColumnBinding> column_bindings;

	DuckTableEntry &table;
	VSSIndexPlanInfo index_info;
	// The limit (the K in KNN search).
	const idx_t limit;
	unsafe_unique_array<float> query_embedding;

public:
	string GetName() const override {
		return "DUCKIR_VSS_INDEX_SCAN";
	}

	void ResolveTypes() override;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

	vector<ColumnBinding> GetColumnBindings() override;
};

} // namespace duckdb

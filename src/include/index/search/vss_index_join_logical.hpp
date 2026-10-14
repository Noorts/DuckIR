#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

class Index;
class DuckTableEntry;

// Searches the index once for each query row of its first child, and emits each query row with its K nearest rows of
// the table. It replaces the cross product between the table (possibly filtered) and the query rows in the plan
// DuckDB builds for a LATERAL join, and emits the cross product's columns: the query rows' columns, and the table
// columns fetched by rowid.
class LogicalVSSIndexJoin : public LogicalExtensionOperator {
public:
	LogicalVSSIndexJoin(DuckTableEntry &table, Index &index, idx_t limit, ColumnBinding query_binding,
	                    vector<ColumnIndex> column_ids, vector<ColumnBinding> column_bindings)
	    : LogicalExtensionOperator(), query_binding(query_binding), column_ids(std::move(column_ids)),
	      column_bindings(std::move(column_bindings)), table(table), index(index), limit(limit) {
	}

	// children[0] emits the query rows. The optional children[1] emits the rowids of the table's rows that pass the
	// predicate, as its only column; without it, all rows of the table are searched.

	// The query vector's column in children[0].
	ColumnBinding query_binding;
	// The table columns fetched by rowid for the results, and the bindings they are emitted under: those the operator
	// above reads.
	vector<ColumnIndex> column_ids;
	vector<ColumnBinding> column_bindings;

	DuckTableEntry &table;
	Index &index;
	// The number of nearest rows per query (`k`).
	const idx_t limit;

public:
	string GetName() const override {
		return "DUCKIR_VSS_INDEX_JOIN";
	}

	void ResolveTypes() override;

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

	vector<ColumnBinding> GetColumnBindings() override;
};

} // namespace duckdb

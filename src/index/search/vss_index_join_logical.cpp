#include "index/search/vss_index_join_logical.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"

#include "index/search/vss_index_join_physical.hpp"

namespace duckdb {

PhysicalOperator &LogicalVSSIndexJoin::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	const auto query_row_bindings = children[0]->GetColumnBindings();
	const auto query_position = std::find(query_row_bindings.begin(), query_row_bindings.end(), query_binding);
	if (query_position == query_row_bindings.end()) {
		throw InternalException("DUCKIR_VSS_INDEX_JOIN: the query vector is not a column of the query rows");
	}
	const auto query_column = static_cast<idx_t>(query_position - query_row_bindings.begin());

	auto &physical_op =
	    planner.Make<PhysicalVSSIndexJoin>(types, table, index, limit, query_column, column_ids, estimated_cardinality);
	// The query rows, then (when the table is filtered) the rowids of the rows that pass the predicate.
	for (auto &child : children) {
		physical_op.children.push_back(planner.CreatePlan(*child));
	}
	return physical_op;
}

vector<ColumnBinding> LogicalVSSIndexJoin::GetColumnBindings() {
	// The query rows' columns, then the fetched table columns.
	auto bindings = children[0]->GetColumnBindings();
	bindings.insert(bindings.end(), column_bindings.begin(), column_bindings.end());
	return bindings;
}

void LogicalVSSIndexJoin::ResolveTypes() {
	types = children[0]->types;
	for (auto &col_idx : column_ids) {
		if (col_idx.IsRowIdColumn()) {
			types.push_back(LogicalType::ROW_TYPE);
		} else {
			auto logical_idx = col_idx.GetPrimaryIndex();
			auto &col = table.GetColumn(LogicalIndex(logical_idx));
			types.push_back(col.Type());
		}
	}
}

} // namespace duckdb

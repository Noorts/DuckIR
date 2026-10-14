#include "index/search/vss_index_filtered_scan_logical.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"

#include "index/search/vss_index_filtered_scan_physical.hpp"

namespace duckdb {

PhysicalOperator &LogicalVSSIndexFilteredScan::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	auto bind_data = make_uniq<VSSIndexPhysicalScanBindData>(table, index, limit, std::move(query_embedding));
	auto &physical_op = planner.Make<PhysicalVSSIndexFilteredScan>(types, std::move(bind_data), column_ids, limit);

	// The child operator should output row_ids (from a modified Get)
	if (!children.empty()) {
		auto &child = planner.CreatePlan(*children[0]);
		physical_op.children.push_back(child);
	}

	return physical_op;
}

vector<ColumnBinding> LogicalVSSIndexFilteredScan::GetColumnBindings() {
	return column_bindings;
}

void LogicalVSSIndexFilteredScan::ResolveTypes() {
	types.clear();
	types.reserve(column_ids.size());
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

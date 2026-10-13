#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

#include "duckdb/parser/parsed_data/create_index_info.hpp"

#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"

#include "index/pdxearch_index.hpp"
#include "index/create/pdxearch_index_create.hpp"

#include <algorithm>

namespace duckdb {

PhysicalOperator &PDXearchIndex::CreatePlan(PlanIndexInput &input) {
	auto &create_index = input.op;
	auto &planner = input.planner;

	// Verify the index creation options.
	// Example: `CREATE INDEX ON t USING PDXearch(vec) WITH (n_probe = 10, seed = 42)`.
	for (auto &option : create_index.info->options) {
		auto &k = option.first;
		auto &v = option.second;
		if (StringUtil::CIEquals(k, "metric")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("PDXearch index 'metric' must be a string");
			}
			auto metric = v.GetValue<string>();
			if (PDXearchIndex::DISTANCE_METRIC_MAP.find(metric) == PDXearchIndex::DISTANCE_METRIC_MAP.end()) {
				vector<string> allowed_metrics;
				for (auto &entry : PDXearchIndex::DISTANCE_METRIC_MAP) {
					allowed_metrics.push_back(StringUtil::Format("'%s'", entry.first));
				}
				std::sort(allowed_metrics.begin(), allowed_metrics.end());
				throw BinderException("PDXearch index 'metric' must be one of: %s",
				                      StringUtil::Join(allowed_metrics, ", "));
			}
		} else if (StringUtil::CIEquals(k, "quantization")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("PDXearch index 'quantization' must be a string");
			}
			auto quantization = v.GetValue<string>();
			if (PDXearchIndex::QUANTIZATION_MAP.find(quantization) == PDXearchIndex::QUANTIZATION_MAP.end()) {
				vector<string> allowed_quantizations;
				for (auto &entry : PDXearchIndex::QUANTIZATION_MAP) {
					allowed_quantizations.push_back(StringUtil::Format("'%s'", entry.first));
				}
				std::sort(allowed_quantizations.begin(), allowed_quantizations.end());
				throw BinderException("PDXearch index 'quantization' must be one of: %s",
				                      StringUtil::Join(allowed_quantizations, ", "));
			}
		} else if (StringUtil::CIEquals(k, "n_probe")) {
			if (v.type() != LogicalType::INTEGER) {
				throw BinderException("PDXearch index 'n_probe' must be an integer");
			}
			if (v.GetValue<int32_t>() < 0) {
				throw BinderException("PDXearch index 'n_probe' must be at least 0, default is %d",
				                      PDXearchWrapper::DEFAULT_N_PROBE);
			}
		} else if (StringUtil::CIEquals(k, "seed")) {
			if (v.type() != LogicalType::INTEGER) {
				throw BinderException("PDXearch index 'seed' must be between -2147483647 and 2147483647, inclusive");
			}
		} else {
			throw BinderException("Unknown option for PDXearch index: '%s'", k);
		}
	}

	if (create_index.expressions.size() != 1) {
		throw BinderException("PDXearch indexes can only be created over a single column of keys.");
	}
	const auto &arr_type = create_index.expressions[0]->return_type;
	if (arr_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("PDXearch index keys must be of type FLOAT[N]");
	}

	const auto &child_type = ArrayType::GetChildType(arr_type);
	if (child_type.id() != LogicalTypeId::FLOAT) {
		throw BinderException("PDXearch index key type must be FLOAT");
	}

	const auto arr_dims = ArrayType::GetSize(arr_type);
	if (arr_dims > PDX::PDX_MAX_DIMS) {
		throw BinderException(
		    "PDXearch index FLOAT array length (i.e., dimensions) must be less than or equal to %d, got %d",
		    PDX::PDX_MAX_DIMS, arr_dims);
	}

	// Projection to execute expressions on the key columns
	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (auto &expression : create_index.expressions) {
		new_column_types.push_back(expression->return_type);
		select_list.push_back(std::move(expression));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(
	    make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, create_index.info->scan_types.size() - 1));

	create_index.estimated_cardinality = input.table_scan.estimated_cardinality;

	auto &projection =
	    planner.Make<PhysicalProjection>(new_column_types, std::move(select_list), create_index.estimated_cardinality);
	projection.children.push_back(input.table_scan);

	// Filter operator for IS_NOT_NULL on each key column
	vector<LogicalType> filter_types;
	vector<unique_ptr<Expression>> filter_select_list;

	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		filter_types.push_back(new_column_types[i]);
		auto is_not_null_expr =
		    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
		auto bound_ref = make_uniq<BoundReferenceExpression>(new_column_types[i], i);
		is_not_null_expr->children.push_back(std::move(bound_ref));
		filter_select_list.push_back(std::move(is_not_null_expr));
	}

	auto &null_filter = planner.Make<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list),
	                                                 create_index.estimated_cardinality);
	null_filter.types.emplace_back(LogicalType::ROW_TYPE);
	null_filter.children.push_back(projection);

	auto &physical_create_index = planner.Make<PhysicalCreatePDXearchIndex>(
	    create_index.types, create_index.table, create_index.info->column_ids, std::move(create_index.info),
	    std::move(create_index.unbound_expressions), create_index.estimated_cardinality);
	physical_create_index.children.push_back(null_filter);
	return physical_create_index;
}

} // namespace duckdb

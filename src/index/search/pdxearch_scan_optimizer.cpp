#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"

#include "index/pdxearch_blob_codec.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "index/pdxearch_module.hpp"
#include "index/pdxearch_index.hpp"
#include "index/search/pdxearch_index_filtered_scan_logical.hpp"
#include "index/search/pdxearch_index_join_logical.hpp"
#include "index/search/pdxearch_index_scan_logical.hpp"
#include "index/search/pdxearch_index_scan_physical.hpp"

namespace duckdb {

/**
 * Optimizes various vector similarity search queries, where the table has a
 * matching PDXearch index.
 *
 * Note: this optimizer sets `optimize_function` and thus runs after DuckDB's
 * optimization passes. Its `pre_optimize_function` runs before them (see LATE
 * MATERIALIZATION below).
 *
 ********************** SCENARIO 1 (non-filtered search): **********************
 *
 * Example query (on a table with an id and a vec column):
 *   SELECT * FROM table
 *   ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3])
 *   LIMIT 10;
 *
 * Scenario 1 targets the logical query plan shown on the left (this has been
 * optimized by DuckDB's optimization passes, but has not been optimized by this
 * optimizer yet; e.g., print TryOptimize's `plan`). This optimizer replaces the
 * TopN operator and the sequential scan with a PDXearch index scan, turning it
 * into the logical query plan shown on the right.
 *
 *              [IN]                                [OUT]
 * ┌───────────────────────────┐        ┌───────────────────────────┐
 * │         PROJECTION        │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │        Expressions:       │        │        Expressions:       │
 * │           #[1.0]          │        │             #0            │
 * │           #[1.1]          │        │             #1            │
 * │                           │        │                           │
 * │          ~10 rows         │        │          ~10 rows         │
 * └─────────────┬─────────────┘        └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐        ┌─────────────┴─────────────┐
 * │           TOP_N           │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │          ~10 rows         │        │        Expressions:       │
 * └─────────────┬─────────────┘        │             id            │
 * ┌─────────────┴─────────────┐        │            vec            │
 * │         PROJECTION        │        │            NULL           │
 * │    ────────────────────   │        │                           │
 * │        Expressions:       │        │      ~1,000,000 rows      │
 * │             id            │        └─────────────┬─────────────┘
 * │            vec            │        ┌─────────────┴─────────────┐
 * │    array_distance(vec,    │        │    PDXEARCH_INDEX_SCAN    │
 * │     [1.0, 2.0, 3.0])      │        │    ────────────────────   │
 * │                           │        │        Table: table       │
 * │      ~1,000,000 rows      │        │    PDXearch Index: idx    │
 * └─────────────┬─────────────┘        │       Clusters: 4000      │
 * ┌─────────────┴─────────────┐        │                           │
 * │          SEQ_SCAN         │        │        Projections:       │
 * │    ────────────────────   │        │             id            │
 * │        Table: table       │        │            vec            │
 * │   Type: Sequential Scan   │        │                           │
 * │                           │        │      ~1,000,000 rows      │
 * │        Projections:       │        └───────────────────────────┘
 * │             id            │
 * │            vec            │
 * │                           │
 * │      ~1,000,000 rows      │
 * └───────────────────────────┘
 *
 *
 ******************** SCENARIO 2 (simple filtered search): *********************
 *
 * Simple (non-composite) filters pushed down into the table scan.
 *
 * Example query (on a table with an id and a vec column):
 *   SELECT * FROM table
 *   WHERE id > 50
 *   ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3])
 *   LIMIT 10;
 *
 * Scenario 2 targets the logical query plan shown on the left (this has
 * been optimized by DuckDB's optimization passes, but has not been optimized by
 * this optimizer yet). This optimizer produces the logical query plan shown on
 * the right. Projections are shown for clarity (normally not part of the
 * logical plan).
 *
 *              [IN]                                [OUT]
 * ┌───────────────────────────┐        ┌───────────────────────────┐
 * │         PROJECTION        │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │        Expressions:       │        │        Expressions:       │
 * │           #[1.0]          │        │             #0            │
 * │           #[1.1]          │        │             #1            │
 * │                           │        │                           │
 * │          ~10 rows         │        │          ~10 rows         │
 * └─────────────┬─────────────┘        └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐        ┌─────────────┴─────────────┐
 * │           TOP_N           │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │          ~10 rows         │        │        Expressions:       │
 * └─────────────┬─────────────┘        │             id            │
 * ┌─────────────┴─────────────┐        │            vec            │
 * │         PROJECTION        │        │            NULL           │
 * │    ────────────────────   │        │                           │
 * │        Expressions:       │        │      ~1,000,000 rows      │
 * │             id            │        └─────────────┬─────────────┘
 * │            vec            │        ┌─────────────┴─────────────┐
 * │    array_distance(vec,    │        │  PDXEARCH_INDEX_FILT_SCAN │
 * │     [1.0, 2.0, 3.0])      │        │    ────────────────────   │
 * │                           │        │        Table: table       │
 * │      ~1,000,000 rows      │        │    PDXearch Index: idx    │
 * └─────────────┬─────────────┘        │       Clusters: 4000      │
 * ┌─────────────┴─────────────┐        │                           │
 * │          SEQ_SCAN         │        │        Projections:       │
 * │    ────────────────────   │        │             id            │
 * │        Table: table       │        │            vec            │
 * │   Type: Sequential Scan   │        │                           │
 * │                           │        │      ~1,000,000 rows      │
 * │        Projections:       │        └─────────────┬─────────────┘
 * │             id            │        ┌─────────────┴─────────────┐
 * │            vec            │        │          SEQ_SCAN         │
 * │                           │        │    ────────────────────   │
 * │       Filters: id>50      │        │        Table: table       │
 * │                           │        │   Type: Sequential Scan   │
 * │      ~1,000,000 rows      │        │                           │
 * └───────────────────────────┘        │        Projections:       │
 *                                      │           rowid           │
 *                                      │                           │
 *                                      │       Filters: id>50      │
 *                                      │                           │
 *                                      │      ~1,000,000 rows      │
 *                                      └───────────────────────────┘
 *
 *
 ****************** SCENARIO 3 (residual filtered search): *********************
 *
 * Predicates DuckDB cannot push into the table scan (e.g. an OR across two
 * columns, or an expression over two columns) stay in FILTER operators above
 * it. Scenario 3 handles one or more FILTER operators between the projection
 * and the table scan (which may have pushed-down filters as well).
 *
 * Example query (on a table with an id, a cat and a vec column):
 *   SELECT * FROM table
 *   WHERE id > 50 OR cat = 3
 *   ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3])
 *   LIMIT 10;
 *
 * The filters move below the PDXEARCH_INDEX_FILT_SCAN unchanged. The table scan
 * and every filter forward the rowid, and a projection on top keeps only the
 * rowid, which the filtered search consumes. When no predicate reads the
 * embedding, the rowid takes the embedding's slot in the table scan, so the
 * filter pipeline does not read embeddings. The filtered search fetches the
 * table scan's original columns by rowid for its K results.
 *
 * Subqueries and views (DuckDB binds a view as a subquery) add projections that
 * forward the table's columns. They may sit anywhere in the chain between the
 * projection and the table scan, mixed with the FILTER operators, and the
 * rowid is forwarded through them as well. The index scans emit the columns
 * the projection reads under the bindings it reads them with, so a chain
 * without FILTER operators is replaced entirely (scenarios 1 and 2). A query
 * that reads a value the chain computes (e.g. `id * 2 AS x` of a subquery) is
 * not optimized. Projections that only forward the distance (`ORDER BY d` over
 * a subquery that computes d) stay above the search.
 *
 * DuckDB turns an IN list of at least five constants that stays in a FILTER
 * into a MARK join between the filter's child and a scan of the constants, and
 * the FILTER tests the join's marker (e.g. `cat = 3 OR id IN (...)`). The MARK
 * join keeps every row of its left child, so it is part of the chain as well:
 * it stays below the search, evaluates the list, and forwards the rowid. A
 * query that reads the marker as a value (IN in the SELECT list) is not
 * optimized.
 *
 * Subqueries used as filters join the chain the same way when the indexed
 * table is the join's left child (the probe side): IN and EXISTS become a SEMI
 * join, NOT EXISTS an ANTI join, and an IN subquery inside a larger predicate
 * (e.g. `NOT IN (SELECT ...)`) a MARK join. These joins keep each row of the
 * table at most once, and keep the runtime filters DuckDB pushes from them into
 * the table scan. When DuckDB builds the hash table on the indexed table
 * instead (RIGHT_SEMI / RIGHT_ANTI), or leaves a correlated subquery as a
 * DELIM_JOIN, the query is not optimized.
 *
 *              IN                                  OUT
 * ┌───────────────────────────┐        ┌───────────────────────────┐
 * │         PROJECTION        │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │        Expressions:       │        │        Expressions:       │
 * │           #[1.0]          │        │             #0            │
 * │           #[1.1]          │        │             #1            │
 * │           #[1.2]          │        │             #2            │
 * │                           │        │                           │
 * │          ~10 rows         │        │          ~10 rows         │
 * └─────────────┬─────────────┘        └─────────────┬─────────────┘
 * ┌─────────────┴─────────────┐        ┌─────────────┴─────────────┐
 * │           TOP_N           │        │         PROJECTION        │
 * │    ────────────────────   │        │    ────────────────────   │
 * │                           │        │        Expressions:       │
 * │          ~10 rows         │        │             id            │
 * └─────────────┬─────────────┘        │            cat            │
 * ┌─────────────┴─────────────┐        │            vec            │
 * │         PROJECTION        │        │            NULL           │
 * │    ────────────────────   │        │                           │
 * │        Expressions:       │        │          ~10 rows         │
 * │             id            │        └─────────────┬─────────────┘
 * │            cat            │        ┌─────────────┴─────────────┐
 * │            vec            │        │  PDXEARCH_INDEX_FILT_SCAN │
 * │    array_distance(vec,    │        │    ────────────────────   │
 * │      [1.0, 2.0, 3.0])     │        │        Table: table       │
 * │                           │        │    PDXearch Index: idx    │
 * │       ~550,000 rows       │        │                           │
 * └─────────────┬─────────────┘        │        Projections:       │
 * ┌─────────────┴─────────────┐        │             id            │
 * │           FILTER          │        │            cat            │
 * │    ────────────────────   │        │            vec            │
 * │        Expressions:       │        │                           │
 * │  ((id > 50) OR (cat = 3)) │        │          ~10 rows         │
 * │                           │        └─────────────┬─────────────┘
 * │       ~550,000 rows       │        ┌─────────────┴─────────────┐
 * └─────────────┬─────────────┘        │         PROJECTION        │
 * ┌─────────────┴─────────────┐        │    ────────────────────   │
 * │          SEQ_SCAN         │        │        Expressions:       │
 * │    ────────────────────   │        │           rowid           │
 * │        Table: table       │        │                           │
 * │   Type: Sequential Scan   │        │       ~550,000 rows       │
 * │                           │        └─────────────┬─────────────┘
 * │        Projections:       │        ┌─────────────┴─────────────┐
 * │             id            │        │           FILTER          │
 * │            cat            │        │    ────────────────────   │
 * │            vec            │        │        Expressions:       │
 * │                           │        │  ((id > 50) OR (cat = 3)) │
 * │      ~1,000,000 rows      │        │                           │
 * └───────────────────────────┘        │       ~550,000 rows       │
 *                                      └─────────────┬─────────────┘
 *                                      ┌─────────────┴─────────────┐
 *                                      │          SEQ_SCAN         │
 *                                      │    ────────────────────   │
 *                                      │        Table: table       │
 *                                      │   Type: Sequential Scan   │
 *                                      │                           │
 *                                      │        Projections:       │
 *                                      │             id            │
 *                                      │            cat            │
 *                                      │           rowid           │
 *                                      │                           │
 *                                      │      ~1,000,000 rows      │
 *                                      └───────────────────────────┘
 *
 ************ SCENARIO 4 (multi-query search, LATERAL join): *******************
 *
 * Each row of another table supplies a query vector, and the table (possibly
 * filtered, as in scenarios 1 to 3) is searched once per query.
 *
 * Example query (q has a qid and a query column):
 *   SELECT q.qid, s.id FROM q, LATERAL (
 *     SELECT id FROM table WHERE id > 50 ORDER BY array_distance(vec, q.query)
 *     LIMIT 10) s;
 *
 * DuckDB decorrelates the subquery before any optimizer runs: the table is
 * joined with the distinct query vectors (a DELIM_GET) by a cross product, and
 * the ORDER BY ... LIMIT becomes a ROW_NUMBER window partitioned by the query
 * vector with a FILTER on the row number. Its TopN window elimination then
 * turns these into an AGGREGATE per query vector (with an OFFSET, the FILTER
 * and the WINDOW stay). There is no TopN, and the DELIM_JOIN above joins the
 * results back to q. The index join replaces only the cross product:
 *
 *                     IN                                         OUT
 *   AGGREGATE [query |                          AGGREGATE [query |
 *     arg_min_nulls_last(id, #1, 10)]             arg_min_nulls_last(id, #1, 10)]
 *     PROJECTION [id,                             PROJECTION [id,
 *       array_distance(vec, query), query]          array_distance(vec, query), query]
 *       CROSS_PRODUCT                               PDXEARCH_INDEX_JOIN (K = 10)
 *         SEQ_SCAN [table, Filters: id>50]            DELIM_GET [query]
 *         DELIM_GET [query]                           SEQ_SCAN [table, Filters: id>50,
 *                                                       Projections: rowid]
 *
 * For each query vector, the index join emits its K nearest rows of the table
 * that pass the predicate, with the query row's columns: the cross product's
 * columns, so the operators above keep working unchanged. The projection
 * computes the distance of those rows, and the AGGREGATE (or the FILTER on the
 * row number) picks each query's top K among them, which are all of them. Its
 * groups are the query rows, so a group's top K is among the K nearest rows of
 * its query rows. The DELIM_JOIN joins the query vectors back to the rows of q
 * (INNER or LEFT). Without a predicate, the index join has no second child
 * (scenario 1). Otherwise its second child emits the rowids of the passing
 * rows, built as in scenarios 2 and 3, which it collects before the queries
 * arrive. Projections of a view or subquery over the table, which forward its
 * columns and the query vector, stay between the projection and the index join.
 *
 * K is at most STANDARD_VECTOR_SIZE (LIMIT + OFFSET), so a query's rows fill at
 * most one output chunk. Descending orders, several ORDER BY keys, NULLS FIRST,
 * predicates that read the query row (e.g. `WHERE table.cat = q.cat`), query
 * vectors computed from the query row, and the same top K written without
 * LATERAL (e.g. with QUALIFY) are not optimized.
 *
 ***************************** LATE MATERIALIZATION: ***************************
 *
 * For a TopN of at most `late_materialization_max_rows` rows (default 50),
 * DuckDB's late materialization reads only the order key and the rowid of
 * every row to find the top K rows, then fetches their other columns with a
 * second scan of the table joined on the rowid (a SEMI join), and sorts them
 * again. This pays off when the other columns are wide. A search already works
 * this way: the index finds the K rowids and the index scan fetches their
 * columns, so on a search the rewrite only adds the second scan, the join and
 * the sort (about twice as slow for K = 10).
 *
 * Therefore `pre_optimize_function` (PreOptimize) looks at the plan before
 * DuckDB's optimizers run, finds each ORDER BY that TryOptimize will turn into
 * a search (TryDisableLateMaterialization, the same checks up to the rewrite),
 * and turns late materialization off for its table scan: the scan's copy of
 * the table function has a `late_materialization` flag that DuckDB checks.
 * Other scans, also of the same table, keep it. DuckDB copies a CTE that it
 * inlines at several places by serializing it, which restores the flag, so a
 * search inside such a CTE still gets the rewrite (correct, but slower).
 *
 * TODO: Leave out the PROJECTION at the top? Our pattern recognition starts at
 * the TopN operator.
 * TODO: Fix cardinality of after-optimization plan (K instead of 1M?).
 *       Also see filtered scan optimizer.
 */
class PDXearchIndexScanOptimizer : public OptimizerExtension {
public:
	PDXearchIndexScanOptimizer() {
		pre_optimize_function = PreOptimize;
		optimize_function = Optimize;
	}

	// The operators the chain between the projection and the table scan may hold, each continuing into children[0]:
	// FILTER operators, projections (subqueries and views), and the joins that keep each row of their left child at
	// most once. DuckDB plans IN and EXISTS subqueries as SEMI joins and NOT EXISTS as an ANTI join. A MARK join keeps
	// every row of its left child and adds a marker column, which a FILTER above it tests: DuckDB makes one of an IN
	// list of at least five constants (the right child is then a column data scan) and of an IN subquery inside a
	// larger predicate (e.g. `NOT IN (SELECT ...)`, `cat = 3 OR id IN (SELECT ...)`).
	static bool IsChainOperator(const LogicalOperator &op) {
		switch (op.type) {
		// To support: OR or sparse IN on one column, Predicates over two or more columns, Conjunction left after
		// pushdown, Volatile predicate, Selective residual filter, Filter reading the embedding, Filter on the rowid,
		// Subquery or view with a residual filter.
		case LogicalOperatorType::LOGICAL_FILTER:
		// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate,
		// Renamed embedding.
		case LogicalOperatorType::LOGICAL_PROJECTION:
			return op.children.size() == 1;
		// To support: IN list of at least five constants and IN lists combined with other predicates (MARK), Subqueries
		// as filters, our table on the probe side (SEMI, ANTI, MARK).
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
			const auto join_type = op.Cast<LogicalComparisonJoin>().join_type;
			return op.children.size() == 2 &&
			       (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::MARK);
		}
		default:
			return false;
		}
	}

	// Follows a binding read above chain.front() down the chain (ordered from the top down to the table scan) to the
	// table scan's binding it comes from. A filter forwards its child's bindings, and a projection forwards a binding
	// when its expression is a plain column reference. False when a projection computes the value instead.
	// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate,
	// Renamed embedding.
	static bool TryTraceToTableScan(const vector<reference<LogicalOperator>> &chain, ColumnBinding &binding) {
		for (auto &op : chain) {
			if (op.get().type != LogicalOperatorType::LOGICAL_PROJECTION) {
				continue;
			}
			auto &projection = op.get().Cast<LogicalProjection>();
			if (binding.table_index != projection.table_index ||
			    binding.column_index >= projection.expressions.size()) {
				return false;
			}
			auto &expression = *projection.expressions[binding.column_index];
			if (expression.type != ExpressionType::BOUND_COLUMN_REF) {
				return false;
			}
			binding = expression.Cast<BoundColumnRefExpression>().binding;
		}
		return true;
	}

	// Collects the bindings the projection reads from its child, and the table scan's column each one traces to: an
	// index scan replacing the projection's child emits those columns, fetched by rowid, under those bindings. False
	// when the projection reads a value that the chain computes. Bindings of `passed_through_table_index` are skipped:
	// the operator replacing the child passes them through (the query rows of an index join).
	// To support: Filtered subquery, Subquery or view with a residual filter, View or subquery without a predicate.
	static bool TryCollectOutputColumns(LogicalProjection &projection, const vector<reference<LogicalOperator>> &chain,
	                                    const LogicalGet &get, vector<ColumnBinding> &bindings,
	                                    vector<ColumnIndex> &column_ids,
	                                    const optional_idx passed_through_table_index = optional_idx()) {
		bool all_traced = true;
		for (auto &expression : projection.expressions) {
			ExpressionIterator::EnumerateExpression(expression, [&](Expression &child) {
				if (child.type != ExpressionType::BOUND_COLUMN_REF) {
					return;
				}
				const auto binding = child.Cast<BoundColumnRefExpression>().binding;
				if (std::find(bindings.begin(), bindings.end(), binding) != bindings.end()) {
					return;
				}
				if (passed_through_table_index.IsValid() &&
				    binding.table_index == passed_through_table_index.GetIndex()) {
					return;
				}
				auto traced_binding = binding;
				if (!TryTraceToTableScan(chain, traced_binding) || traced_binding.table_index != get.table_index) {
					all_traced = false;
					return;
				}
				bindings.push_back(binding);
				column_ids.push_back(get.GetColumnIds()[traced_binding.column_index]);
			});
		}
		return all_traced;
	}

	// Whether the rowid can take the embedding's slot in the table scan: the scan has no rowid yet, no pushed-down
	// filter reads the embedding, and the chain only forwards it (no predicate and no computed value reads it).
	// `forwards` receives the projections' column references that forward it.
	static bool CanReplaceEmbeddingByRowId(const LogicalGet &get, const vector<reference<LogicalOperator>> &chain,
	                                       const idx_t embedding_column_position,
	                                       vector<reference<BoundColumnRefExpression>> &forwards) {
		const auto &column_ids = get.GetColumnIds();
		for (auto &column_id : column_ids) {
			if (column_id.IsRowIdColumn()) {
				return false;
			}
		}
		if (get.table_filters.filters.find(column_ids[embedding_column_position].GetPrimaryIndex()) !=
		    get.table_filters.filters.end()) {
			return false;
		}
		// The bindings that carry the embedding, from the table scan up the chain.
		vector<ColumnBinding> embedding_bindings {ColumnBinding(get.table_index, embedding_column_position)};
		const auto reads_embedding = [&](unique_ptr<Expression> &expression) {
			bool reads = false;
			ExpressionIterator::EnumerateExpression(expression, [&](Expression &child) {
				if (child.type == ExpressionType::BOUND_COLUMN_REF &&
				    std::find(embedding_bindings.begin(), embedding_bindings.end(),
				              child.Cast<BoundColumnRefExpression>().binding) != embedding_bindings.end()) {
					reads = true;
				}
			});
			return reads;
		};
		for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
			auto &op = it->get();
			// To support filter reading the embedding (the scan keeps the embedding and appends the rowid instead).
			if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
				for (auto &expression : op.expressions) {
					if (reads_embedding(expression)) {
						return false;
					}
				}
				continue;
			}
			// To support: IN list of at least five constants, IN lists combined with other predicates, Subqueries as
			// filters, our table on the probe side.
			if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				// A join's predicate is its condition, whose left side reads the chain below it.
				for (auto &condition : op.Cast<LogicalComparisonJoin>().conditions) {
					if (reads_embedding(condition.left)) {
						return false;
					}
				}
				continue;
			}
			// To support: Subquery or view with a residual filter, Renamed embedding (projections that forward it).
			auto &projection = op.Cast<LogicalProjection>();
			vector<ColumnBinding> forwarded_bindings;
			for (idx_t i = 0; i < projection.expressions.size(); i++) {
				auto &expression = projection.expressions[i];
				if (expression->type == ExpressionType::BOUND_COLUMN_REF && reads_embedding(expression)) {
					forwards.push_back(expression->Cast<BoundColumnRefExpression>());
					forwarded_bindings.emplace_back(projection.table_index, i);
				} else if (reads_embedding(expression)) {
					return false;
				}
			}
			embedding_bindings = std::move(forwarded_bindings);
		}
		return true;
	}

	// Makes the rowid of the rows that pass the chain (ordered from the top down to `get`) come out of the chain's top
	// operator, and returns its binding there. `embedding_binding` is the embedding's binding at the top.
	static ColumnBinding PassRowIdThroughChain(LogicalGet &get, const vector<reference<LogicalOperator>> &chain,
	                                           const ColumnBinding embedding_binding,
	                                           const optional_idx embedding_column_position) {
		auto &column_ids = get.GetMutableColumnIds();
		// Above the filtered search the embedding is fetched again by rowid, so below it the scan only needs it for
		// predicates. If no predicate reads it and the scan has no rowid yet, the rowid takes the embedding's slot:
		// the chain forwards it along the embedding's path, and the filter pipeline no longer reads embeddings.
		// To support the chain cases whose predicates do not read the embedding.
		vector<reference<BoundColumnRefExpression>> forwards;
		if (embedding_column_position.IsValid() &&
		    CanReplaceEmbeddingByRowId(get, chain, embedding_column_position.GetIndex(), forwards)) {
			column_ids[embedding_column_position.GetIndex()] = ColumnIndex(COLUMN_IDENTIFIER_ROW_ID);
			for (auto &forward : forwards) {
				forward.get().return_type = LogicalType::ROW_TYPE;
			}
			return embedding_binding;
		}
		// Otherwise the scan emits the rowid as well, and every operator of the chain forwards it. Appending keeps the
		// positions of the other columns, which bindings and projection maps refer to.
		// To support: Filter reading the embedding (rowid appended), Filter on the rowid (existing rowid reused).
		idx_t rowid_position = column_ids.size();
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].IsRowIdColumn()) {
				rowid_position = i;
				break;
			}
		}
		if (rowid_position == column_ids.size()) {
			get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
		}
		if (!get.projection_ids.empty() && std::find(get.projection_ids.begin(), get.projection_ids.end(),
		                                             rowid_position) == get.projection_ids.end()) {
			get.projection_ids.push_back(rowid_position);
		}
		ColumnBinding rowid_binding(get.table_index, rowid_position);
		for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
			// To support subquery or view with a residual filter.
			if (it->get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &projection = it->get().Cast<LogicalProjection>();
				projection.expressions.push_back(
				    make_uniq<BoundColumnRefExpression>(LogicalType::ROW_TYPE, rowid_binding));
				rowid_binding = ColumnBinding(projection.table_index, projection.expressions.size() - 1);
				continue;
			}
			// A FILTER and a join forward the bindings of children[0], trimmed by a projection map when it is set.
			// To support: the FILTER cases listed in IsChainOperator, IN list of at least five constants, IN lists
			// combined with other predicates, Subqueries as filters, our table on the probe side.
			auto &projection_map = it->get().type == LogicalOperatorType::LOGICAL_FILTER
			                           ? it->get().Cast<LogicalFilter>().projection_map
			                           : it->get().Cast<LogicalComparisonJoin>().left_projection_map;
			if (projection_map.empty()) {
				continue;
			}
			const auto child_bindings = it->get().children[0]->GetColumnBindings();
			const auto child_position = std::find(child_bindings.begin(), child_bindings.end(), rowid_binding);
			D_ASSERT(child_position != child_bindings.end());
			const auto position = static_cast<idx_t>(child_position - child_bindings.begin());
			if (std::find(projection_map.begin(), projection_map.end(), position) == projection_map.end()) {
				projection_map.push_back(position);
			}
		}
		return rowid_binding;
	}

	// Whether `orders` (of a TopN, or of an ORDER BY before DuckDB's optimizers) is a single ascending key that puts
	// NULLs last and reads a column of the operator's child. `binding` receives that column.
	static bool TryMatchOrderKey(const vector<BoundOrderByNode> &orders, ColumnBinding &binding) {
		if (orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.null_order != OrderByNullType::NULLS_LAST) {
			// Rows without an embedding have a NULL distance and are not in the index, so the index can only produce
			// them last. The binder has already resolved an unspecified null order (default_null_order).
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		binding = order.expression->Cast<BoundColumnRefExpression>().binding;
		return true;
	}

	// Follows a key that reads `child` (e.g. the TopN key) through projections that only forward it (e.g. `ORDER BY d`
	// over a subquery that computes d) to the projection that computes it, and returns that projection. `binding`
	// becomes the key's binding in it. The forwarding projections stay above the search.
	// Needed to support distance aliased in a subquery.
	static optional_ptr<LogicalProjection> TryFindDistanceProjection(LogicalOperator &child, ColumnBinding &binding) {
		if (child.type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return nullptr;
		}
		reference<LogicalProjection> distance_projection = child.Cast<LogicalProjection>();
		while (binding.column_index < distance_projection.get().expressions.size() &&
		       distance_projection.get().expressions[binding.column_index]->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &forwarding_projection = distance_projection.get();
			binding = forwarding_projection.expressions[binding.column_index]->Cast<BoundColumnRefExpression>().binding;
			if (forwarding_projection.children.size() != 1 ||
			    forwarding_projection.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
				return nullptr;
			}
			distance_projection = forwarding_projection.children.front()->Cast<LogicalProjection>();
			if (binding.table_index != distance_projection.get().table_index) {
				return nullptr;
			}
		}
		if (binding.column_index >= distance_projection.get().expressions.size()) {
			return nullptr;
		}
		return distance_projection.get();
	}

	// Walks the chain of FILTER operators (the predicates DuckDB could not push into the table scan), projections
	// (subqueries and views) and joins (IN lists and subqueries) from `top` down to the table scan it must end in.
	// IsChainOperator lists which case each kind of operator supports. `chain` receives the operators from `top` down
	// to the table scan (excluded), and `chain_filters_rows` whether they decide which rows pass: FILTER operators and
	// joins do, forwarding projections do not. Returns the table scan's slot, or nullptr when the chain does not end in
	// a table scan we can search.
	static unique_ptr<LogicalOperator> *TryWalkChainToTableScan(unique_ptr<LogicalOperator> &top,
	                                                            vector<reference<LogicalOperator>> &chain,
	                                                            bool &chain_filters_rows) {
		auto *get_ptr_ptr = &top;
		while (IsChainOperator(**get_ptr_ptr)) {
			chain_filters_rows |= (*get_ptr_ptr)->type != LogicalOperatorType::LOGICAL_PROJECTION;
			chain.push_back(**get_ptr_ptr);
			get_ptr_ptr = &(*get_ptr_ptr)->children.front();
		}
		if ((*get_ptr_ptr)->type != LogicalOperatorType::LOGICAL_GET) {
			return nullptr;
		}

		auto &get = (*get_ptr_ptr)->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return nullptr;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return nullptr;
		}

		// We can only replace the scan if the table is a duck table
		if (!get.GetTable()->IsDuckTable()) {
			return nullptr;
		}
		return get_ptr_ptr;
	}

	// Walks the projections that only forward columns (plain column references) from `top` down to the cross product
	// below them. They come from a view or subquery over the table in a LATERAL join: DuckDB adds the query rows'
	// columns to its projection. `forwarding_projections` receives them from the top down. Returns the cross product's
	// slot, or nullptr when the walk ends in another operator.
	static unique_ptr<LogicalOperator> *
	TryWalkToCrossProduct(unique_ptr<LogicalOperator> &top,
	                      vector<reference<LogicalOperator>> &forwarding_projections) {
		const auto only_forwards = [](const LogicalOperator &op) {
			return op.type == LogicalOperatorType::LOGICAL_PROJECTION && op.children.size() == 1 &&
			       std::all_of(op.expressions.begin(), op.expressions.end(), [](const unique_ptr<Expression> &expr) {
				       return expr->type == ExpressionType::BOUND_COLUMN_REF;
			       });
		};
		auto *cross_product_ptr_ptr = &top;
		while (only_forwards(**cross_product_ptr_ptr)) {
			forwarding_projections.push_back(**cross_product_ptr_ptr);
			cross_product_ptr_ptr = &(*cross_product_ptr_ptr)->children.front();
		}
		if ((*cross_product_ptr_ptr)->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
			return nullptr;
		}
		return cross_product_ptr_ptr;
	}

	// A distance function matched to an index of the scanned table.
	struct IndexMatch {
		optional_ptr<PDXearchIndex> index;
		// The distance function's argument that holds the query vector.
		optional_ptr<Expression> query_argument;
		// When the indexed argument is a plain column: the embedding's binding as the distance function reads it, and
		// its position in the table scan.
		ColumnBinding embedding_binding;
		optional_idx embedding_column_position;
	};

	// Finds an index of the table `get` scans that `distance_expression` can use: a distance function of the index's
	// metric between the indexed expression, read through the chain (ordered from the top down to `get`), and an
	// argument `is_query_argument` accepts as the query vector.
	static bool TryMatchIndex(ClientContext &context, LogicalGet &get, const vector<reference<LogicalOperator>> &chain,
	                          const unique_ptr<Expression> &distance_expression,
	                          const std::function<bool(const Expression &)> &is_query_argument, IndexMatch &match) {
		auto &table_info = *get.GetTable()->GetStorage().GetDataTableInfo();
		vector<reference<Expression>> bindings;

		table_info.BindIndexes(context, PDXearchIndex::TYPE_NAME);
		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || PDXearchIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<PDXearchIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(distance_expression, bindings)) {
				continue;
			}
			// Check that the PDXearch index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				continue;
			}

			// Now, ensure that one of the bindings is the query vector, and the other our index expression
			auto &query_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			// The index expression is bound to the table scan, the argument reads it through the chain: compare the
			// argument traced down to the table scan.
			// Needed to support a renamed embedding column.
			// (and every case with a projection between the distance and the scan).
			const auto is_index_expression = [&](const Expression &argument) {
				if (argument.type != ExpressionType::BOUND_COLUMN_REF) {
					return index_expr->Equals(argument);
				}
				auto binding = argument.Cast<BoundColumnRefExpression>().binding;
				return TryTraceToTableScan(chain, binding) &&
				       index_expr->Equals(BoundColumnRefExpression(argument.return_type, binding));
			};

			if (!is_query_argument(query_expr_ref.get()) || !is_index_expression(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(query_expr_ref, index_expr_ref);
				if (!is_query_argument(query_expr_ref.get()) || !is_index_expression(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					continue;
				}
			}
			match.index = cast_index;
			match.query_argument = query_expr_ref.get();
			if (index_expr_ref.get().type == ExpressionType::BOUND_COLUMN_REF) {
				match.embedding_binding = index_expr_ref.get().Cast<BoundColumnRefExpression>().binding;
				auto traced_binding = match.embedding_binding;
				TryTraceToTableScan(chain, traced_binding);
				match.embedding_column_position = traced_binding.column_index;
			}
			return true;
		}
		// No index found
		return false;
	}

	// Copies a constant query vector (a FLOAT array, or a BLOB of our encoding) into a float array of the index's
	// dimension. nullptr when the BLOB's dimension differs from the index's.
	static unsafe_unique_array<float> TryDecodeConstantQuery(const PDXearchIndex &index, const Value &query) {
		const auto num_dimensions = index.GetNumDimensions();
		auto query_embedding = make_unsafe_uniq_array<float>(num_dimensions);

		if (query.type().id() == LogicalTypeId::BLOB) {
			// BLOB path: decode the quantized blob to float array
			auto blob = StringValue::Get(query);
			auto blob_dims = BlobDimensionCount(blob.size());
			if (blob_dims != num_dimensions) {
				return nullptr;
			}
			DecodeBlobToFloatArray(const_data_ptr_cast(blob.data()), blob.size(), query_embedding.get());
		} else {
			// ARRAY path: existing logic
			auto embedding_elements = ArrayValue::GetChildren(query);
			for (idx_t i = 0; i < num_dimensions; i++) {
				query_embedding[i] = embedding_elements[i].GetValue<float>();
			}
		}
		return query_embedding;
	}

	// A join above this search may drop some of the K rows the search returns, but must not decide which K rows those
	// are. DuckDB's join filter pushdown also hands the join's runtime filters to this table scan, through the TopN the
	// search replaces, so the scan is detached from them. The join keeps its own reference and fills a filter set
	// nothing reads. For example (other = {1, 3, 7, 11, 4474}):
	//
	//   SELECT id FROM (SELECT id FROM t WHERE id < 15000 ORDER BY array_distance(emb, q) LIMIT 10) s
	//   WHERE s.id IN (SELECT x FROM other);
	//
	// keeps the rows of `other` among the 10 nearest rows with id < 15000 (only 4474). With the join's filters on the
	// scan, the search would pick its neighbours among those five rows only and return all of them. The same IN inside
	// the subquery (`WHERE id < 15000 AND id IN (SELECT x FROM other)`) is a predicate of the search: the joins inside
	// the chain keep their runtime filters, moved to a filter set that only they fill.
	static void DetachDynamicFilters(LogicalGet &get, const vector<reference<LogicalOperator>> &chain) {
		auto chain_dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
		for (auto &op : chain) {
			if (op.get().type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				continue;
			}
			auto &join = op.get().Cast<LogicalComparisonJoin>();
			if (!join.filter_pushdown || !get.dynamic_filters) {
				continue;
			}
			for (auto &probe_filter : join.filter_pushdown->probe_info) {
				if (probe_filter.dynamic_filters == get.dynamic_filters) {
					probe_filter.dynamic_filters = chain_dynamic_filters;
				}
			}
		}
		get.dynamic_filters = std::move(chain_dynamic_filters);
	}

	// Builds the subtree that emits, as its only column, the rowid of every row that passes the table scan's
	// pushed-down filters and the chain (ordered from the top down to the table scan), and returns it with that rowid's
	// binding. `chain_top` and `get_ptr` are the slots of the chain's top operator and of the table scan; the subtree
	// is moved out of them.
	static unique_ptr<LogicalOperator>
	BuildRowIdChild(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &chain_top,
	                unique_ptr<LogicalOperator> &get_ptr, const vector<reference<LogicalOperator>> &chain,
	                const bool chain_filters_rows, const IndexMatch &match, ColumnBinding &rowid_binding) {
		auto &get = get_ptr->Cast<LogicalGet>();
		if (!chain_filters_rows) {
			// Scenario 2: Simple filtered search. The table scan has pushed down filters, possibly below projections
			// that forward its columns, which the search replaces. The table scan becomes the child.

			// Set the table scan's column_ids to include only those needed for the filters and the projected rowid
			// column.
			get.ClearColumnIds();
			for (auto &filter_entry : get.table_filters.filters) {
				get.AddColumnId(filter_entry.first);
			}
			// Add the rowid column (if it has not already been added because it is relied on by pushed-down filters).
			idx_t rowid_pos = DConstants::INVALID_INDEX;
			for (idx_t i = 0; i < get.GetColumnIds().size(); i++) {
				if (get.GetColumnIds()[i].GetPrimaryIndex() == COLUMN_IDENTIFIER_ROW_ID) {
					rowid_pos = i;
					break;
				}
			}
			if (rowid_pos == DConstants::INVALID_INDEX) {
				get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
				rowid_pos = get.GetColumnIds().size() - 1;
			}

			// Set the projection to only include the rowid column.
			get.projection_ids.clear();
			get.projection_ids.push_back(rowid_pos);

			// Re-resolve the get operator types after changing projection
			get.ResolveOperatorTypes();

			rowid_binding = ColumnBinding(get.table_index, rowid_pos);
			return std::move(get_ptr);
		}
		// Scenario 3: Filtered search where FILTER operators or joins between the projection and the table scan hold
		// (part of) the predicate, possibly mixed with projections that forward the table scan's columns. The table
		// scan may have pushed-down filters as well.

		// Make the rowid of every row that passes the predicate come out of the top of the chain.
		const auto chain_rowid_binding =
		    PassRowIdThroughChain(get, chain, match.embedding_binding, match.embedding_column_position);

		// Keep only that rowid on top of the chain: the filtered search's sink takes a single rowid column.
		vector<unique_ptr<Expression>> rowid_expressions;
		rowid_expressions.push_back(make_uniq<BoundColumnRefExpression>(LogicalType::ROW_TYPE, chain_rowid_binding));
		auto rowid_projection =
		    make_uniq<LogicalProjection>(input.optimizer.binder.GenerateTableIndex(), std::move(rowid_expressions));
		rowid_projection->children.push_back(std::move(chain_top));

		rowid_binding = ColumnBinding(rowid_projection->table_index, 0);
		return std::move(rowid_projection);
	}

	static bool TryOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto &context = input.context;
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		ColumnBinding distance_binding;
		if (!TryMatchOrderKey(top_n.orders, distance_binding)) {
			return false;
		}

		// find the expression that is referenced
		if (top_n.children.size() != 1) {
			return false;
		}
		auto distance_projection = TryFindDistanceProjection(*top_n.children.front(), distance_binding);
		if (!distance_projection) {
			return false;
		}
		auto &projection = *distance_projection;

		// This the expression that is referenced by the order by expression
		const auto &projection_expr = projection.expressions[distance_binding.column_index];

		// The projection must sit on top of a get, possibly with a chain of operators in between.
		if (projection.children.size() != 1) {
			return false;
		}
		vector<reference<LogicalOperator>> chain; // From the projection's child down to the table scan (excluded).
		bool chain_filters_rows = false;
		auto *get_ptr_ptr = TryWalkChainToTableScan(projection.children.front(), chain, chain_filters_rows);
		if (!get_ptr_ptr) {
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)
		auto &get_ptr = *get_ptr_ptr;
		auto &get = get_ptr->Cast<LogicalGet>();
		auto &duck_table = get.GetTable()->Cast<DuckTableEntry>();

		// Find the index: the projection expression is a distance function between our index expression and a constant
		// query vector.
		IndexMatch match;
		const auto is_constant = [](const Expression &argument) {
			return argument.type == ExpressionType::VALUE_CONSTANT;
		};
		if (!TryMatchIndex(context, get, chain, projection_expr, is_constant, match)) {
			return false;
		}
		auto query_embedding =
		    TryDecodeConstantQuery(*match.index, match.query_argument->Cast<BoundConstantExpression>().value);
		if (!query_embedding) {
			return false;
		}
		// With an OFFSET the TopN stays in the plan and skips the first `offset` of the rows we return.
		auto bind_data = make_uniq<PDXearchIndexScanBindData>(duck_table, *match.index, top_n.limit + top_n.offset,
		                                                      std::move(query_embedding));

		// The columns the index scan emits: those the projection reads. Checked before the plan is changed.
		vector<ColumnBinding> output_bindings;
		vector<ColumnIndex> output_column_ids;
		if (!TryCollectOutputColumns(projection, chain, get, output_bindings, output_column_ids)) {
			return false;
		}

		DetachDynamicFilters(get, chain);

		bool has_pushed_down_filters = !get.table_filters.filters.empty();

		if (!chain_filters_rows && !has_pushed_down_filters) {
			// Scenario 1: Non-filtered search.

			// Replace the table scan, and the projections above it that forward its columns, with the index scan.
			auto pdxearch_index_scan = make_uniq<LogicalPDXearchIndexScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    std::move(output_column_ids), std::move(output_bindings));

			projection.children.clear();
			projection.children.push_back(std::move(pdxearch_index_scan));
			projection.estimated_cardinality = top_n.estimated_cardinality;
			projection.ResolveOperatorTypes();

			// Remove the TopN operator, unless it has an OFFSET to apply.
			if (top_n.offset == 0) {
				plan = std::move(top_n.children[0]);
			}
			return true;
		} else {
			// Scenarios 2 and 3: Filtered search. The PDXearchIndexFilteredScan emits the columns the projection reads
			// (output_bindings), fetched by rowid, so the operators above it keep working unchanged. Its child emits
			// the rowid of every row that passes the predicate.
			ColumnBinding rowid_binding;
			auto rowid_child = BuildRowIdChild(input, projection.children.front(), get_ptr, chain, chain_filters_rows,
			                                   match, rowid_binding);

			// Insert a PDXearchIndexFilteredScan operator above the rowid child.
			auto pdxearch_index_filtered_scan = make_uniq<LogicalPDXearchIndexFilteredScan>(
			    duck_table, bind_data->index, bind_data->limit, std::move(bind_data->query_embedding),
			    std::move(output_column_ids), std::move(output_bindings));
			pdxearch_index_filtered_scan->expressions.push_back(
			    make_uniq<BoundColumnRefExpression>(LogicalType::ROW_TYPE, rowid_binding));
			pdxearch_index_filtered_scan->children.push_back(std::move(rowid_child));
			pdxearch_index_filtered_scan->ResolveOperatorTypes();

			projection.children.clear();
			projection.children.push_back(std::move(pdxearch_index_filtered_scan));
			projection.estimated_cardinality = top_n.estimated_cardinality;
			projection.ResolveOperatorTypes();

			// Remove the TopN operator, unless it has an OFFSET to apply.
			if (top_n.offset == 0) {
				plan = std::move(top_n.children[0]);
			}
			return true;
		}
	}

	// Runs before DuckDB's optimizers (see LATE MATERIALIZATION above): turns late materialization off for the table
	// scan of an ORDER BY that TryOptimize will turn into a search, checked as TryOptimize does up to the rewrite. The
	// constant query vector is not folded yet (e.g. a CAST of a list), so any foldable argument is accepted.
	static bool TryDisableLateMaterialization(ClientContext &context, LogicalOperator &op) {
		if (op.type != LogicalOperatorType::LOGICAL_ORDER_BY || op.children.size() != 1) {
			return false;
		}
		ColumnBinding distance_binding;
		if (!TryMatchOrderKey(op.Cast<LogicalOrder>().orders, distance_binding)) {
			return false;
		}
		auto distance_projection = TryFindDistanceProjection(*op.children.front(), distance_binding);
		if (!distance_projection || distance_projection->children.size() != 1) {
			return false;
		}
		auto &projection = *distance_projection;

		vector<reference<LogicalOperator>> chain; // From the projection's child down to the table scan (excluded).
		bool chain_filters_rows = false;
		auto *get_ptr_ptr = TryWalkChainToTableScan(projection.children.front(), chain, chain_filters_rows);
		if (!get_ptr_ptr) {
			return false;
		}
		auto &get = (*get_ptr_ptr)->Cast<LogicalGet>();

		IndexMatch match;
		const auto is_foldable = [](const Expression &argument) {
			return argument.IsFoldable();
		};
		if (!TryMatchIndex(context, get, chain, projection.expressions[distance_binding.column_index], is_foldable,
		                   match)) {
			return false;
		}
		vector<ColumnBinding> output_bindings;
		vector<ColumnIndex> output_column_ids;
		if (!TryCollectOutputColumns(projection, chain, get, output_bindings, output_column_ids)) {
			return false;
		}

		get.function.late_materialization = false;
		return true;
	}

	// Finds the top K per query of a LATERAL join (scenario 4). DuckDB turns the `ORDER BY distance LIMIT K` of the
	// subquery into a ROW_NUMBER window partitioned by the query row, and a FILTER on the row number. Its TopN window
	// elimination then makes an AGGREGATE of them, grouped by the query row: `arg_min(payload, distance, K)` (or
	// `min(distance, K)` without payload; K is left out when it is 1). With an OFFSET, the FILTER and the WINDOW stay.
	// Returns the top K's child, and fills in K, the key's binding in that child, and the groups (or partitions).
	static optional_ptr<LogicalOperator> TryMatchPerQueryTopK(LogicalOperator &op, idx_t &limit,
	                                                          ColumnBinding &key_binding,
	                                                          vector<reference<Expression>> &groups) {
		if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			auto &aggregate = op.Cast<LogicalAggregate>();
			if (aggregate.children.size() != 1 || aggregate.expressions.size() != 1 ||
			    aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty() ||
			    aggregate.expressions[0]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
				return nullptr;
			}
			auto &function = aggregate.expressions[0]->Cast<BoundAggregateExpression>();
			if (function.IsDistinct() || function.filter ||
			    (function.order_bys && !function.order_bys->orders.empty())) {
				return nullptr;
			}
			// arg_min(payload, key[, K]) or min(key[, K]). The descending order becomes arg_max or max.
			idx_t key_position;
			if (function.function.name == "arg_min" || function.function.name == "arg_min_nulls_last") {
				key_position = 1;
			} else if (function.function.name == "min") {
				key_position = 0;
			} else {
				return nullptr;
			}
			if (function.children.size() != key_position + 1 && function.children.size() != key_position + 2) {
				return nullptr;
			}
			auto &key = *function.children[key_position];
			if (key.type != ExpressionType::BOUND_COLUMN_REF) {
				return nullptr;
			}
			limit = 1;
			if (function.children.size() == key_position + 2) {
				auto &k = *function.children[key_position + 1];
				if (k.type != ExpressionType::VALUE_CONSTANT) {
					return nullptr;
				}
				auto &k_value = k.Cast<BoundConstantExpression>().value;
				if (k_value.IsNull() || k_value.type().id() != LogicalTypeId::BIGINT ||
				    k_value.GetValue<int64_t>() <= 0) {
					return nullptr;
				}
				limit = static_cast<idx_t>(k_value.GetValue<int64_t>());
			}
			key_binding = key.Cast<BoundColumnRefExpression>().binding;
			for (auto &group : aggregate.groups) {
				groups.push_back(*group);
			}
			return aggregate.children[0].get();
		}

		if (op.type != LogicalOperatorType::LOGICAL_FILTER || op.children.size() != 1) {
			return nullptr;
		}
		// Projections and FILTERs may sit between the FILTER and the WINDOW (e.g. DuckDB's debug verification
		// projections).
		vector<reference<LogicalOperator>> path;
		reference<LogicalOperator> child = *op.children[0];
		while ((child.get().type == LogicalOperatorType::LOGICAL_PROJECTION ||
		        child.get().type == LogicalOperatorType::LOGICAL_FILTER) &&
		       child.get().children.size() == 1) {
			path.push_back(child);
			child = *child.get().children[0];
		}
		if (child.get().type != LogicalOperatorType::LOGICAL_WINDOW || child.get().children.size() != 1) {
			return nullptr;
		}
		auto &window = child.get().Cast<LogicalWindow>();
		if (window.expressions.size() != 1 || window.expressions[0]->type != ExpressionType::WINDOW_ROW_NUMBER) {
			return nullptr;
		}
		auto &row_number = window.expressions[0]->Cast<BoundWindowExpression>();
		if (row_number.orders.size() != 1 || !row_number.arg_orders.empty() || row_number.filter_expr ||
		    row_number.distinct) {
			return nullptr;
		}
		auto &order = row_number.orders[0];
		if (order.type != OrderType::ASCENDING || order.null_order != OrderByNullType::NULLS_LAST ||
		    order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			return nullptr;
		}
		// K is the smallest bound on the row number (`rownum <= K`, `rownum < K + 1` or `rownum = K`). The K nearest
		// rows of each query get the row numbers 1 to K they get among all rows, so the FILTER's other conditions
		// (e.g. the OFFSET's `rownum > o`) stay as they are.
		const ColumnBinding row_number_binding(window.window_index, 0);
		optional_idx smallest_bound;
		for (auto &condition : op.expressions) {
			if (condition->GetExpressionClass() != ExpressionClass::BOUND_COMPARISON) {
				continue;
			}
			auto &comparison = condition->Cast<BoundComparisonExpression>();
			if (comparison.left->type != ExpressionType::BOUND_COLUMN_REF ||
			    comparison.right->type != ExpressionType::VALUE_CONSTANT) {
				continue;
			}
			auto binding = comparison.left->Cast<BoundColumnRefExpression>().binding;
			auto &bound_value = comparison.right->Cast<BoundConstantExpression>().value;
			if (!TryTraceToTableScan(path, binding) || binding != row_number_binding || bound_value.IsNull() ||
			    bound_value.type().id() != LogicalTypeId::BIGINT) {
				continue;
			}
			auto bound = bound_value.GetValue<int64_t>();
			if (comparison.type == ExpressionType::COMPARE_LESSTHAN) {
				bound -= 1;
			} else if (comparison.type != ExpressionType::COMPARE_LESSTHANOREQUALTO &&
			           comparison.type != ExpressionType::COMPARE_EQUAL) {
				continue;
			}
			if (bound <= 0) {
				return nullptr;
			}
			if (!smallest_bound.IsValid() || static_cast<idx_t>(bound) < smallest_bound.GetIndex()) {
				smallest_bound = static_cast<idx_t>(bound);
			}
		}
		if (!smallest_bound.IsValid()) {
			return nullptr;
		}
		limit = smallest_bound.GetIndex();
		key_binding = order.expression->Cast<BoundColumnRefExpression>().binding;
		for (auto &partition : row_number.partitions) {
			groups.push_back(*partition);
		}
		return window.children[0].get();
	}

	// Whether every group (or partition) of the top K is a column of the query rows, traced through the projections
	// from the top K's child down to the distance projection, and the forwarding projections below it. A group then
	// holds whole query rows, and its top K is among the K nearest rows of those query rows, which the index join
	// emits.
	static bool GroupsAreQueryRowColumns(const vector<reference<Expression>> &groups, LogicalOperator &top_k_child,
	                                     const LogicalProjection &distance_projection,
	                                     const vector<reference<LogicalOperator>> &forwarding_projections,
	                                     const idx_t query_rows_table_index) {
		vector<reference<LogicalOperator>> path;
		for (reference<LogicalOperator> op = top_k_child;; op = *op.get().children[0]) {
			path.push_back(op);
			if (&op.get() == &distance_projection) {
				break;
			}
		}
		path.insert(path.end(), forwarding_projections.begin(), forwarding_projections.end());
		for (auto &group : groups) {
			if (group.get().type != ExpressionType::BOUND_COLUMN_REF) {
				return false;
			}
			auto binding = group.get().Cast<BoundColumnRefExpression>().binding;
			if (!TryTraceToTableScan(path, binding) || binding.table_index != query_rows_table_index) {
				return false;
			}
		}
		return true;
	}

	// Scenario 4: a LATERAL join searching the table once per query row. Replaces the cross product between the table
	// and the query rows, below the projection that computes the distance, with a PDXearchIndexJoin.
	static bool TryOptimizeIndexJoin(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		idx_t limit = 0;
		ColumnBinding key_binding;
		vector<reference<Expression>> groups;
		auto top_k_child = TryMatchPerQueryTopK(*plan, limit, key_binding, groups);
		// A query's K nearest rows are emitted as one chunk.
		if (!top_k_child || limit > STANDARD_VECTOR_SIZE) {
			return false;
		}

		auto distance_projection = TryFindDistanceProjection(*top_k_child, key_binding);
		if (!distance_projection) {
			return false;
		}
		auto &projection = *distance_projection;
		if (projection.children.size() != 1) {
			return false;
		}
		vector<reference<LogicalOperator>> forwarding_projections; // Between the projection and the cross product.
		auto *cross_product_ptr_ptr = TryWalkToCrossProduct(projection.children.front(), forwarding_projections);
		if (!cross_product_ptr_ptr) {
			return false;
		}
		auto &cross_product_ptr = *cross_product_ptr_ptr;
		auto &cross_product = *cross_product_ptr;
		// The projection that reads the cross product's columns, which the index join emits.
		auto &cross_product_reader =
		    forwarding_projections.empty() ? projection : forwarding_projections.back().get().Cast<LogicalProjection>();

		// One side of the cross product holds the query rows: the DELIM_GET of the LATERAL join, which DuckDB fills
		// with the distinct query rows. The other side is the table, possibly below a chain (the subquery's predicate).
		idx_t query_rows_side;
		if (cross_product.children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			query_rows_side = 0;
		} else if (cross_product.children[1]->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			query_rows_side = 1;
		} else {
			return false;
		}
		const auto query_rows_table_index =
		    cross_product.children[query_rows_side]->Cast<LogicalDelimGet>().table_index;
		auto &table_side = cross_product.children[1 - query_rows_side];

		vector<reference<LogicalOperator>> chain; // From the table side's top down to the table scan (excluded).
		bool chain_filters_rows = false;
		auto *get_ptr_ptr = TryWalkChainToTableScan(table_side, chain, chain_filters_rows);
		if (!get_ptr_ptr) {
			return false;
		}
		auto &get = (*get_ptr_ptr)->Cast<LogicalGet>();
		auto &duck_table = get.GetTable()->Cast<DuckTableEntry>();

		// The distance is between our index expression and a column of the query rows, which its arguments read
		// through the forwarding projections (and the index expression then through the table side's chain).
		vector<reference<LogicalOperator>> distance_chain = forwarding_projections;
		distance_chain.insert(distance_chain.end(), chain.begin(), chain.end());
		IndexMatch match;
		const auto is_query_column = [&](const Expression &argument) {
			if (argument.type != ExpressionType::BOUND_COLUMN_REF) {
				return false;
			}
			auto binding = argument.Cast<BoundColumnRefExpression>().binding;
			return TryTraceToTableScan(forwarding_projections, binding) &&
			       binding.table_index == query_rows_table_index;
		};
		if (!TryMatchIndex(input.context, get, distance_chain, projection.expressions[key_binding.column_index],
		                   is_query_column, match)) {
			return false;
		}
		// The query vector's and the embedding's bindings among the cross product's columns.
		auto query_binding = match.query_argument->Cast<BoundColumnRefExpression>().binding;
		TryTraceToTableScan(forwarding_projections, query_binding);
		if (match.embedding_column_position.IsValid()) {
			TryTraceToTableScan(forwarding_projections, match.embedding_binding);
		}

		if (!GroupsAreQueryRowColumns(groups, *top_k_child, projection, forwarding_projections,
		                              query_rows_table_index)) {
			return false;
		}

		// The table columns the index join fetches: those the projection above the cross product reads. The query rows'
		// columns pass through.
		vector<ColumnBinding> output_bindings;
		vector<ColumnIndex> output_column_ids;
		if (!TryCollectOutputColumns(cross_product_reader, chain, get, output_bindings, output_column_ids,
		                             query_rows_table_index)) {
			return false;
		}

		DetachDynamicFilters(get, chain);
		const bool is_filtered = chain_filters_rows || !get.table_filters.filters.empty();

		auto index_join = make_uniq<LogicalPDXearchIndexJoin>(duck_table, *match.index, limit, query_binding,
		                                                      std::move(output_column_ids), std::move(output_bindings));
		const auto num_query_rows = cross_product.children[query_rows_side]->EstimateCardinality(input.context);
		index_join->children.push_back(std::move(cross_product.children[query_rows_side]));
		if (is_filtered) {
			// The rowids of the rows that pass the predicate, as the filtered search consumes them (scenarios 2, 3).
			ColumnBinding rowid_binding;
			index_join->children.push_back(
			    BuildRowIdChild(input, table_side, *get_ptr_ptr, chain, chain_filters_rows, match, rowid_binding));
		}
		index_join->ResolveOperatorTypes();
		index_join->SetEstimatedCardinality(num_query_rows * limit);

		// Replaces the cross product (and, without a predicate, the table scan).
		cross_product_ptr = std::move(index_join);
		return true;
	}

	static bool OptimizeChildren(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto ok = TryOptimize(input, plan) || TryOptimizeIndexJoin(input, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(input, child);
		}
		return ok;
	}

	static column_binding_set_t CollectReadBindings(LogicalProjection &projection) {
		column_binding_set_t read_bindings;
		for (auto &expression : projection.expressions) {
			ExpressionIterator::EnumerateExpression(expression, [&](Expression &child) {
				if (child.type == ExpressionType::BOUND_COLUMN_REF) {
					read_bindings.insert(child.Cast<BoundColumnRefExpression>().binding);
				}
			});
		}
		return read_bindings;
	}

	// Replaces each expression of `projection` that `reader`, the projection above it, does not read with a constant,
	// so it is no longer computed. The expressions stay: a projection's bindings are positions in its expression list.
	static void DropUnreadExpressions(LogicalProjection &reader, LogicalProjection &projection) {
		const auto read_bindings = CollectReadBindings(reader);
		const auto bindings = projection.GetColumnBindings();
		for (idx_t i = 0; i < projection.expressions.size(); i++) {
			if (read_bindings.find(bindings[i]) == read_bindings.end()) {
				projection.expressions[i] =
				    make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
			}
		}
	}

	// Keeps only the columns of an index scan that `projection` reads: once MergeProjections has dropped an unread
	// distance, the embedding is no longer fetched. Any change to the binding list is safe: the projection resolves
	// its column references by binding.
	template <class INDEX_SCAN>
	static void DropUnreadColumns(INDEX_SCAN &index_scan, LogicalProjection &projection) {
		const auto read_bindings = CollectReadBindings(projection);
		vector<ColumnIndex> column_ids;
		vector<ColumnBinding> column_bindings;
		for (idx_t i = 0; i < index_scan.column_ids.size(); i++) {
			if (read_bindings.find(index_scan.column_bindings[i]) != read_bindings.end()) {
				column_ids.push_back(index_scan.column_ids[i]);
				column_bindings.push_back(index_scan.column_bindings[i]);
			}
		}
		// Still fetch the rowid, under a binding nothing reads: the fetch drops the rows the transaction cannot see,
		// and a rowid is computed without reading a block.
		if (column_ids.empty() && !index_scan.column_ids.empty()) {
			column_ids.emplace_back(COLUMN_IDENTIFIER_ROW_ID);
			column_bindings.push_back(index_scan.column_bindings[0]);
		}
		index_scan.column_ids = std::move(column_ids);
		index_scan.column_bindings = std::move(column_bindings);
		index_scan.ResolveOperatorTypes();
	}

	// Drops what the projections above an index scan compute or fetch for nothing: the distance that the removed TopN
	// read, and the embedding only that distance read. The top projection of the run stays as is, its outputs may be
	// read by position (the query result, a UNION); any other projection is only read by the one above it (e.g. those
	// a view or a subquery adds).
	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		vector<reference<LogicalProjection>> projections;
		reference<LogicalOperator> below = *plan;
		while (below.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
			projections.push_back(below.get().Cast<LogicalProjection>());
			below = *below.get().children[0];
		}
		auto &op = below.get();
		const auto is_index_scan =
		    op.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR &&
		    (op.GetName() == "PDXEARCH_INDEX_SCAN" || op.GetName() == "PDXEARCH_INDEX_FILT_SCAN");
		if (projections.size() < 2 || !is_index_scan) {
			for (auto &child : op.children) {
				MergeProjections(child);
			}
			return;
		}
		for (idx_t i = 1; i < projections.size(); i++) {
			DropUnreadExpressions(projections[i - 1], projections[i]);
		}
		if (op.GetName() == "PDXEARCH_INDEX_SCAN") {
			DropUnreadColumns(op.Cast<LogicalPDXearchIndexScan>(), projections.back());
		} else {
			DropUnreadColumns(op.Cast<LogicalPDXearchIndexFilteredScan>(), projections.back());
		}
	}

	static void PreOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		TryDisableLateMaterialization(input.context, *plan);
		for (auto &child : plan->children) {
			PreOptimize(input, child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_pdxearch_scan = OptimizeChildren(input, plan);
		if (did_use_pdxearch_scan) {
			MergeProjections(plan);
		}
	}
};

void PDXearchModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	OptimizerExtension::Register(db.config, PDXearchIndexScanOptimizer());
}

} // namespace duckdb

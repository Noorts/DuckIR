#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"

#include "index/pdxearch_index.hpp"
#include "index/pdxearch_module.hpp"

namespace duckdb {

struct IndexInfoColumnInput {
	const IndexCatalogEntry &index_entry;
	const TableCatalogEntry &table_entry;
	const PDXearchIndexStats &stats;
};

struct IndexInfoColumn {
	const char *name;
	LogicalType type;
	std::function<Value(const IndexInfoColumnInput &)> get_value;
};

// Defines the columns included in `CALL pdxearch_index_info();`.
// Make sure to sync any changes made here to `PDXearchIndex::GetStats`.
static const IndexInfoColumn INDEX_INFO_COLUMNS[] = {
    {"catalog_name", LogicalType::VARCHAR,
     [](const IndexInfoColumnInput &input) {
	     return Value(input.index_entry.catalog.GetName());
     }},
    {"schema_name", LogicalType::VARCHAR,
     [](const IndexInfoColumnInput &input) {
	     return Value(input.index_entry.schema.name);
     }},
    {"index_name", LogicalType::VARCHAR,
     [](const IndexInfoColumnInput &input) {
	     return Value(input.index_entry.name);
     }},
    {"table_name", LogicalType::VARCHAR,
     [](const IndexInfoColumnInput &input) {
	     return Value(input.table_entry.name);
     }},
    {"metric", LogicalType::VARCHAR,
     [](const IndexInfoColumnInput &input) {
	     return Value(input.stats.metric);
     }},
    {"num_dimensions", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.num_dimensions);
     }},
    {"quantization", LogicalType::VARCHAR,
     [](const IndexInfoColumnInput &input) {
	     return Value(input.stats.quantization);
     }},
    {"n_probe", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.n_probe);
     }},
    {"seed", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.seed);
     }},
    {"is_normalized", LogicalType::BOOLEAN,
     [](const IndexInfoColumnInput &input) {
	     return Value::BOOLEAN(input.stats.is_normalized);
     }},
    {"approx_lower_bound_memory_usage_bytes", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.approximate_lower_bound_memory_usage_bytes);
     }},
    {"cluster_acquires", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.cluster_acquires);
     }},
    {"cluster_cache_misses", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.cluster_cache_misses);
     }},
    {"cluster_bytes_fetched", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.cluster_bytes_fetched);
     }},
    {"blocks_read", LogicalType::BIGINT,
     [](const IndexInfoColumnInput &input) {
	     return Value::BIGINT(input.stats.blocks_read);
     }},
};

static unique_ptr<FunctionData> PDXearchIndexInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	for (const auto &col : INDEX_INFO_COLUMNS) {
		names.emplace_back(col.name);
		return_types.push_back(col.type);
	}
	return nullptr;
}

struct PDXearchIndexInfoGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	vector<reference<IndexCatalogEntry>> entries;
};

static unique_ptr<GlobalTableFunctionState> PDXearchIndexInfoInitGlobal(ClientContext &context,
                                                                        TableFunctionInitInput &input) {
	auto result = make_uniq<PDXearchIndexInfoGlobalState>();

	// scan all the schemas for indexes and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type == PDXearchIndex::TYPE_NAME) {
				result->entries.push_back(index_entry);
			}
		});
	};
	return std::move(result);
}

static void PDXearchIndexInfoExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<PDXearchIndexInfoGlobalState>();
	if (data.offset >= data.entries.size()) {
		return;
	}

	idx_t row = 0;
	while (data.offset < data.entries.size() && row < STANDARD_VECTOR_SIZE) {
		auto &index_entry = data.entries[data.offset++].get();
		auto &table_entry = index_entry.schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry.GetSchemaName(),
		                                                                           index_entry.GetTableName());
		auto &storage = table_entry.GetStorage();
		PDXearchIndex *pdxearch_index = nullptr;

		auto &table_info = *storage.GetDataTableInfo();

		table_info.BindIndexes(context, PDXearchIndex::TYPE_NAME);
		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || PDXearchIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<PDXearchIndex>();
			if (cast_index.name == index_entry.name) {
				pdxearch_index = &cast_index;
				break;
			}
			continue;
		};

		if (!pdxearch_index) {
			throw BinderException("Index %s not found", index_entry.name);
		}

		auto stats = pdxearch_index->GetStats(context);
		IndexInfoColumnInput col_input {index_entry, table_entry, *stats};

		for (idx_t col = 0; col < sizeof(INDEX_INFO_COLUMNS) / sizeof(INDEX_INFO_COLUMNS[0]); col++) {
			output.data[col].SetValue(row, INDEX_INFO_COLUMNS[col].get_value(col_input));
		}

		row++;
	}
	output.SetCardinality(row);
}

void PDXearchModule::RegisterIndexInfo(ExtensionLoader &loader) {
	TableFunction info_function("pdxearch_index_info", {}, PDXearchIndexInfoExecute, PDXearchIndexInfoBind,
	                            PDXearchIndexInfoInitGlobal);
	loader.RegisterFunction(info_function);
}

} // namespace duckdb

#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"

#include "index/vss_index.hpp"
#include "index/vss_module.hpp"

namespace duckdb {

struct IndexInfoColumnInput {
	const IndexCatalogEntry &index_entry;
	const TableCatalogEntry &table_entry;
	const VSSIndexStats &stats;
};

struct IndexInfoColumn {
	const char *name;
	LogicalType type;
	std::function<Value(const IndexInfoColumnInput &)> get_value;
};

// Defines the columns included in `CALL duckir_vss_index_info();`.
// Make sure to sync any changes made here to `VSSIndex::GetStats`.
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

static unique_ptr<FunctionData> VSSIndexInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<Identifier> &names) {
	for (const auto &col : INDEX_INFO_COLUMNS) {
		names.emplace_back(col.name);
		return_types.push_back(col.type);
	}
	return nullptr;
}

struct VSSIndexInfoGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	vector<reference<IndexCatalogEntry>> entries;
};

static unique_ptr<GlobalTableFunctionState> VSSIndexInfoInitGlobal(ClientContext &context,
                                                                   TableFunctionInitInput &input) {
	auto result = make_uniq<VSSIndexInfoGlobalState>();

	// scan all the schemas for indexes and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type == VSSIndex::TYPE_NAME) {
				result->entries.push_back(index_entry);
			}
		});
	};
	return std::move(result);
}

static void VSSIndexInfoExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<VSSIndexInfoGlobalState>();
	if (data.offset >= data.entries.size()) {
		return;
	}

	idx_t row = 0;
	while (data.offset < data.entries.size() && row < STANDARD_VECTOR_SIZE) {
		auto &index_entry = data.entries[data.offset++].get();
		auto &table_entry = index_entry.schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry.GetSchemaName(),
		                                                                           index_entry.GetTableName());
		auto &storage = table_entry.GetStorage();
		auto &table_info = *storage.GetDataTableInfo();

		table_info.BindIndexes(context, VSSIndex::TYPE_NAME);
		unique_ptr<VSSIndexStats> stats;
		for (auto entry : table_info.GetIndexes().IndexEntries()) {
			if (entry->GetBindState() != IndexBindState::BOUND || entry->GetIndexType() != VSSIndex::TYPE_NAME ||
			    entry->GetName() != index_entry.name) {
				continue;
			}
			auto index = entry->GetReadHandle<VSSIndex>();
			stats = index->GetStats(context);
			break;
		}

		if (!stats) {
			throw BinderException("Index %s not found", index_entry.name);
		}
		IndexInfoColumnInput col_input {index_entry, table_entry, *stats};

		for (idx_t col = 0; col < sizeof(INDEX_INFO_COLUMNS) / sizeof(INDEX_INFO_COLUMNS[0]); col++) {
			output.data[col].Append(INDEX_INFO_COLUMNS[col].get_value(col_input));
		}

		row++;
	}
	output.CheckCardinality(row);
}

void VSSModule::RegisterIndexInfo(ExtensionLoader &loader) {
	TableFunction info_function("duckir_vss_index_info", {}, VSSIndexInfoExecute, VSSIndexInfoBind,
	                            VSSIndexInfoInitGlobal);
	loader.RegisterFunction(info_function);
}

} // namespace duckdb

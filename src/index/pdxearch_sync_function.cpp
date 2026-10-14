#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"

#include "index/pdxearch_index.hpp"
#include "index/pdxearch_module.hpp"

namespace duckdb {

// `CALL pdxearch_sync_index('index_name');` indexes the rows committed since the last search, which a search would
// otherwise do at its start.

struct PDXearchSyncIndexBindData : public TableFunctionData {
	string index_name;
};

struct PDXearchSyncIndexGlobalState : public GlobalTableFunctionState {
	bool done = false;
};

static unique_ptr<FunctionData> PDXearchSyncIndexBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<PDXearchSyncIndexBindData>();
	result->index_name = input.inputs[0].GetValue<string>();
	names.emplace_back("index_name");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("had_unindexed_rows");
	return_types.push_back(LogicalType::BOOLEAN);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> PDXearchSyncIndexInitGlobal(ClientContext &context,
                                                                        TableFunctionInitInput &input) {
	return make_uniq<PDXearchSyncIndexGlobalState>();
}

static PDXearchIndex &FindPDXearchIndex(ClientContext &context, const string &index_name,
                                        optional_ptr<DataTable> &table) {
	optional_ptr<IndexCatalogEntry> index_entry;
	for (auto &schema : Catalog::GetAllSchemas(context)) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &candidate = entry.Cast<IndexCatalogEntry>();
			if (candidate.index_type == PDXearchIndex::TYPE_NAME && candidate.name == index_name) {
				index_entry = &candidate;
			}
		});
	}
	if (!index_entry) {
		throw BinderException("PDXearch index '%s' not found", index_name);
	}

	auto &table_entry = index_entry->schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry->GetSchemaName(),
	                                                                            index_entry->GetTableName());
	table = &table_entry.GetStorage();
	auto &table_info = *table_entry.GetStorage().GetDataTableInfo();
	table_info.BindIndexes(context, PDXearchIndex::TYPE_NAME);
	for (auto &index : table_info.GetIndexes().Indexes()) {
		if (index.IsBound() && index.GetIndexType() == PDXearchIndex::TYPE_NAME && index.GetIndexName() == index_name) {
			return index.Cast<PDXearchIndex>();
		}
	}
	throw BinderException("PDXearch index '%s' is not bound", index_name);
}

static void PDXearchSyncIndexExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<PDXearchSyncIndexGlobalState>();
	if (state.done) {
		return;
	}
	auto &bind_data = data_p.bind_data->Cast<PDXearchSyncIndexBindData>();
	optional_ptr<DataTable> table;
	auto &index = FindPDXearchIndex(context, bind_data.index_name, table);
	const bool had_unindexed_rows = index.HasUnindexedRows();
	index.SyncWithTable(context, *table);

	output.SetValue(0, 0, Value(bind_data.index_name));
	output.SetValue(1, 0, Value::BOOLEAN(had_unindexed_rows));
	output.SetCardinality(1);
	state.done = true;
}

void PDXearchModule::RegisterSyncIndex(ExtensionLoader &loader) {
	TableFunction sync_function("pdxearch_sync_index", {LogicalType::VARCHAR}, PDXearchSyncIndexExecute,
	                            PDXearchSyncIndexBind, PDXearchSyncIndexInitGlobal);
	loader.RegisterFunction(sync_function);
}

} // namespace duckdb

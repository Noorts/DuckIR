#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"

#include "index/vss_index.hpp"
#include "index/vss_module.hpp"

namespace duckdb {

// `CALL duckir_vss_sync_index('index_name');` indexes the rows committed since the last search, which a search would
// otherwise do at its start.

struct VSSSyncIndexBindData : public TableFunctionData {
	string index_name;
};

struct VSSSyncIndexGlobalState : public GlobalTableFunctionState {
	bool done = false;
};

static unique_ptr<FunctionData> VSSSyncIndexBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<VSSSyncIndexBindData>();
	result->index_name = input.inputs[0].GetValue<string>();
	names.emplace_back("index_name");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("had_unindexed_rows");
	return_types.push_back(LogicalType::BOOLEAN);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> VSSSyncIndexInitGlobal(ClientContext &context,
                                                                   TableFunctionInitInput &input) {
	return make_uniq<VSSSyncIndexGlobalState>();
}

static VSSIndex &FindVSSIndex(ClientContext &context, const string &index_name, optional_ptr<DataTable> &table) {
	optional_ptr<IndexCatalogEntry> index_entry;
	for (auto &schema : Catalog::GetAllSchemas(context)) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &candidate = entry.Cast<IndexCatalogEntry>();
			if (candidate.index_type == VSSIndex::TYPE_NAME && candidate.name == index_name) {
				index_entry = &candidate;
			}
		});
	}
	if (!index_entry) {
		throw BinderException("DuckIR VSS index '%s' not found", index_name);
	}

	auto &table_entry = index_entry->schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry->GetSchemaName(),
	                                                                            index_entry->GetTableName());
	table = &table_entry.GetStorage();
	auto &table_info = *table_entry.GetStorage().GetDataTableInfo();
	table_info.BindIndexes(context, VSSIndex::TYPE_NAME);
	for (auto &index : table_info.GetIndexes().Indexes()) {
		if (index.IsBound() && index.GetIndexType() == VSSIndex::TYPE_NAME && index.GetIndexName() == index_name) {
			return index.Cast<VSSIndex>();
		}
	}
	throw BinderException("DuckIR VSS index '%s' is not bound", index_name);
}

static void VSSSyncIndexExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<VSSSyncIndexGlobalState>();
	if (state.done) {
		return;
	}
	auto &bind_data = data_p.bind_data->Cast<VSSSyncIndexBindData>();
	optional_ptr<DataTable> table;
	auto &index = FindVSSIndex(context, bind_data.index_name, table);
	const bool had_unindexed_rows = index.HasUnindexedRows();
	index.SyncWithTable(context, *table);

	output.SetValue(0, 0, Value(bind_data.index_name));
	output.SetValue(1, 0, Value::BOOLEAN(had_unindexed_rows));
	output.SetCardinality(1);
	state.done = true;
}

void VSSModule::RegisterSyncIndex(ExtensionLoader &loader) {
	TableFunction sync_function("duckir_vss_sync_index", {LogicalType::VARCHAR}, VSSSyncIndexExecute, VSSSyncIndexBind,
	                            VSSSyncIndexInitGlobal);
	loader.RegisterFunction(sync_function);
}

} // namespace duckdb

#pragma once

#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

struct VSSModule {
public:
	static void Register(ExtensionLoader &loader) {
		auto &db = loader.GetDatabaseInstance();

		RegisterIndex(db);

		RegisterScanOptimizer(db);

		RegisterIndexInfo(loader);
		RegisterSyncIndex(loader);
		RegisterBlobFunctions(loader);
	}

private:
	static void RegisterIndex(DatabaseInstance &ldb);

	static void RegisterScanOptimizer(DatabaseInstance &db);

	static void RegisterIndexInfo(ExtensionLoader &loader);
	static void RegisterSyncIndex(ExtensionLoader &loader);
	static void RegisterBlobFunctions(ExtensionLoader &loader);
};

} // namespace duckdb

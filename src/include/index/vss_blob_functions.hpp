#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

struct VSSBlobFunctions {
	static void Register(ExtensionLoader &loader);
};

} // namespace duckdb

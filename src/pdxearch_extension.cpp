#include "pdxearch_extension.hpp"

#include "index/pdxearch_module.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	PDXearchModule::Register(loader);
}

void PdxearchExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string PdxearchExtension::Name() {
	return "pdxearch";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(pdxearch, loader) {
	duckdb::LoadInternal(loader);
}
}

#include "duckir_extension.hpp"

#include "index/vss_module.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	VSSModule::Register(loader);
}

void DuckirExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string DuckirExtension::Name() {
	return "duckir";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckir, loader) {
	duckdb::LoadInternal(loader);
}
}

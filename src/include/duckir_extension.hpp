#pragma once

#include "duckdb/main/extension.hpp"

namespace duckdb {

class DuckirExtension : public Extension {
public:
	void Load(ExtensionLoader &db) override;
	std::string Name() override;
};

} // namespace duckdb

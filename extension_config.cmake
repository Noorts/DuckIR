# This file is included by DuckDB's build system. It specifies which extension to load

# Force CXX_STANDARD from 11 to 17 as PDXearch currently uses C++17 features.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to enforce" FORCE)

# Before pdxearch: a binary with the extensions linked in loads them in this order, and pdxearch adds overloads to
# core_functions' distance functions (e.g. array_distance) when it loads.
duckdb_extension_load(core_functions)

# Extension from this repo
duckdb_extension_load(pdxearch
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Any extra extensions that should be built
# e.g.: duckdb_extension_load(json)
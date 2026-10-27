#pragma once

#include "duckdb/common/types.hpp"

#include <cstring>

namespace duckdb {

// A query vector as a BLOB: its float32 values, little-endian, without a header (4 bytes per dimension).

inline void EncodeFloatArrayToBlob(const float *input, idx_t count, data_ptr_t output) {
	std::memcpy(output, input, count * sizeof(float));
}

inline void DecodeBlobToFloatArray(const_data_ptr_t input, idx_t blob_size, float *output) {
	std::memcpy(output, input, blob_size);
}

} // namespace duckdb

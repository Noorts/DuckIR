#pragma once

#include "duckdb/common/types.hpp"

#include <cmath>
#include <cstdint>

namespace duckdb {

// Lossy int16 quantization for float vectors.
//
// Encoding: each float -> int16_t
//   Lower 2 bits = X (scale selector, 0..3)
//   Upper 14 bits = quantized value (signed, [-8192, 8191])
//   Decode: float = (int16 >> 2) * BLOB_SCALE[int16 & 3]
//   Encode: pick smallest X where round(float * inv_scale[X]) fits in 14-bit signed range
//
// Blob size = N * 2 bytes (no header).

constexpr float BLOB_SCALE[4] = {0.00001f, 0.0001f, 0.001f, 0.01f};
constexpr float BLOB_INV_SCALE[4] = {100000.0f, 10000.0f, 1000.0f, 100.0f};

inline int16_t EncodeFloatToInt16(float value) {
	for (int x = 0; x < 4; x++) {
		auto q = static_cast<int32_t>(std::round(value * BLOB_INV_SCALE[x]));
		if (q >= -8192 && q <= 8191) {
			return static_cast<int16_t>((q << 2) | x);
		}
	}
	// Clamp to max range at largest scale
	auto q = static_cast<int32_t>(std::round(value * BLOB_INV_SCALE[3]));
	if (q > 8191) {
		q = 8191;
	} else if (q < -8192) {
		q = -8192;
	}
	return static_cast<int16_t>((q << 2) | 3);
}

inline float DecodeInt16ToFloat(int16_t encoded) {
	int x = encoded & 3;
	int32_t q = encoded >> 2; // arithmetic right shift
	return static_cast<float>(q) * BLOB_SCALE[x];
}

inline void EncodeFloatArrayToBlob(const float *input, idx_t count, data_ptr_t output) {
	auto *out = reinterpret_cast<int16_t *>(output);
	for (idx_t i = 0; i < count; i++) {
		out[i] = EncodeFloatToInt16(input[i]);
	}
}

inline void DecodeBlobToFloatArray(const_data_ptr_t input, idx_t blob_size, float *output) {
	idx_t count = blob_size / 2;
	auto *in = reinterpret_cast<const int16_t *>(input);
	for (idx_t i = 0; i < count; i++) {
		output[i] = DecodeInt16ToFloat(in[i]);
	}
}

inline idx_t BlobDimensionCount(idx_t blob_byte_size) {
	return blob_byte_size / 2;
}

} // namespace duckdb

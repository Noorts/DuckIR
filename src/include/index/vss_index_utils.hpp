#pragma once

#include <Eigen/Dense>
#include <random>

#include "duckdb/common/string_util.hpp"
#include "duckdb/storage/storage_info.hpp"

namespace duckdb {

// Generate a rotation matrix suitable for PDXearch's ADSampling pruning algorithm.
//
// Based on https://github.com/cwida/PDX/blob/main/python/pdxearch/preprocessors.py#L39
//
// TODO: Only worth it above 4096 dimensions: a blocked Householder QR on skmeans::Sgemm (LAPACK's sgeqrf + sorgqr)
// builds the same matrix 4-8x faster than Eigen.
[[nodiscard]] inline unique_ptr<float[]> GenerateRandomRotationMatrix(const size_t num_dimensions, const int32_t seed) {
	auto rotation_matrix = make_uniq_array<float>(num_dimensions * num_dimensions);

	std::mt19937 gen(seed);
	std::normal_distribution<float> normal_dist;

	Eigen::MatrixXf random_matrix {
	    Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_dimensions), static_cast<Eigen::Index>(num_dimensions))};
	for (idx_t i = 0; i < num_dimensions; ++i) {
		for (idx_t j = 0; j < num_dimensions; ++j) {
			random_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = normal_dist(gen);
		}
	}

	const Eigen::HouseholderQR<Eigen::MatrixXf> qr {random_matrix};
	const Eigen::MatrixXf transformation_matrix {qr.householderQ()};

	for (idx_t i = 0; i < num_dimensions; ++i) {
		for (idx_t j = 0; j < num_dimensions; ++j) {
			rotation_matrix[i * num_dimensions + j] =
			    transformation_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
		}
	}

	return rotation_matrix;
}

// The rotation of a u4 index before its SAQ fit: its rows stay as they are until the fit's rotation replaces it.
[[nodiscard]] inline unique_ptr<float[]> GenerateIdentityRotationMatrix(const size_t num_dimensions) {
	auto rotation_matrix = make_uniq_array<float>(num_dimensions * num_dimensions);
	Eigen::Map<Eigen::MatrixXf>(rotation_matrix.get(), static_cast<Eigen::Index>(num_dimensions),
	                            static_cast<Eigen::Index>(num_dimensions))
	    .setIdentity();
	return rotation_matrix;
}

[[nodiscard]] inline string ConvertBytesToHumanReadableString(const uint64_t bytes) {
	constexpr uint64_t kB = 1000;
	constexpr uint64_t MB = kB * 1000;
	constexpr uint64_t GB = MB * 1000;
	constexpr uint64_t TB = GB * 1000;

	if (bytes >= TB) {
		return StringUtil::Format("%.2f TB", static_cast<double>(bytes) / static_cast<double>(TB));
	} else if (bytes >= GB) {
		return StringUtil::Format("%.2f GB", static_cast<double>(bytes) / static_cast<double>(GB));
	} else if (bytes >= MB) {
		return StringUtil::Format("%.2f MB", static_cast<double>(bytes) / static_cast<double>(MB));
	} else if (bytes >= kB) {
		return StringUtil::Format("%.2f kB", static_cast<double>(bytes) / static_cast<double>(kB));
	} else {
		return StringUtil::Format("%llu bytes", bytes);
	}
}

} // namespace duckdb

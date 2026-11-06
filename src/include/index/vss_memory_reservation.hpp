#pragma once

#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

class VSSMemoryReservation {
public:
	explicit VSSMemoryReservation(BufferManager &buffer_manager) : buffer_manager(buffer_manager) {
	}
	~VSSMemoryReservation() {
		buffer_manager.FreeReservedMemory(reserved_bytes);
	}
	VSSMemoryReservation(const VSSMemoryReservation &) = delete;
	VSSMemoryReservation &operator=(const VSSMemoryReservation &) = delete;

	// Evicts other buffers first, or throws DuckDB's out-of-memory error.
	void Reserve(const idx_t bytes) {
		buffer_manager.ReserveMemory(bytes);
		reserved_bytes += bytes;
	}

private:
	BufferManager &buffer_manager;
	idx_t reserved_bytes = 0;
};

} // namespace duckdb

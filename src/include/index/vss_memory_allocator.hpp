#pragma once

#include <new>

#include "duckdb/common/allocator.hpp"
#include "pdx/common.hpp"

namespace duckdb {

// PDX's and SuperKMeans' large buffers on a DuckDB allocator. DuckDB does not allocate 0 bytes, while PDX asks for them
// and needs a non-null pointer: those come from the heap.
class DuckDBMemoryAllocator final : public PDX::MemoryAllocator {
public:
	explicit DuckDBMemoryAllocator(Allocator &allocator) : allocator(allocator) {
	}

	void *Allocate(const size_t bytes) override {
		if (bytes == 0) {
			return ::operator new[](0);
		}
		return allocator.AllocateData(bytes);
	}

	void Free(void *const ptr, const size_t bytes) override {
		if (bytes == 0) {
			::operator delete[](ptr);
			return;
		}
		allocator.FreeData(static_cast<data_ptr_t>(ptr), bytes);
	}

private:
	Allocator &allocator;
};

} // namespace duckdb

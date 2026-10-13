#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include <istream>
#include <ostream>
#include <streambuf>

namespace duckdb {

// One serialized object (the directory, the rotation, a row group's PDX index) in a chain of allocator segments. Each
// segment starts with the IndexPointer of the next one.
struct PDXearchBlockChain {
	IndexPointer head;
	idx_t num_bytes = 0;
	// In memory only, so that freeing the chain does not read it back.
	vector<IndexPointer> segments;

	static idx_t GetSegmentSize(const BlockManager &block_manager);
	// Marks each segment's buffer modified before freeing it, so the checkpoint writes the buffer's new bitmask.
	void Free(FixedSizeAllocator &allocator);
};

// Streams bytes into a new chain. Segments are marked modified as they are allocated: SegmentHandle never sets the
// dirty flag, and the checkpoint only writes dirty buffers.
class PDXearchBlockChainWriter : public std::streambuf {
public:
	explicit PDXearchBlockChainWriter(FixedSizeAllocator &allocator);

	// Returns the chain and releases its last segment.
	PDXearchBlockChain Finish();

protected:
	int_type overflow(int_type ch) override;

private:
	void StartSegment();

	FixedSizeAllocator &allocator;
	const idx_t segment_size;
	PDXearchBlockChain chain;
	unique_ptr<SegmentHandle> current_segment;
};

// Streams chains from the blocks the allocator was persisted to, one pinned block at a time.
class PDXearchBlockChainReader : public std::streambuf {
public:
	PDXearchBlockChainReader(BlockManager &block_manager, const FixedSizeAllocatorInfo &allocator_info);

	void Open(const PDXearchBlockChain &chain);
	// Returns the chain with its segments. All of its bytes must have been read.
	PDXearchBlockChain Finish();

protected:
	int_type underflow() override;

private:
	BlockManager &block_manager;
	const idx_t segment_size;
	unordered_map<idx_t, BlockPointer> block_pointers_by_buffer_id;
	PDXearchBlockChain chain;
	IndexPointer next_segment;
	idx_t unread_bytes = 0;
	BufferHandle pinned_block;
};

static constexpr uint32_t PDXEARCH_STORAGE_VERSION = 1;

// The root chain: where the rotation and the row groups are.
struct PDXearchDirectory {
	struct RowGroupEntry {
		row_t row_start;
		row_t row_end;
		PDXearchBlockChain chain;
	};

	uint32_t num_dimensions = 0;
	PDXearchBlockChain rotation;
	vector<RowGroupEntry> row_groups;

	void Write(std::ostream &out) const;
	static PDXearchDirectory Read(std::istream &in);
};

} // namespace duckdb

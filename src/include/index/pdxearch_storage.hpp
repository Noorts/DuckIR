#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include <istream>
#include <ostream>
#include <streambuf>

#include "pdx/indexes/ivf_vanilla.hpp"

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

// What an index's paging did since it loaded, counted if pdxearch_paging_counters was on when it loaded.
struct PDXearchPagingCounters {
	atomic<bool> enabled {false};
	atomic<idx_t> cluster_acquires {0};
	atomic<idx_t> cluster_cache_misses {0};
	atomic<idx_t> cluster_bytes_fetched {0};
	atomic<idx_t> blocks_read {0};
};

// Streams chains from the blocks the allocator was persisted to, one pinned block at a time.
class PDXearchBlockChainReader : public std::streambuf {
public:
	PDXearchBlockChainReader(BlockManager &block_manager, const FixedSizeAllocatorInfo &allocator_info);

	void Open(const PDXearchBlockChain &chain);
	// Returns the chain with its segments. All of its bytes must have been read.
	PDXearchBlockChain Finish();
	// How many bytes of the open chain were read so far.
	idx_t GetBytesRead() const;
	// Stops reading the open chain before its end.
	void Close();
	// Copies size bytes at offset of a chain whose segments are known into dst. Concurrent searches may call it.
	void ReadRange(const PDXearchBlockChain &chain, idx_t offset, idx_t size, char *dst) const;
	// A checkpoint can move the allocator's buffers to other blocks.
	void UpdateBlockPointers(const FixedSizeAllocatorInfo &allocator_info);

	mutable PDXearchPagingCounters counters;

protected:
	int_type underflow() override;

private:
	// Pins the block that holds the segment into handle, and returns where the segment starts.
	data_ptr_t PinSegment(IndexPointer segment, BufferHandle &handle) const;

	BlockManager &block_manager;
	const idx_t segment_size;
	unordered_map<idx_t, BlockPointer> block_pointers_by_buffer_id;
	PDXearchBlockChain chain;
	IndexPointer next_segment;
	idx_t unread_bytes = 0;
	BufferHandle pinned_block;
};

// The clusters of a row group loaded without them (pdxearch_cluster_paging), one DuckDB buffer each. DuckDB may evict
// an unpinned buffer: the next Acquire then reads the cluster from the row group's chain again.
class PDXearchClusterCache : public PDX::IClusterSource {
public:
	PDXearchClusterCache(const PDXearchBlockChainReader &reader, BufferManager &buffer_manager);

	// Once the row group's resident data is loaded: its index, its chain, and where the cluster data starts in it.
	void Bind(const PDX::IPDXIndex &index, const PDXearchBlockChain &chain, idx_t cluster_data_start);

	const char *Acquire(uint32_t cluster_id) override;
	void Release(uint32_t cluster_id) override;

	uint64_t GetInMemorySizeInBytes() const;

private:
	struct CachedCluster {
		mutex lock;
		shared_ptr<BlockHandle> block;
		BufferHandle pinned_block;
		idx_t pin_count = 0;
	};

	const PDXearchBlockChainReader &reader;
	BufferManager &buffer_manager;
	optional_ptr<const PDX::IPDXIndex> index;
	optional_ptr<const PDXearchBlockChain> chain;
	idx_t cluster_data_start = 0;
	idx_t num_clusters = 0;
	unique_array<CachedCluster> clusters;
};

static constexpr uint32_t PDXEARCH_STORAGE_VERSION = 3;

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

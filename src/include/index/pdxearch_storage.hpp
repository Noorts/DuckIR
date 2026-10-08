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
	bool cache_tiers = false;

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

// A row group's serialized index until a checkpoint persists it, in DuckDB buffers of one block each: DuckDB writes them
// to its temporary files when it evicts them, and reads them back when they are pinned. Destroying the chain frees them.
struct PDXearchTemporaryChain {
	vector<shared_ptr<BlockHandle>> blocks;
	idx_t num_bytes = 0;

	// Copies size bytes at offset into dst. Concurrent searches may call it.
	void ReadRange(BufferManager &buffer_manager, idx_t offset, idx_t size, char *dst) const;
	// Writes all of its bytes to out, one pinned block at a time.
	void CopyTo(BufferManager &buffer_manager, std::ostream &out) const;
};

// Streams bytes into a new temporary chain. Each block is unpinned once full, so DuckDB can evict it.
class PDXearchTemporaryChainWriter : public std::streambuf {
public:
	explicit PDXearchTemporaryChainWriter(BufferManager &buffer_manager);

	// Returns the chain and unpins its last block.
	PDXearchTemporaryChain Finish();

protected:
	int_type overflow(int_type ch) override;

private:
	void StartBlock();

	BufferManager &buffer_manager;
	const idx_t block_size;
	PDXearchTemporaryChain chain;
	BufferHandle current_block;
};

// Streams a temporary chain, one pinned block at a time.
class PDXearchTemporaryChainReader : public std::streambuf {
public:
	PDXearchTemporaryChainReader(BufferManager &buffer_manager, const PDXearchTemporaryChain &chain);

	// How many bytes of the chain were read so far.
	idx_t GetBytesRead() const;

protected:
	int_type underflow() override;

private:
	BufferManager &buffer_manager;
	const idx_t block_size;
	const PDXearchTemporaryChain &chain;
	idx_t next_block = 0;
	BufferHandle pinned_block;
};

// The clusters of a row group loaded without them (pdxearch_cluster_paging), one DuckDB buffer each. DuckDB may evict
// an unpinned buffer: the next Acquire then reads the cluster from the row group's chain again.
class PDXearchClusterCache : public PDX::IClusterSource {
public:
	PDXearchClusterCache(const PDXearchBlockChainReader &reader, BufferManager &buffer_manager);

	// Once the row group's resident data is loaded: its index, the chain it was loaded from, and where the cluster data
	// starts in it.
	void Bind(const PDX::IPDXIndex &index, const PDXearchBlockChain &chain, idx_t cluster_data_start);
	void Bind(const PDX::IPDXIndex &index, const PDXearchTemporaryChain &chain, idx_t cluster_data_start);

	const char *Acquire(uint32_t cluster_id) override;
	void Release(uint32_t cluster_id) override;

	uint64_t GetInMemorySizeInBytes() const;

private:
	// pdxearch_cache_tiers: the access counts halve every ACQUIRES_PER_CLUSTER_BETWEEN_DECAYS acquires per cluster, and
	// a cluster acquired at least HOT_ACCESS_FACTOR times the average when it is fetched goes to the eviction queue
	// HOT_CLUSTER_EVICTION_QUEUE: evicted after the other clusters, before the temporary chains.
	static constexpr idx_t ACQUIRES_PER_CLUSTER_BETWEEN_DECAYS = 10;
	static constexpr idx_t HOT_ACCESS_FACTOR = 2;
	static constexpr idx_t HOT_CLUSTER_EVICTION_QUEUE = 1;

	struct CachedCluster {
		mutex lock;
		shared_ptr<BlockHandle> block;
		BufferHandle pinned_block;
		idx_t pin_count = 0;
		atomic<idx_t> access_count {0};
	};

	void RecordAccess(CachedCluster &cluster);
	void SetEvictionQueue(const CachedCluster &cluster, BufferHandle &new_block) const;
	void BindIndex(const PDX::IPDXIndex &index, idx_t cluster_data_start);

	const PDXearchBlockChainReader &reader;
	BufferManager &buffer_manager;
	optional_ptr<const PDX::IPDXIndex> index;
	// The chain the row group was loaded from: a persisted one, or a temporary one until a checkpoint persists it.
	optional_ptr<const PDXearchBlockChain> chain;
	optional_ptr<const PDXearchTemporaryChain> temporary_chain;
	idx_t cluster_data_start = 0;
	idx_t num_clusters = 0;
	unique_array<CachedCluster> clusters;
	atomic<idx_t> total_access_count {0};
	atomic<idx_t> acquires_since_decay {0};
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

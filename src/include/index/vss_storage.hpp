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
struct VSSBlockChain {
	IndexPointer head;
	idx_t num_bytes = 0;
	// In memory only, so that freeing the chain does not read it back.
	vector<IndexPointer> segments;

	static idx_t GetSegmentSize(const BlockManager &block_manager);
	// Marks each segment's buffer modified before freeing it, so the checkpoint writes the buffer's new bitmask.
	void Free(FixedSizeAllocator &allocator);
};

// Streams over a chain rethrow what its buffer throws (a block it cannot allocate or pin): a plain std::ostream or
// std::istream would only set badbit, which leaves a truncated chain behind or reads as a short one.
class VSSChainOutputStream : public std::ostream {
public:
	explicit VSSChainOutputStream(std::streambuf &buffer) : std::ostream(&buffer) {
		exceptions(std::ios::badbit);
	}
};

class VSSChainInputStream : public std::istream {
public:
	explicit VSSChainInputStream(std::streambuf &buffer) : std::istream(&buffer) {
		exceptions(std::ios::badbit);
	}
};

// Streams bytes into a new chain. Segments are marked modified as they are allocated: SegmentHandle never sets the
// dirty flag, and the checkpoint only writes dirty buffers.
class VSSBlockChainWriter : public std::streambuf {
public:
	explicit VSSBlockChainWriter(FixedSizeAllocator &allocator);

	// Returns the chain and releases its last segment.
	VSSBlockChain Finish();

protected:
	int_type overflow(int_type ch) override;

private:
	void StartSegment();

	FixedSizeAllocator &allocator;
	const idx_t segment_size;
	VSSBlockChain chain;
	unique_ptr<SegmentHandle> current_segment;
};

// What an index's paging did since it loaded (if duckir_vss_paging_counters was on when it loaded).
struct VSSPagingCounters {
	atomic<bool> enabled {false};
	atomic<idx_t> cluster_acquires {0};
	atomic<idx_t> cluster_cache_misses {0};
	atomic<idx_t> cluster_bytes_fetched {0};
	atomic<idx_t> blocks_read {0};
};

// Streams chains from the blocks the allocator was persisted to, one pinned block at a time.
class VSSBlockChainReader : public std::streambuf {
public:
	VSSBlockChainReader(BlockManager &block_manager, const FixedSizeAllocatorInfo &allocator_info);

	void Open(const VSSBlockChain &chain);
	// Returns the chain with its segments. All of its bytes must have been read.
	VSSBlockChain Finish();
	idx_t GetBytesRead() const;
	void Close();
	// Copies `size` bytes at offset of a chain whose segments are known into `dst`. Concurrent searches can call it.
	void ReadRange(const VSSBlockChain &chain, idx_t offset, idx_t size, char *dst) const;
	// A checkpoint can move the allocator's buffers to other blocks.
	void UpdateBlockPointers(const FixedSizeAllocatorInfo &allocator_info);

	mutable VSSPagingCounters counters;
	bool cache_tiers = false;

protected:
	int_type underflow() override;

private:
	// Pins the block that holds the segment into handle, and returns where the segment starts.
	data_ptr_t PinSegment(IndexPointer segment, BufferHandle &handle) const;

	BlockManager &block_manager;
	const idx_t segment_size;
	unordered_map<idx_t, BlockPointer> block_pointers_by_buffer_id;
	VSSBlockChain chain;
	IndexPointer next_segment;
	idx_t unread_bytes = 0;
	BufferHandle pinned_block;
};

// A row group's serialized index until a checkpoint persists it, in DuckDB buffers of one block each: DuckDB writes
// them to its temporary files when it evicts them, and reads them back when they are pinned. Destroying the chain frees
// them.
struct VSSTemporaryChain {
	vector<shared_ptr<BlockHandle>> blocks;
	idx_t num_bytes = 0;

	void ReadRange(BufferManager &buffer_manager, idx_t offset, idx_t size, char *dst) const;
	void CopyTo(BufferManager &buffer_manager, std::ostream &out) const;
};

// Streams bytes into a new temporary chain. Each block is unpinned once full, so DuckDB can evict it.
class VSSTemporaryChainWriter : public std::streambuf {
public:
	explicit VSSTemporaryChainWriter(BufferManager &buffer_manager);

	// Returns the chain and unpins its last block.
	VSSTemporaryChain Finish();

protected:
	int_type overflow(int_type ch) override;

private:
	void StartBlock();

	BufferManager &buffer_manager;
	const idx_t block_size;
	VSSTemporaryChain chain;
	BufferHandle current_block;
};

// Streams a temporary chain, one pinned block at a time.
class VSSTemporaryChainReader : public std::streambuf {
public:
	VSSTemporaryChainReader(BufferManager &buffer_manager, const VSSTemporaryChain &chain);

	idx_t GetBytesRead() const;

protected:
	int_type underflow() override;

private:
	BufferManager &buffer_manager;
	const idx_t block_size;
	const VSSTemporaryChain &chain;
	idx_t next_block = 0;
	BufferHandle pinned_block;
};

// The clusters of a row group that was loaded with paging (duckir_vss_cluster_paging).
// Each cluster a search probes gets its own DuckDB buffer, allocated on miss and sized to its bytes. DuckDB may evict
// an unpinned buffer. The next Acquire then reads the cluster from the row group's chain again.
class VSSClusterCache : public PDX::IClusterSource {
public:
	VSSClusterCache(const VSSBlockChainReader &reader, BufferManager &buffer_manager);

	// Called right after the row group's resident data is loaded (PageRowGroup).
	void Bind(const PDX::IPDXIndex &index, const VSSBlockChain &chain, idx_t cluster_data_start);
	void Bind(const PDX::IPDXIndex &index, const VSSTemporaryChain &chain, idx_t cluster_data_start);

	const char *Acquire(uint32_t cluster_id) override;
	void Release(uint32_t cluster_id) override;

	uint64_t GetInMemorySizeInBytes() const;

private:
	// duckir_vss_cache_tiers: the access counts halve every ACQUIRES_PER_CLUSTER_BETWEEN_DECAYS acquires per cluster,
	// and a cluster acquired at least HOT_ACCESS_FACTOR times the average when it is fetched goes to the eviction queue
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

	const VSSBlockChainReader &reader;
	BufferManager &buffer_manager;
	optional_ptr<const PDX::IPDXIndex> index;
	// The chain the row group was loaded from: a persisted one, or a temporary one until a checkpoint persists it.
	optional_ptr<const VSSBlockChain> chain;
	optional_ptr<const VSSTemporaryChain> temporary_chain;
	idx_t cluster_data_start = 0;
	idx_t num_clusters = 0;
	unique_array<CachedCluster> clusters;
	atomic<idx_t> total_access_count {0};
	atomic<idx_t> acquires_since_decay {0};
};

static constexpr uint32_t VSS_STORAGE_VERSION = 4;

// The root chain: where the rotation and the row groups are.
struct VSSDirectory {
	struct RowGroupEntry {
		row_t row_start;
		row_t row_end;
		VSSBlockChain chain;
	};

	uint32_t num_dimensions = 0;
	VSSBlockChain rotation;
	vector<RowGroupEntry> row_groups;

	void Write(std::ostream &out) const;
	static VSSDirectory Read(std::istream &in);
};

} // namespace duckdb

#include "index/pdxearch_storage.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "pdx/utils.hpp"

namespace duckdb {

// A FixedSizeAllocator buffer (ART's storage format too) starts with a bitmask of one validity_t per 64 segments.
static constexpr idx_t ALLOCATOR_BITMASK_SIZE = sizeof(validity_t);
// Each segment of a chain starts with the IndexPointer of the next one.
static constexpr idx_t SEGMENT_HEADER_SIZE = sizeof(idx_t);

// Half of what the block holds after the bitmask, so a buffer holds exactly two segments.
idx_t PDXearchBlockChain::GetSegmentSize(const BlockManager &block_manager) {
	const idx_t usable_bytes = block_manager.GetBlockSize() - ALLOCATOR_BITMASK_SIZE;
	// Otherwise the allocator adds a second bitmask entry, and the reader would look for segments at the wrong offset.
	if (usable_bytes % 2 != 0) {
		throw InternalException("PDXearch does not support a block size of %llu bytes", block_manager.GetBlockSize());
	}
	return usable_bytes / 2;
}

// Allocates nothing: the put area starts empty, so the first write calls overflow.
PDXearchBlockChainWriter::PDXearchBlockChainWriter(FixedSizeAllocator &allocator)
    : allocator(allocator), segment_size(allocator.GetSegmentSize()) {
}

// Allocates the next segment, links it from the current one, and makes its payload the put area.
void PDXearchBlockChainWriter::StartSegment() {
	const auto segment = allocator.New();
	auto handle = make_uniq<SegmentHandle>(allocator.GetHandle(segment));
	handle->MarkModified();
	if (current_segment) {
		// The current segment is full.
		chain.num_bytes += static_cast<idx_t>(pptr() - pbase());
		Store<idx_t>(segment.Get(), current_segment->GetPtr());
	} else {
		chain.head = segment;
	}
	chain.segments.push_back(segment);
	current_segment = std::move(handle);
	const auto data = current_segment->GetPtr<char>();
	setp(data + SEGMENT_HEADER_SIZE, data + segment_size);
}

// Called by the ostream when the put area is full, with the character that did not fit.
PDXearchBlockChainWriter::int_type PDXearchBlockChainWriter::overflow(const int_type ch) {
	if (traits_type::eq_int_type(ch, traits_type::eof())) {
		return traits_type::not_eof(ch);
	}
	StartSegment();
	*pptr() = traits_type::to_char_type(ch);
	pbump(1);
	return ch;
}

// Counts the bytes of the last segment and releases it: DuckDB asserts that no handle holds a buffer it serializes.
PDXearchBlockChain PDXearchBlockChainWriter::Finish() {
	chain.num_bytes += static_cast<idx_t>(pptr() - pbase());
	setp(nullptr, nullptr);
	current_segment.reset();
	auto result = std::move(chain);
	chain = PDXearchBlockChain();
	return result;
}

PDXearchBlockChainReader::PDXearchBlockChainReader(BlockManager &block_manager,
                                                   const FixedSizeAllocatorInfo &allocator_info)
    : block_manager(block_manager), segment_size(allocator_info.segment_size) {
	UpdateBlockPointers(allocator_info);
}

// Indexes the persisted buffers by id, for every chain this reader opens.
void PDXearchBlockChainReader::UpdateBlockPointers(const FixedSizeAllocatorInfo &allocator_info) {
	block_pointers_by_buffer_id.clear();
	for (idx_t i = 0; i < allocator_info.buffer_ids.size(); i++) {
		block_pointers_by_buffer_id.emplace(allocator_info.buffer_ids[i], allocator_info.block_pointers[i]);
	}
}

// Starts reading a chain from its head. The get area starts empty, so the first read calls underflow.
void PDXearchBlockChainReader::Open(const PDXearchBlockChain &chain_p) {
	chain = PDXearchBlockChain {chain_p.head, chain_p.num_bytes, {}};
	next_segment = chain_p.head;
	unread_bytes = chain_p.num_bytes;
	setg(nullptr, nullptr, nullptr);
}

// Unpins the last block and returns the chain with every segment it went through.
PDXearchBlockChain PDXearchBlockChainReader::Finish() {
	// Unread segments would be missing from the chain, and leak when it is freed.
	if (unread_bytes > 0 || gptr() != egptr()) {
		throw InternalException("PDXearch read only part of a chain of its index storage");
	}
	setg(nullptr, nullptr, nullptr);
	pinned_block.Destroy();
	auto result = std::move(chain);
	chain = PDXearchBlockChain();
	return result;
}

// What the chain holds, minus what is still unread in it and in the get area.
idx_t PDXearchBlockChainReader::GetBytesRead() const {
	return chain.num_bytes - unread_bytes - static_cast<idx_t>(egptr() - gptr());
}

// Unlike Finish, collects no segments: the directory has them.
void PDXearchBlockChainReader::Close() {
	setg(nullptr, nullptr, nullptr);
	pinned_block.Destroy();
	chain = PDXearchBlockChain();
	unread_bytes = 0;
}

// Every segment but the last is full, so the segment that holds a byte follows from its offset. Each block is pinned
// only while its bytes are copied.
void PDXearchBlockChainReader::ReadRange(const PDXearchBlockChain &chain_p, idx_t offset, idx_t size, char *dst) const {
	if (offset + size > chain_p.num_bytes) {
		throw InternalException("PDXearch read past the end of a chain of its index storage");
	}
	const idx_t payload_size = segment_size - SEGMENT_HEADER_SIZE;
	while (size > 0) {
		const idx_t offset_in_payload = offset % payload_size;
		const idx_t num_bytes = MinValue<idx_t>(size, payload_size - offset_in_payload);
		BufferHandle handle;
		const auto segment = PinSegment(chain_p.segments[offset / payload_size], handle);
		memcpy(dst, segment + SEGMENT_HEADER_SIZE + offset_in_payload, num_bytes);
		offset += num_bytes;
		size -= num_bytes;
		dst += num_bytes;
	}
}

PDXearchClusterCache::PDXearchClusterCache(const PDXearchBlockChainReader &reader, BufferManager &buffer_manager)
    : reader(reader), buffer_manager(buffer_manager) {
}

void PDXearchClusterCache::Bind(const PDX::IPDXIndex &index_p, const PDXearchBlockChain &chain_p,
                                const idx_t cluster_data_start_p) {
	index = &index_p;
	chain = &chain_p;
	cluster_data_start = cluster_data_start_p;
	num_clusters = index_p.GetNumClusters();
	clusters = make_uniq_array<CachedCluster>(num_clusters);
}

// Only the cluster's own lock is held while it is read, so searches of other clusters go on.
const char *PDXearchClusterCache::Acquire(const uint32_t cluster_id) {
	const bool record_counters = reader.counters.enabled.load(std::memory_order_relaxed);
	if (record_counters) {
		reader.counters.cluster_acquires++;
	}
	auto &cluster = clusters[cluster_id];
	lock_guard<mutex> guard(cluster.lock);
	if (cluster.pin_count == 0) {
		if (cluster.block) {
			cluster.pinned_block = buffer_manager.Pin(cluster.block);
		}
		// Never read, or destroyed when DuckDB evicted it.
		if (!cluster.pinned_block.IsValid()) {
			const auto range = index->GetClusterDataRange(cluster_id);
			if (record_counters) {
				reader.counters.cluster_cache_misses++;
				reader.counters.cluster_bytes_fetched += range.second;
			}
			auto new_block = buffer_manager.Allocate(MemoryTag::EXTENSION, range.second, /*can_destroy=*/true);
			reader.ReadRange(*chain, cluster_data_start + range.first, range.second, char_ptr_cast(new_block.Ptr()));
			cluster.block = new_block.GetBlockHandle();
			cluster.pinned_block = std::move(new_block);
		}
	}
	cluster.pin_count++;
	return const_char_ptr_cast(cluster.pinned_block.Ptr());
}

void PDXearchClusterCache::Release(const uint32_t cluster_id) {
	auto &cluster = clusters[cluster_id];
	lock_guard<mutex> guard(cluster.lock);
	cluster.pin_count--;
	if (cluster.pin_count == 0) {
		cluster.pinned_block.Destroy();
	}
}

// The buffers are DuckDB's: they count toward memory_limit by themselves.
uint64_t PDXearchClusterCache::GetInMemorySizeInBytes() const {
	return sizeof(*this) + num_clusters * sizeof(CachedCluster);
}

// A buffer is persisted at block_pointer.offset of its block, possibly sharing the block with other buffers.
data_ptr_t PDXearchBlockChainReader::PinSegment(const IndexPointer segment, BufferHandle &handle) const {
	const auto entry = block_pointers_by_buffer_id.find(segment.GetBufferId());
	if (entry == block_pointers_by_buffer_id.end()) {
		throw SerializationException("PDXearch index storage points to a buffer that does not exist");
	}
	const auto &block_pointer = entry->second;
	auto block_handle = block_manager.RegisterBlock(block_pointer.block_id);
	if (counters.enabled.load(std::memory_order_relaxed) &&
	    block_handle->GetMemory().GetState() == BlockState::BLOCK_UNLOADED) {
		counters.blocks_read++;
	}
	handle = block_manager.buffer_manager.Pin(block_handle);
	return handle.Ptr() + block_pointer.offset + ALLOCATOR_BITMASK_SIZE + segment.GetOffset() * segment_size;
}

// Called by the istream when the get area is exhausted: pins the next segment's block and makes its payload the get
// area. Pinning the next block unpins the previous one.
PDXearchBlockChainReader::int_type PDXearchBlockChainReader::underflow() {
	if (gptr() < egptr()) {
		return traits_type::to_int_type(*gptr());
	}
	if (unread_bytes == 0) {
		return traits_type::eof();
	}
	chain.segments.push_back(next_segment);
	const auto segment = PinSegment(next_segment, pinned_block);
	next_segment.Set(Load<idx_t>(segment));
	// Every segment but the last is full.
	const auto payload_size = MinValue<idx_t>(unread_bytes, segment_size - SEGMENT_HEADER_SIZE);
	unread_bytes -= payload_size;
	const auto payload = reinterpret_cast<char *>(segment + SEGMENT_HEADER_SIZE);
	setg(payload, payload, payload + payload_size);
	return traits_type::to_int_type(*gptr());
}

// Marking a buffer modified makes the checkpoint write its new bitmask. A buffer left empty is dropped, and its block
// freed at the checkpoint.
void PDXearchBlockChain::Free(FixedSizeAllocator &allocator) {
	for (const auto &segment : segments) {
		allocator.GetHandle(segment).MarkModified();
		allocator.Free(segment);
	}
	*this = PDXearchBlockChain();
}

// Where the chain starts, how long it is, and its segments, so that any range of it can be read alone (ReadRange).
static void WriteChain(std::ostream &out, const PDXearchBlockChain &chain) {
	PDX::WriteValue(out, chain.head.Get());
	PDX::WriteValue(out, chain.num_bytes);
	PDX::WriteValue(out, static_cast<uint64_t>(chain.segments.size()));
	for (const auto &segment : chain.segments) {
		PDX::WriteValue(out, segment.Get());
	}
}

static PDXearchBlockChain ReadChain(PDX::StreamReader &reader) {
	PDXearchBlockChain chain;
	chain.head.Set(PDX::ReadValue<idx_t>(reader));
	chain.num_bytes = PDX::ReadValue<idx_t>(reader);
	chain.segments.resize(PDX::ReadValue<uint64_t>(reader));
	for (auto &segment : chain.segments) {
		segment.Set(PDX::ReadValue<idx_t>(reader));
	}
	return chain;
}

void PDXearchDirectory::Write(std::ostream &out) const {
	PDX::WriteValue(out, num_dimensions);
	WriteChain(out, rotation);
	PDX::WriteValue(out, static_cast<uint64_t>(row_groups.size()));
	for (const auto &row_group : row_groups) {
		PDX::WriteValue(out, row_group.row_start);
		PDX::WriteValue(out, row_group.row_end);
		WriteChain(out, row_group.chain);
	}
}

// Throws std::runtime_error (PDX's StreamReader) if the stream ends early.
PDXearchDirectory PDXearchDirectory::Read(std::istream &in) {
	PDX::StreamReader reader {in};
	PDXearchDirectory directory;
	directory.num_dimensions = PDX::ReadValue<uint32_t>(reader);
	directory.rotation = ReadChain(reader);
	directory.row_groups.resize(PDX::ReadValue<uint64_t>(reader));
	for (auto &row_group : directory.row_groups) {
		row_group.row_start = PDX::ReadValue<row_t>(reader);
		row_group.row_end = PDX::ReadValue<row_t>(reader);
		row_group.chain = ReadChain(reader);
	}
	return directory;
}

} // namespace duckdb

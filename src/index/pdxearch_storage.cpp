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

// Indexes the persisted buffers by id once, for every chain this reader opens.
PDXearchBlockChainReader::PDXearchBlockChainReader(BlockManager &block_manager,
                                                   const FixedSizeAllocatorInfo &allocator_info)
    : block_manager(block_manager), segment_size(allocator_info.segment_size) {
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

// Called by the istream when the get area is exhausted: pins the next segment's block and makes its payload the get
// area. Pinning the next block unpins the previous one.
PDXearchBlockChainReader::int_type PDXearchBlockChainReader::underflow() {
	if (gptr() < egptr()) {
		return traits_type::to_int_type(*gptr());
	}
	if (unread_bytes == 0) {
		return traits_type::eof();
	}
	const auto entry = block_pointers_by_buffer_id.find(next_segment.GetBufferId());
	if (entry == block_pointers_by_buffer_id.end()) {
		throw SerializationException("PDXearch index storage points to a buffer that does not exist");
	}
	const auto &block_pointer = entry->second;
	auto block_handle = block_manager.RegisterBlock(block_pointer.block_id);
	pinned_block = block_manager.buffer_manager.Pin(block_handle);
	chain.segments.push_back(next_segment);

	// A buffer is persisted at block_pointer.offset of its block, possibly sharing the block with other buffers.
	const auto segment =
	    pinned_block.Ptr() + block_pointer.offset + ALLOCATOR_BITMASK_SIZE + next_segment.GetOffset() * segment_size;
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

// Only where the chain starts and how long it is: the reader collects its segments.
static void WriteChain(std::ostream &out, const PDXearchBlockChain &chain) {
	PDX::WriteValue(out, chain.head.Get());
	PDX::WriteValue(out, chain.num_bytes);
}

static PDXearchBlockChain ReadChain(PDX::StreamReader &reader) {
	PDXearchBlockChain chain;
	chain.head.Set(PDX::ReadValue<idx_t>(reader));
	chain.num_bytes = PDX::ReadValue<idx_t>(reader);
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

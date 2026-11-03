#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/storage/storage_lock.hpp"

namespace duckdb {

class DuckTableEntry;
class VSSIndex;

// What a plan needs of its index without holding it, as of planning: the oid that finds the index again when the query
// runs (AcquireSearchGuard), the dimension of its query vectors, and what EXPLAIN shows.
struct VSSIndexPlanInfo {
	idx_t oid;
	string name;
	idx_t num_dimensions;
	idx_t total_num_clusters;
	idx_t num_row_groups;
	idx_t in_memory_size;
};

// A query's hold on an index while it searches it: a pin, so DROP INDEX (VSSIndex::ResetStorage) waits for the query
// before the index is destroyed, and the shared search lock. The query's operators on the index share one guard, so the
// query takes the lock once.
class VSSSearchGuard {
public:
	explicit VSSSearchGuard(VSSIndex &index);
	~VSSSearchGuard();

	VSSIndex &index;
	unique_ptr<StorageLockKey> search_lock;
};

// The running query's search guards, by index oid.
class VSSSearchGuardRegistry : public ClientContextState {
public:
	void QueryEnd(ClientContext &context) override;

	mutex lock;
	unordered_map<idx_t, weak_ptr<VSSSearchGuard>> guards;
};

// The running query's guard on the index with this oid, synced with the table: the first operator that asks creates it.
// Throws a TransactionException when a concurrent transaction dropped the index.
shared_ptr<VSSSearchGuard> AcquireSearchGuard(ClientContext &context, DuckTableEntry &table, idx_t index_oid);

// Whether the running query holds a guard, and so the shared search lock, on this index.
bool HasSearchGuard(const ClientContext &context, const VSSIndex &index);

} // namespace duckdb

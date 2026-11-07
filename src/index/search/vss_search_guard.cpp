#include "index/search/vss_search_guard.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/index_entry.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "index/vss_index.hpp"

namespace duckdb {

static constexpr const char *SEARCH_GUARD_REGISTRY_KEY = "duckir_vss_search_guards";

VSSSearchGuard::VSSSearchGuard(VSSIndex &index) : index(index) {
	const lock_guard<mutex> pins(index.search_pins_lock);
	if (index.is_retired) {
		throw TransactionException("DuckIR VSS index %s was dropped by a concurrent transaction", index.GetIndexName());
	}
	index.search_pins++;
}

// The lock goes first, the pin last: ResetStorage destroys the index once its pins are gone.
VSSSearchGuard::~VSSSearchGuard() {
	search_lock.reset();
	const lock_guard<mutex> pins(index.search_pins_lock);
	index.search_pins--;
	index.search_pins_released.notify_all();
}

void VSSSearchGuardRegistry::QueryEnd(ClientContext &context) {
	const lock_guard<mutex> registry_lock(lock);
	guards.clear();
}

shared_ptr<VSSSearchGuard> AcquireSearchGuard(ClientContext &context, DuckTableEntry &table, const idx_t index_oid) {
	auto registry = context.registered_state->GetOrCreate<VSSSearchGuardRegistry>(SEARCH_GUARD_REGISTRY_KEY);
	const lock_guard<mutex> registry_lock(registry->lock);
	auto &registered_guard = registry->guards[index_oid];
	auto guard = registered_guard.lock();
	if (guard) {
		return guard;
	}
	// DROP INDEX removes the index's entry under the list lock before it retires the index, so an entry found here is
	// alive: it is pinned under its read handle.
	auto &storage = table.GetStorage();
	for (auto entry : storage.GetDataTableInfo()->GetIndexes().IndexEntries()) {
		if (entry->GetIndexOid() != index_oid) {
			continue;
		}
		auto index = entry->GetReadHandle<VSSIndex>();
		// The handle only exposes a const index; the search changes it under the search lock.
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		guard = make_shared_ptr<VSSSearchGuard>(const_cast<VSSIndex &>(*index.operator->()));
		break;
	}
	if (!guard) {
		throw TransactionException("A DuckIR VSS index of table %s was dropped by a concurrent transaction",
		                           table.name.GetIdentifierName());
	}
	// Outside the index list's lock, which commits take while they wait for the search lock.
	guard->search_lock = guard->index.SyncAndLockForSearch(context, storage);
	registered_guard = guard;
	return guard;
}

bool HasSearchGuard(const ClientContext &context, const VSSIndex &index) {
	auto registry = context.registered_state->Get<VSSSearchGuardRegistry>(SEARCH_GUARD_REGISTRY_KEY);
	if (!registry) {
		return false;
	}
	const lock_guard<mutex> registry_lock(registry->lock);
	for (auto &registered_guard : registry->guards) {
		auto guard = registered_guard.second.lock();
		if (guard && &guard->index == &index && guard->search_lock) {
			return true;
		}
	}
	return false;
}

} // namespace duckdb

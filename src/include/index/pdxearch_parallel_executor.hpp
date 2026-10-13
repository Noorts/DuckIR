#pragma once

#include <functional>

#include "duckdb/common/error_data.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "pdx/common.hpp"

namespace duckdb {

// PDX's parallel loops on DuckDB's scheduler; the calling thread works on the tasks too.
class DuckDBParallelExecutor final : public PDX::ParallelExecutor {
public:
	DuckDBParallelExecutor(ClientContext &context, const idx_t num_workers)
	    : context(context), num_workers(MaxValue<idx_t>(num_workers, 1)) {
	}

	size_t NumWorkers() const override {
		return num_workers;
	}

	void ParallelFor(size_t n, const std::function<void(size_t, size_t, size_t)> &fn) override {
		if (n == 0) {
			return;
		}
		const idx_t num_tasks = MinValue<idx_t>(num_workers, n);
		if (num_tasks == 1) {
			fn(0, n, 0);
			return;
		}
		TaskExecutor executor(context);
		try {
			for (idx_t worker = 0; worker < num_tasks; worker++) {
				executor.ScheduleTask(
				    make_uniq<RangeTask>(executor, fn, n * worker / num_tasks, n * (worker + 1) / num_tasks, worker));
			}
		} catch (std::exception &ex) {
			executor.PushError(ErrorData(ex));
			executor.WorkOnTasks();
			throw;
		}
		executor.WorkOnTasks();
	}

private:
	// Holds references only: DuckDB may destroy a task after WorkOnTasks returns.
	class RangeTask final : public BaseExecutorTask {
	public:
		RangeTask(TaskExecutor &executor, const std::function<void(size_t, size_t, size_t)> &fn, const idx_t begin,
		          const idx_t end, const idx_t worker)
		    : BaseExecutorTask(executor), fn(fn), begin(begin), end(end), worker(worker) {
		}

		void ExecuteTask() override {
			fn(begin, end, worker);
		}

		string TaskType() const override {
			return "PDXearchRangeTask";
		}

	private:
		const std::function<void(size_t, size_t, size_t)> &fn;
		const idx_t begin;
		const idx_t end;
		const idx_t worker;
	};

	ClientContext &context;
	const idx_t num_workers;
};

} // namespace duckdb

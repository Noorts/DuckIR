#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb==2.0.0", "numpy", "pyarrow", "h5py", "gdown"]
# ///
"""DuckIR vs duckdb-vss on 1M OpenAI embeddings (d = 1536).

This script reproduces the comparison table in our README.

Run it with uv:

  uv run benchmarks/duckir_vs_vss.py <subcommand> --system <vss|duckir>

Subcommands:

  build      Indexing time and Index size on disk (12 threads)
  search     Search at recall@10 = 0.95 (1 thread)
  filtered   Filtered search at 50%, 10%, 1% and 0.1% of rows passing (1 thread)
  restart    First query after a restart, with the database dropped from the page cache (1 thread)
  ingest     Ingestion and I/O on CHECKPOINT (1 thread). It changes the database, so it runs last.

Data: the openai vectors as raw little-endian float32 (rows x 1536, no header) and their exact nearest neighbours.
If you lack the dataset, this script will download openai-1536-angular.hdf5 (12 GB, from Google Drive), converts its
train and test arrays into the two files below, and computes the ground truth with numpy.
  VB_DATA_DIR       holds data_openai.bin (999,000 rows) and data_openai_test.bin (1,000 queries)
  VB_GROUND_TRUTH   openai.json: {"<query index>": [the nearest ids, closest first]}
  VB_EXTENSION      the built duckir.duckdb_extension
  VB_DB_DIR         where the two databases are written

Linux only: `restart` drops the page cache with posix_fadvise, and `ingest` reads the bytes written from /proc.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('VB_DATA_DIR', f'{REPO}/benchmarks/data')
GROUND_TRUTH = os.environ.get('VB_GROUND_TRUTH', f'{DATA_DIR}/openai.json')
EXTENSION = os.environ.get('VB_EXTENSION', f'{REPO}/build/release/extension/duckir/duckir.duckdb_extension')
DB_DIR = os.environ.get('VB_DB_DIR', f'{REPO}/benchmarks/db')
HDF5_FILE = 'openai-1536-angular.hdf5'
HDF5_DRIVE_ID = '1cwIY76n_HEbbZAANJVWiVkjBsIy3b9jB'

N = 999_000
DIM = 1536
K = 10
SEED = 42
BUILD_THREADS = 12
QUERY_THREADS = 1


# ---------------------------------------------------------------- connections

def db_path(system):
    return f'{DB_DIR}/{system}.duckdb'


def connect(system, threads, read_only=True):
    config = {'threads': threads, 'memory_limit': '16GB', 'allow_unsigned_extensions': 'true'}
    con = duckdb.connect(db_path(system), read_only=read_only, config=config)
    if system == 'vss':
        con.execute('INSTALL vss')
        con.execute('LOAD vss')
        con.execute('SET hnsw_enable_experimental_persistence = true')
    else:
        con.execute(f"LOAD '{EXTENSION}'")
    return con


def set_effort(con, system, value):
    """The search effort: HNSW's ef_search for duckdb-vss, clusters probed per row group for DuckIR."""
    con.execute(f"SET {'hnsw_ef_search' if system == 'vss' else 'duckir_vss_n_probe'} = {value}")


def file_size(path):
    return sum(os.path.getsize(p) for p in (path, path + '.wal') if os.path.exists(p))


def report(row, system, **values):
    """One JSON line per measured value, named after the README row it fills."""
    print(json.dumps({'row': row, 'system': system, **values}), flush=True)


# ---------------------------------------------------------------- data

def base_vectors():
    return np.memmap(f'{DATA_DIR}/data_openai.bin', dtype='<f4', mode='r', shape=(N, DIM))


def queries():
    return np.fromfile(f'{DATA_DIR}/data_openai_test.bin', dtype='<f4').reshape(-1, DIM)


def ground_truth():
    neighbours = json.load(open(GROUND_TRUTH))
    return np.array([neighbours[str(i)] for i in range(len(neighbours))], dtype=np.int64)


def categories():
    """The filter column: uniform in [0, 1000), so `cat < 10` passes 1% of the rows."""
    return np.random.default_rng(SEED).integers(0, 1000, N, dtype=np.int32)


def exact_top_k(qs, passing):
    """The exact top-10 of each query among the passing rows, by brute force in numpy."""
    base, qs = base_vectors(), qs.astype(np.float64)
    passing = np.flatnonzero(passing)
    best_ids, best_dist = np.zeros((len(qs), 0), dtype=np.int64), np.zeros((len(qs), 0))
    for start in range(0, len(passing), 100_000):
        part = passing[start:start + 100_000]
        x = np.asarray(base[part], dtype=np.float64)
        dist = (x * x).sum(1)[None, :] - 2 * qs @ x.T  # squared L2 without |q|^2, which doesn't change the order
        top = np.argpartition(dist, min(K, dist.shape[1]) - 1, axis=1)[:, :K]
        ids = np.concatenate([best_ids, part[top]], 1)
        dist = np.concatenate([best_dist, np.take_along_axis(dist, top, 1)], 1)
        order = np.argsort(dist, 1)[:, :K]
        best_ids, best_dist = np.take_along_axis(ids, order, 1), np.take_along_axis(dist, order, 1)
    return best_ids


def prepare_data():
    """Prepares what is missing: the vectors, converted from the hdf5 (downloaded first if needed), as float32 and
    not normalized, like the files the README's numbers were measured on; then their ground truth."""
    base_path, test_path = f'{DATA_DIR}/data_openai.bin', f'{DATA_DIR}/data_openai_test.bin'
    if not (os.path.exists(base_path) and os.path.exists(test_path)):
        os.makedirs(DATA_DIR, exist_ok=True)
        hdf5_path = f'{DATA_DIR}/{HDF5_FILE}'
        if not os.path.exists(hdf5_path):
            import gdown
            gdown.download(id=HDF5_DRIVE_ID, output=hdf5_path)
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            train = f['train']
            assert train.shape == (N, DIM), f'unexpected train shape {train.shape}'
            with open(base_path, 'wb') as out:
                for start in range(0, N, 100_000):
                    np.asarray(train[start:start + 100_000], dtype=np.float32).tofile(out)
            np.asarray(f['test'], dtype=np.float32).tofile(test_path)
    if not os.path.exists(GROUND_TRUTH):
        truth = exact_top_k(queries(), np.ones(N, dtype=bool))
        with open(GROUND_TRUTH, 'w') as f:
            json.dump({str(i): ids.tolist() for i, ids in enumerate(truth)}, f)


def insert_rows(con, ids, rows_per_insert=None):
    """Inserts the rows with these ids, `rows_per_insert` per INSERT statement (default: all in one)."""
    base, cat = base_vectors(), categories()
    step = rows_per_insert or len(ids)
    for start in range(0, len(ids), step):
        part = ids[start:start + step]
        emb = pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(base[part]).reshape(-1)), DIM)
        con.register('new_rows', pa.table({'id': pa.array(part.astype(np.int32)), 'cat': pa.array(cat[part]), 'emb': emb}))
        con.execute('INSERT INTO t SELECT * FROM new_rows')
        con.unregister('new_rows')


# ---------------------------------------------------------------- queries

def query_runner(con, system, qs, where=None):
    """A function that runs the top-10 search for query i and returns its rows. Each system gets its fastest way to
    pass the vector: a FLOAT[1536] parameter for duckdb-vss, and the vector's float32 bytes as a
    BLOB parameter for DuckIR."""
    vector = '?::FLOAT[1536]' if system == 'vss' else '?'
    sql = f"SELECT id FROM t{f' WHERE {where}' if where else ''} ORDER BY array_distance(emb, {vector}) LIMIT {K}"
    params = [[q.tolist()] if system == 'vss' else [np.asarray(q, dtype='<f4').tobytes()] for q in qs]
    return lambda i: con.execute(sql, params[i]).fetchall()


def recall(rows, truth):
    return len({row[0] for row in rows} & set(truth[:K].tolist())) / K


# ---------------------------------------------------------------- Indexing time, Index size on disk

def cmd_build(args):
    """Loads the 999,000 rows (untimed), reads the table warm, then times CREATE INDEX. The index's size on disk is the
    database file's growth once the index is checkpointed."""
    path = db_path(args.system)
    os.makedirs(DB_DIR, exist_ok=True)
    assert not os.path.exists(path), f'{path} already exists'
    con = connect(args.system, BUILD_THREADS, read_only=False)
    con.execute('CREATE TABLE t (id INTEGER, cat INTEGER, emb FLOAT[1536])')
    con.execute('BEGIN')
    insert_rows(con, np.arange(N), rows_per_insert=50_000)
    con.execute('COMMIT')
    con.execute('CHECKPOINT')
    table_bytes = file_size(path)
    con.close()

    con = connect(args.system, BUILD_THREADS, read_only=False)
    con.execute('SELECT count(*), sum(emb[1]) FROM t').fetchall()  # reads the table warm
    if args.system == 'vss':
        ddl = "CREATE INDEX idx ON t USING HNSW (emb) WITH (metric = 'l2sq')"
    else:
        ddl = f"CREATE INDEX idx ON t USING DUCKIR_VSS (emb) WITH (metric = 'l2sq', quantization = 'u8', seed = {SEED})"
    start = time.perf_counter()
    con.execute(ddl)
    seconds = time.perf_counter() - start
    con.execute('CHECKPOINT')
    con.close()
    report('Indexing time', args.system, seconds=seconds, threads=BUILD_THREADS)
    report('Index size on disk', args.system, gb=(file_size(path) - table_bytes) / 1e9)


# ---------------------------------------------------------------- Search (recall@10 = 0.95)

SEARCH_EFFORTS = {'vss': [16, 32, 64, 128, 256, 512], 'duckir': [4, 8, 16, 24, 48, 96, 192]}


def cmd_search(args):
    """Sweeps the search effort. Each effort gets a warm-up pass over the 1,000 queries, then 10 passes in shuffled
    order (10,000 timed queries). QPS = 1 / mean latency, interpolated at recall@10 = 0.95."""
    con = connect(args.system, QUERY_THREADS)
    qs, truth = queries(), ground_truth()
    run = query_runner(con, args.system, qs)
    recalls, qps = [], []
    for effort in SEARCH_EFFORTS[args.system]:
        set_effort(con, args.system, effort)
        for i in range(len(qs)):
            run(i)
        rng = np.random.default_rng(SEED)
        seconds, hits = [], []
        for _ in range(10):
            for i in rng.permutation(len(qs)):
                start = time.perf_counter()
                rows = run(i)
                seconds.append(time.perf_counter() - start)
                hits.append(recall(rows, truth[i]))
        recalls.append(float(np.mean(hits)))
        qps.append(1 / float(np.mean(seconds)))
        report('Search, one effort', args.system, effort=effort, recall_at_10=recalls[-1], qps=qps[-1])
    report('Search (recall@10 = 0.95)', args.system, qps=float(np.interp(0.95, recalls, qps)))


# ---------------------------------------------------------------- Filtered search

FILTERS = [('50%', 500), ('10%', 100), ('1%', 10), ('0.1%', 1)]  # rows passing `cat < threshold`
FILTERED_EFFORT = {'vss': 64, 'duckir': 96}  # ef_search 64 is duckdb-vss's default
FILTERED_QUERIES = 100


def cmd_filtered(args):
    """For each filter, 100 queries after 5 warm-up ones; one warm-up pass over the 1,000 queries without a filter
    first. QPS = 1 / p50 latency. Recall is reported so the two systems can be compared at a similar recall."""
    con = connect(args.system, QUERY_THREADS)
    qs, cat = queries(), categories()
    set_effort(con, args.system, FILTERED_EFFORT[args.system])
    warm_up = query_runner(con, args.system, qs)
    for i in range(len(qs)):
        warm_up(i)
    qs = qs[:FILTERED_QUERIES]
    for share, threshold in FILTERS:
        truth = exact_top_k(qs, cat < threshold)
        run = query_runner(con, args.system, qs, where=f'cat < {threshold}')
        for i in range(5):
            run(i)
        seconds, hits = [], []
        for i in range(len(qs)):
            start = time.perf_counter()
            rows = run(i)
            seconds.append(time.perf_counter() - start)
            hits.append(recall(rows, truth[i]))
        report('Filtered search', args.system, rows_passing=share, effort=FILTERED_EFFORT[args.system],
               qps=1 / float(np.percentile(seconds, 50)), recall_at_10=float(np.mean(hits)))


# ---------------------------------------------------------------- First query after a restart

def cmd_restart(args):
    """Three reopenings, each in a fresh process (`first_query`). Reports the median."""
    runs = [float(subprocess.run([sys.executable, __file__, 'first_query', '--system', args.system],
                                 check=True, capture_output=True, text=True).stdout) for _ in range(3)]
    report('First query after a restart', args.system, seconds=float(np.median(runs)), runs=runs)


def cmd_first_query(args):
    """Drops the database file from the page cache, opens it, and prints how long the first query takes."""
    path = db_path(args.system)
    for p in (path, path + '.wal'):
        if os.path.exists(p):
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    con = connect(args.system, QUERY_THREADS)
    run = query_runner(con, args.system, queries()[:1])
    start = time.perf_counter()
    run(0)
    print(time.perf_counter() - start)


# ---------------------------------------------------------------- Ingestion, I/O on CHECKPOINT

def bytes_written():
    """The bytes this process has made the kernel write to storage."""
    with open('/proc/self/io') as f:
        return next(int(line.split()[1]) for line in f if line.startswith('write_bytes'))


def insert_and_index(con, system, ids, rows_per_insert=None):
    """Inserts the rows and returns the seconds until they're indexed. DuckIR indexes inserted rows at the next search,
    so duckir_vss_sync_index is included here instead."""
    start = time.perf_counter()
    insert_rows(con, ids, rows_per_insert)
    if system == 'duckir':
        con.execute("CALL duckir_vss_sync_index('idx')")
    return time.perf_counter() - start


def cmd_ingest(args):
    """1% of the rows are deleted, then inserted back in one INSERT; then the CHECKPOINT that persists them; then
    another 1% inserted 100 rows per INSERT. Each insert starts after its rows' DELETE has been checkpointed."""
    con = connect(args.system, QUERY_THREADS, read_only=False)
    ids = np.arange(N)

    batch = ids[ids % 100 == 3]
    con.execute('DELETE FROM t WHERE id % 100 = 3')
    con.execute('CHECKPOINT')
    seconds = insert_and_index(con, args.system, batch)
    report('Ingestion', args.system, rows=len(batch), rows_per_s=len(batch) / seconds)
    before = bytes_written()
    con.execute('CHECKPOINT')
    report('I/O on CHECKPOINT', args.system, rows_inserted=len(batch), gb_written=(bytes_written() - before) / 1e9)

    stream = ids[ids % 100 == 5]
    con.execute('DELETE FROM t WHERE id % 100 = 5')
    con.execute('CHECKPOINT')
    seconds = insert_and_index(con, args.system, stream, rows_per_insert=100)
    report('Ingestion, 100 rows per INSERT', args.system, rows=len(stream), rows_per_s=len(stream) / seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='cmd', required=True)
    for name in ('build', 'search', 'filtered', 'restart', 'first_query', 'ingest'):
        commands.add_parser(name).add_argument('--system', required=True, choices=['vss', 'duckir'])
    args = parser.parse_args()
    prepare_data()
    globals()['cmd_' + args.cmd](args)


if __name__ == '__main__':
    main()

# Development

DuckIR builds on Linux (x86, ARM), macOS, Windows (x86) and Wasm. Our CI builds and tests it for all of them on every push (except Windows with MinGW and multithreaded Wasm). We develop on macOS and Linux, so the steps below are for those two.

DuckIR builds against **DuckDB v2.0.0**, pinned by the `duckdb` submodule. 

## Prerequisites

- A C++17 compiler: GCC or Clang on Linux, Apple Clang on macOS.
- CMake 3.26 or newer.
- Python 3, for the tests, formatting and tidy-check.
- vcpkg, which provides Eigen.
- *[Recommended]* Ninja and ccache, for faster builds. DuckDB picks up ccache by itself.

### macOS

```sh
brew install cmake ninja ccache pkg-config
```

### Linux (Ubuntu, Debian)

```sh
sudo apt install build-essential git cmake ninja-build ccache pkg-config python3 curl zip unzip tar
```

Ubuntu 24.04's CMake (3.28) is new enough. On older distributions, install a newer one with `pip install cmake`.

### vcpkg

We use vcpkg to fetch Eigen. Check out the same commit as our CI, which provides Eigen 5.0.1:

```sh
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
cd ~/vcpkg && git checkout cd61e1e26a038e82d6550a3ebbe0fbbfe7da78e3
./bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=~/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Add the `export` to your shell profile, so that every build finds vcpkg.

If you already have Eigen installed, you can skip vcpkg and point the build to it instead: `EXT_FLAGS="-DEigen3_DIR=<directory with Eigen3Config.cmake>" make`.

## Build

```sh
git clone --recurse-submodules https://github.com/Noorts/DuckIR.git
cd DuckIR
GEN=ninja make
```

`make` runs an optimized `release` build. The first build also compiles DuckDB, so it takes a few minutes. It produces:
- `build/release/extension/duckir/duckir.duckdb_extension`: the loadable extension.
- `build/release/duckdb`: a DuckDB shell with DuckIR already linked in.
- `build/release/test/unittest`: the test runner.

For a debug build, run `DISABLE_SANITIZER=1 make debug`. Without `DISABLE_SANITIZER=1`, DuckDB's own code trips the sanitizers. Other build modes are `reldebug` and `relassert`.

Extra CMake flags go in `EXT_FLAGS`. For example, to build against your local checkouts of PDX and SuperKMeans:

```sh
GEN=ninja EXT_FLAGS="-DFETCHCONTENT_SOURCE_DIR_PDX=$HOME/PDX -DFETCHCONTENT_SOURCE_DIR_SUPERKMEANS=$HOME/SuperKMeans" make
```

For DuckIR usage, see the [README](README.md).

### Clangd

Every build writes a compilation database. For [clangd](https://clangd.llvm.org/) support, link it once from the repository root:

```sh
ln -s ./build/release/compile_commands.json ./
```

## Test

```sh
make test
```

This runs every SQLLogicTest in `test/`, including the slow ones (`.test_slow`). `make test_debug` runs them on the debug build. To run a single test:

```sh
build/release/test/unittest "test/sql/search/index_scan_monotonic_recall.test"
```

See [test/README.md](test/README.md) for more details.

## Format and Tidy

CI checks the formatting of `src` and `test`:

```sh
make format-check
make format-fix
```

These need `black`, `clang-format` 11.0.1 and `cmake-format`. With [uv](https://docs.astral.sh/uv/), you don't have to install them:

```sh
uv run --with black --with clang_format==11.0.1 --with cmake-format python3 duckdb/scripts/format.py --all --fix --noconfirm --workdir $PWD --directories src test
```

A test's `# description:` must fit on one line. The formatter can't handle more.

`make tidy-check` runs clang-tidy over `src`. Use clang-tidy 18: on macOS, newer versions report errors in libc++ headers.

```sh
brew install llvm@18
PATH="$(brew --prefix llvm@18)/bin:$PATH" make tidy-check
```

On Linux, pass the binary instead: `make tidy-check TIDY_BINARY=clang-tidy-18`.

`make clean` removes all builds.

## FAQ

- **I pulled the latest commits, and now the build fails.** Did they bump a submodule (e.g., DuckDB)? Then `git status` shows `modified: duckdb (new commits)`. Run `git submodule update --init --recursive` from the repository root.

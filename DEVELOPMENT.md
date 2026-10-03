# Development

> [!WARNING]
> The instructions below are for Apple Silicon (M1-M5). We've had success with an Apple M4 Pro running macOS Sequoia 15.6.1. Instructions for other systems will be added at a later date.

- [Development](#development)
  - [Prerequisites](#prerequisites)
    - [Install Clang](#install-clang)
    - [Install CMake](#install-cmake)
    - [Install vcpkg](#install-vcpkg)
    - [Install BLAS](#install-blas)
    - [Install OpenMP](#install-openmp)
  - [Build](#build)
    - [Building the Extension](#building-the-extension)
    - [Clangd Language Server Support](#clangd-language-server-support)
  - [Clean, Format, Tidy-Check](#clean-format-tidy-check)
  - [Test](#test)
  - [FAQ](#faq)


## Prerequisites

- Clang (LLVM Clang 18)
- CMake (>= 3.12)
- vcpkg
- A BLAS implementation
- OpenMP

Once you have these you can [build the extension](#build).

### Install Clang

```sh
brew install llvm@18
```

> [!NOTE]
> When building the extension (discussed below), explicitly pass the LLVM 18
> compiler binaries to make sure LLVM@18 is used. For example:
> `CC=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang CXX=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang++ make`.

### Install CMake

Versions >= 3.12 are supported.

```sh
brew install cmake
```

Or instead build and install from the [source](https://gitlab.kitware.com/cmake/cmake).

### Install vcpkg

Install vcpkg prerequisite: pkg-config.

```sh
brew install pkg-config
```

Install vcpkg using the instructions below, or check out the [extension template's instructions for vcpkg](https://github.com/duckdb/extension-template?tab=readme-ov-file#managing-dependencies).

Change to a directory where you want vcpkg to be installed.

```sh
cd ~
```

```sh
git clone https://github.com/Microsoft/vcpkg.git
```

Optionally check out the version we've used.

```sh
git checkout 4334d8b4c8
```

```sh
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
```

```sh
export VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Install BLAS

On Apple Silicon (M1-M5) we rely on the Apple Accelerate framework. This means there is nothing to install.

### Install OpenMP

```sh
brew install libomp
```

Note: You might have to set `OpenMP_ROOT` in your `.zshrc` file.

```sh
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
```

## Build

### Building the Extension

1. Clone the repo and ensure the submodules are also cloned:

    ```sh
    git clone --recurse-submodules https://github.com/Noorts/PDXearch.git
    ```

2. Build the extension (the default is an optimized `release` build):

    ```sh
    cd PDXearch
    ```

    ```sh
    make
    ```

    Other build modes include `debug` and `reldebug`. Include `DISABLE_SANITIZER=1` to avoid errors raised by the DuckDB core itself.

    ```sh
    DISABLE_SANITIZER=1 make debug
    ```

3. [Recommended] For faster builds, install [ccache](https://ccache.dev/) and [ninja](https://ninja-build.org/), and then set the generator when you build:

    ```sh
    brew install ccache ninja
    ```

    ```sh
    GEN=ninja make
    ```

@Noorts personally uses

```sh
GEN=ninja DISABLE_SANITIZER=1 CC=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang CXX=$HOMEBREW_PREFIX/opt/llvm@18/bin/clang++ EXTRA_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" make
```

The built extension artifact can be found at `PDXearch/build/release/extension/pdxearch/pdxearch.duckdb_extension`.

For PDXearch extension usage, please see to the [README.md](README.md).

### Clangd Language Server Support

For
[clangd](https://open-vsx.org/extension/llvm-vs-code-extensions/vscode-clangd)
support in VSCode-based editors, ensure you build the compilation database.

Include the following argument in your build command (each time you build):

```sh
EXTRA_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=1" make
```

Then in the PDXearch root directory, symlink the generated compilation database (once).

```sh
ln -s ./build/release/compile_commands.json ./
```

You might have to manually set the clangd executable path in the VSCode clangd extension configuration.

## Clean, Format, Tidy-Check

```sh
make clean
```

```sh
make format-fix
```

CI runs `make format-check`, which reports the files that need formatting. A test's `# description:` must fit on one
line: the formatter cannot handle more.

Portable `make format-fix` alternative:

```sh
uv run --with black --with clang_format==11.0.1 --with cmake-format duckdb/scripts/format.py --all --fix --noconfirm --directories src test
```

```sh
make tidy-check
```

Tidy-check alternative that explicitly uses LLVM 18:

```sh
make tidy-check TIDY_BINARY=/opt/homebrew/opt/llvm@18/bin/clang-tidy
```

## Test

After building the extension you can run the tests. See
[test/README.md](test/README.md) for more details.

```sh
make test
```

```sh
make test_debug
```

Run one specific test:

```sh
./build/release/"/test/unittest" "test/sql/search/index_scan_uncommon_dimensions.test"
```

## FAQ

- Q: I pulled the latest commits and now I run into compiler errors when I build the extension.
  - A: Did the latest commits include the bump of a submodule? (e.g., DuckDB was updated) If this is the case, then when you run `git status` it will state `"modified: duckdb (new commits)"`. In the root PDXearch directory, run `git submodule update --init --recursive` to ensure your local submodules match the committed versions.

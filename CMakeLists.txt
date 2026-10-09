cmake_minimum_required(VERSION 3.26)

# Set extension name here
set(TARGET_NAME pdxearch)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

include_directories(src/include)
set(EXTENSION_SOURCES src/pdxearch_extension.cpp)
add_subdirectory(src/index)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
# Found before PDX so that PDX and SuperKMeans reuse vcpkg's Eigen instead of their bundled copies
find_package(Eigen3 REQUIRED)

# PDX::PDX carries the PDX and SuperKMeans headers, their GEMM backend (Accelerate on macOS, Eigen elsewhere)
# and the SIMD flags. The extension is a distributed binary: portable flags (AVX2 on x86_64) and no FFTW.
# PDX's parallel loops run on DuckDB's scheduler (DuckDBParallelExecutor) or serially, so the libraries'
# default executor is the serial one: they start no threads of their own.
set(PDX_PORTABLE ON CACHE BOOL "" FORCE)
set(PDX_SKIP_FFTW ON CACHE BOOL "" FORCE)
set(SKMEANS_EXECUTOR serial CACHE STRING "" FORCE)
# -DFETCHCONTENT_SOURCE_DIR_PDX=<checkout> builds against a local PDX.
include(FetchContent)
FetchContent_Declare(
  pdx
  GIT_REPOSITORY https://github.com/cwida/pdx.git
  GIT_TAG 413d6ec5a810bfb8c52d295f58c73297229bb01a)
FetchContent_MakeAvailable(pdx)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Kept out of the exported interface: the static extension is in DuckDB's export set, PDX and its
# dependencies are not
target_link_libraries(${EXTENSION_NAME} $<BUILD_LOCAL_INTERFACE:PDX::PDX>)
target_link_libraries(${LOADABLE_EXTENSION_NAME} PDX::PDX)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

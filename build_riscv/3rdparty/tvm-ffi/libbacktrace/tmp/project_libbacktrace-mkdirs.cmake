# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/var/home/hoku/repo/tvm/3rdparty/tvm-ffi/cmake/Utils/../../3rdparty/libbacktrace")
  file(MAKE_DIRECTORY "/var/home/hoku/repo/tvm/3rdparty/tvm-ffi/cmake/Utils/../../3rdparty/libbacktrace")
endif()
file(MAKE_DIRECTORY
  "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace"
  "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace"
  "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace/tmp"
  "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace/src/project_libbacktrace-stamp"
  "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace/src"
  "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace/logs"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace/src/project_libbacktrace-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/var/home/hoku/repo/tvm/build_riscv/3rdparty/tvm-ffi/libbacktrace/src/project_libbacktrace-stamp${cfgdir}") # cfgdir has leading slash
endif()

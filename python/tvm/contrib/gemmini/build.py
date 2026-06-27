# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, dangerous-default-value, arguments-differ
# ruff: noqa: F821
"""Driver for building a Relax module for Gemmini offload."""

import logging
import os

from tvm_ffi import register_global_func

import tvm


logger = logging.getLogger("gemmini")


def has_gemmini():
    """Returns true if the Gemmini custom codegen is available"""
    return tvm.get_global_func("relax.ext.gemmini", True) is not None


def _get_gemmini_path():
    invalid_paths = []
    for rel in ["../../../../", "../../../", "../../"]:
        tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel)
        gemmini_path = os.path.join(tvm_root, "3rdparty/gemmini")
        if os.path.exists(gemmini_path):
            return gemmini_path
        invalid_paths.append(gemmini_path)
    raise AssertionError(f"The Gemmini root directory not found in: {invalid_paths}")


def _get_tvm_root():
    for rel in ["../../../../", "../../../", "../../"]:
        tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel)
        if os.path.exists(os.path.join(tvm_root, "3rdparty/tvm-ffi/include")):
            return os.path.realpath(tvm_root)
    raise AssertionError("TVM root not found")


def _get_gemmini_compile_options():
    gemmini_root = _get_gemmini_path()
    gemmini_include = os.path.join(gemmini_root, "include")
    gemmini_riscv_tests_include = os.path.join(gemmini_root, "riscv-tests")
    gemmini_riscv_tests_env_include = os.path.join(gemmini_root, "riscv-tests/env")
    gemmini_riscv_tests_benchmarks_common_include = os.path.join(gemmini_root, "riscv-tests/benchmarks/common")

    tvm_root = _get_tvm_root()
    tvm_ffi_include = os.path.join(tvm_root, "3rdparty/tvm-ffi/include")
    dlpack_include = os.path.join(tvm_root, "3rdparty/tvm-ffi/3rdparty/dlpack/include")

    kwargs = {}
    kwargs["cc"] = "riscv64-unknown-linux-gnu-g++"
    kwargs["options"] = [
        "-c",
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-std=gnu++17",
        "-O2",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc",
        "-Wa,-march=rv64gc",
        "-lm",
        "-lgcc",
        f"-I{tvm_ffi_include}",
        f"-I{dlpack_include}",
        f"-I{gemmini_root}",
        f"-I{gemmini_include}",
        f"-I{gemmini_riscv_tests_include}",
        f"-I{gemmini_riscv_tests_env_include}",
        f"-I{gemmini_riscv_tests_benchmarks_common_include}",
        "-DPRINT_TILE=0",
    ]
    return kwargs


@register_global_func("contrib.gemmini.compile")
def compile_gemmini_module(c_source_module, options):
    """Compile all Gemmini kernels in the given C-source module.

    Parameters
    ----------
    c_source_module: runtime.Module
        A C-source module containing Gemmini kernels.

    Returns
    -------
    rt_mod : runtime.Module
        A runtime module where all gemmini kernels have been compiled.
    """
    tmp_dir = options.get("tmp_dir", "./gemmini_out")
    os.makedirs(tmp_dir, exist_ok=True)

    function_names = c_source_module.get_function("get_func_names")()
    compile_options = _get_gemmini_compile_options()
    lib_path = os.path.join(tmp_dir, "gemmini.o")
    logger.info("Compiling generated Gemmini code")
    c_source_module.export_library(lib_path, workspace_dir=tmp_dir, **compile_options)

    # Recover static library
    return tvm.runtime.load_static_library(lib_path, function_names)

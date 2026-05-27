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
"""Driver for building Gemmini C code from a Relax module."""

import os
import subprocess

from tvm_ffi import register_global_func


def has_gemmini():
    """Returns true if the Gemmini custom codegen is available."""
    import tvm
    return tvm.get_global_func("relax.ext.gemmini", True) is not None


@register_global_func("contrib.gemmini.compile")
def compile_gemmini_module(c_source_module, options):
    """Write generated Gemmini C code to a file and optionally cross-compile.

    Parameters
    ----------
    c_source_module : runtime.Module
        A C-source module containing generated Gemmini calls.

    options : dict
        Compilation options:
          "out_dir"  : Directory to write generated .c file (default: "./gemmini_out")
          "cc"       : Cross-compiler to use (default: "riscv64-unknown-elf-gcc")
          "compile"  : Whether to actually compile (default: False)
          "gemmini_include" : Path to gemmini include directory

    Returns
    -------
    c_source_module : runtime.Module
        The same C-source module (compilation result when compile=True).
    """
    out_dir = options.get("out_dir", "./gemmini_out")
    cc = options.get("cc", "riscv64-unknown-elf-gcc")
    do_compile = options.get("compile", False)

    os.makedirs(out_dir, exist_ok=True)

    src = c_source_module.inspect_source()
    src_path = os.path.join(out_dir, "gemmini_model.c")
    with open(src_path, "w") as f:
        f.write(src)
    print(f"[Gemmini] Generated C code written to: {src_path}")

    if do_compile:
        gemmini_include = options.get("gemmini_include", ".")
        obj_path = os.path.join(out_dir, "gemmini_model.o")
        cmd = [
            cc,
            "-O2",
            f"-I{gemmini_include}",
            "-c", src_path,
            "-o", obj_path,
        ]
        print(f"[Gemmini] Compiling: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Gemmini compilation failed:\n{result.stderr}"
            )
        print(f"[Gemmini] Compiled to: {obj_path}")

    return c_source_module

  riscv64-unknown-linux-gnu-g++ \
      -march=rv64gc -mcmodel=medany \
      -DPREALLOCATE=1 -DMULTITHREAD=1 \
      -O2 -ffast-math -fno-common -fno-builtin-printf \
      -fno-tree-loop-distribute-patterns \
      -I include \
	  -I 3rdparty/gemmini \
      -I 3rdparty/tvm-ffi/include \
      -I 3rdparty/tvm-ffi/3rdparty/dlpack/include \
      apps/braggnn_gemmini/inference.cc \
      -o apps/braggnn_gemmini/inference \
      -Wl,--whole-archive \
        build_riscv/libtvm_runtime.a \
        build_riscv/lib/libtvm_ffi_static.a \
      -Wl,--no-whole-archive \
      build_riscv/3rdparty/tvm-ffi/libbacktrace/lib/libbacktrace.a \
      -static-libstdc++ -static-libgcc \
      -lm -ldl -lpthread

  riscv64-unknown-linux-gnu-g++ \
      -march=rv64gc -mcmodel=medany \
      -DPREALLOCATE=1 -DMULTITHREAD=1 \
      -O2 -ffast-math -fno-common -fno-builtin-printf \
      -fno-tree-loop-distribute-patterns \
      -I include \
      -I 3rdparty/tvm-ffi/include \
      -I 3rdparty/tvm-ffi/3rdparty/dlpack/include \
      apps/braggnn_gemmini/inference.cc \
      -o apps/braggnn_gemmini/inference \
      -L build_riscv \
      -L build_riscv/lib \
      -ltvm_runtime \
      -ltvm_ffi \
      -lm -lstdc++ -lgcc \
      -Wl,-rpath,'$ORIGIN'

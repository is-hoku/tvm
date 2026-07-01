set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   /var/home/hoku/repo/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /var/home/hoku/repo/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_C_FLAGS   "-march=rv64gc -mcmodel=medany -DPREALLOCATE=1 -DMULTITHREAD=1 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -std=gnu99")
set(CMAKE_CXX_FLAGS "-march=rv64gc -mcmodel=medany -DPREALLOCATE=1 -DMULTITHREAD=1 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

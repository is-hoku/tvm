#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>

#include <cstdio>

static inline unsigned long long read_cycles() {
  unsigned long long cycles;
  asm volatile("csrr %0, cycle" : "=r"(cycles));
  return cycles;
}

#define NUM_ITERS 10

static float kPatch0[11][11] = {
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.748000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.646286f, 0.753714f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
  {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
};


// Refrence implementation is in /3rdparty/tvm-ffi/examples/quickstart/load/load_cpp.cc
struct CPUAllocator {
  void AllocData(DLTensor* tensor) {
	int64_t numel = 1;
	for (int i = 0; i < tensor->ndim; ++i) {
		numel *= tensor->shape[i];
	}
    tensor->data = std::malloc(numel * sizeof(float));
  }
  void FreeData(DLTensor* tensor) { std::free(tensor->data); }
};

int main(int argc, char* argv[]){
	// Reference implementation is in /src/runtime/disco/builtin.cc
	std::string path = argc > 1 ? argv[1] : "./braggnn.so";
	tvm::ffi::Module ex = tvm::ffi::Module::LoadFromFile(path);
	//tvm::ObjectPtr<tvm::runtime::vm::VirtualMachine> vm = tvm::runtime::vm::VirtualMachine::Create();

	tvm::ffi::Optional<tvm::ffi::Function> vm_load_executable = ex->GetFunction("vm_load_executable");
	if (!vm_load_executable.has_value()) {
    TVM_FFI_THROW(ValueError)
        << "File `" << path
        << "` does not contain a Relax VM executable";
		return 1;
	}
	auto mod = (*vm_load_executable)().cast<tvm::ffi::Module>();
	tvm::ffi::Optional<tvm::ffi::Function> vm_initialization = mod->GetFunction("vm_initialization");

  if (!vm_initialization.has_value()) {
    TVM_FFI_THROW(ValueError)
        << "File `" << path
        << "` is not built by RelaxVM, because `vm_initialization` does not exist";
	return 1;
  }
  DLDevice dev{kDLCPU, 0};
  (*vm_initialization)(static_cast<int>(dev.device_type), static_cast<int>(dev.device_id),
                       static_cast<int>(tvm::runtime::AllocatorType::kPooled),
					   static_cast<int>(kDLCPU), 0,
                       static_cast<int>(tvm::runtime::AllocatorType::kPooled));

  tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(CPUAllocator(), {1, 1, 11, 11}, DLDataType{kDLFloat, 32, 1}, dev);
  std::memcpy(input.data_ptr(), kPatch0, sizeof(kPatch0));

  tvm::ffi::Function main_func = mod->GetFunction("main").value();

  printf("==============================================\n");
  printf("BraggNN Gemmini through TVM Relax VM\n");
  printf("==============================================\n");
  printf("\nInput shape: [1, 1, 11, 11]\n");
  printf("Running %d inferences (%d warmup + %d measured)\n",
         NUM_ITERS + 1, 1, NUM_ITERS);

  // Warmup: drop first run to warm caches
  printf("\n--- Warmup ---\n");
  unsigned long long start = read_cycles();
  tvm::ffi::Tensor output = main_func(input).cast<tvm::ffi::Tensor>();
  unsigned long long end = read_cycles();
  {
    const float* out_data = static_cast<const float*>(output.data_ptr());
    printf("cycles: %llu\n", end - start);
    printf("(x, y) = (%.4f, %.4f)\n", out_data[0], out_data[1]);
  }

  // Measured iterations
  unsigned long long total_cycles = 0;
  for (int i = 0; i < NUM_ITERS; i++) {
    printf("\n--- Inference %d/%d ---\n", i + 1, NUM_ITERS);

    start = read_cycles();
    asm volatile(".word 0x8013");  // TracerV start trigger
    output = main_func(input).cast<tvm::ffi::Tensor>();
    asm volatile(".word 0x10013"); // TracerV end trigger
    end = read_cycles();

    unsigned long long elapsed = end - start;
    total_cycles += elapsed;

    const float* out_data = static_cast<const float*>(output.data_ptr());
    printf("cycles: %llu\n", elapsed);
    printf("(x, y) = (%.4f, %.4f)\n", out_data[0], out_data[1]);
  }

  printf("\n==============================================\n");
  printf("Avg cycles over %d runs: %llu\n", NUM_ITERS,
         total_cycles / NUM_ITERS);
  printf("==============================================\n");
  printf("BraggNN inference completed successfully\n");
  printf("==============================================\n");

  return 0;
}

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/backend/contrib/gemmini/codegen.cc
 * \brief Implementation of the Gemmini code generator for Relax.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/name_supply.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/module.h>

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "../codegen_c/codegen_c.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

std::string EmitSignature(const std::vector<Output>& out, const std::string& func_id,
                          const std::vector<std::string>& arg_names) {
  std::ostringstream code_stream_;
  code_stream_ << "void " << func_id << "_(";
  for (const auto& arg_name : arg_names) {
    code_stream_ << "DLTensor* " << arg_name << ", ";
  }
  for (size_t i = 0; i < out.size() - 1; ++i) {
    code_stream_ << "DLTensor* out" << i << ", ";
  }
  code_stream_ << "DLTensor* out" << out.size() - 1 << ")";
  return code_stream_.str();
}

constexpr int kGemminiNumClockCycles = 14;

ffi::Module Finalize(const std::string& code, const ffi::Array<ffi::String>& func_names) {
  TVM_FFI_ICHECK(!func_names.empty())
      << "Should only create Gemmini CSourceModule if there is at least one Gemmini partition";

  std::ostringstream default_headers;
  default_headers << "#include <tvm/ffi/c_api.h>\n";
  default_headers << "#include <tvm/ffi/function.h>\n";
  default_headers << "#include <dlpack/dlpack.h>\n";
  default_headers << "#include <assert.h>\n";
  default_headers << "#include <stddef.h>\n";
  default_headers << "#include <stdint.h>\n";
  default_headers << "#include <stdio.h>\n";
  default_headers << "#include <stdlib.h>\n";
  default_headers << "#ifndef BAREMETAL\n";
  default_headers << "#include <sys/mman.h>\n";
  default_headers << "#endif\n";
  default_headers << "extern \"C\" {\n";
  default_headers << "#include \"include/gemmini_params.h\"\n";
  default_headers << "#include \"include/gemmini.h\"\n";
  default_headers << "}\n";

  default_headers << "static inline DLTensor* GEMMINI_READ_TENSOR(const TVMFFIAny* v) {\n";
  default_headers << "  if (v->type_index == kTVMFFIDLTensorPtr) return (DLTensor*)(v->v_ptr);\n";
  default_headers << "  return (DLTensor*)((char*)(v->v_obj) + sizeof(TVMFFIObject));\n";
  default_headers << "}\n";
  // GEMMINI_DATA folds the offset into a byte pointer for casting.
  default_headers << "#define GEMMINI_DATA(t) ((void*)((char*)((t)->data) + (t)->byte_offset))\n";


  // For debugging: measure per-Gemmini-call elapsed cycles.
  default_headers << "static inline unsigned long long read_cycles() {\n"
  				  << "	uint64_t cycles;\n"
				  << "	asm volatile (\"rdcycle %0\" : \"=r\" (cycles));\n"
				  << "	return cycles;\n"
				  << "}\n";
  default_headers << "#define GEMMINI_NUM_CLOCK_CYCLES " << kGemminiNumClockCycles << "\n";
  default_headers << "static unsigned long long "
                      "gemmini_clock_cycles[GEMMINI_NUM_CLOCK_CYCLES];\n";
  default_headers << "static int gemmini_clock_cycle_idx = 0;\n";

  // Host-callable accessor: copies the recorded per-call cycle counts into the
  // caller-supplied uint64 output tensor (shape [GEMMINI_NUM_CLOCK_CYCLES]) and
  // resets the counter so the next inference starts recording from index 0.
  default_headers << "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n";
  default_headers << "TVM_FFI_DLL_EXPORT int __tvm_ffi_gemmini_clock_cycles(\n";
  default_headers << "    void* handle, const TVMFFIAny* args,\n";
  default_headers << "    int32_t num_args, TVMFFIAny* result) {\n";
  default_headers << "  DLTensor* out0 = GEMMINI_READ_TENSOR(&args[0]);\n";
  default_headers << "  unsigned long long* dst = (unsigned long long*)GEMMINI_DATA(out0);\n";
  default_headers << "  for (int k = 0; k < GEMMINI_NUM_CLOCK_CYCLES; k++) "
                      "dst[k] = gemmini_clock_cycles[k];\n";
  default_headers << "  gemmini_clock_cycle_idx = 0;\n";
  default_headers << "  return 0;\n}\n";
  default_headers << "#ifdef __cplusplus\n}\n#endif\n";

  ffi::Array<ffi::String> all_func_names = func_names;
  all_func_names.push_back("gemmini_clock_cycles");

  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.CSourceModuleCreate");
  VLOG(1) << "Generated Gemmini code:" << std::endl << code;
  return pf(default_headers.str() + code, "c", all_func_names,
            /*const_vars=*/ffi::Array<ffi::String>())
      .cast<ffi::Module>();
}


using OutputType = std::vector<Output>;

class CodegenGemmini : public relax::MemoizedExprTranslator<OutputType>,
                       public relax::contrib::CodegenCBase {
 public:
  CodegenGemmini(const std::string& id, const ffi::Map<Var, Expr>& bindings,
                 const ffi::Map<Constant, ffi::String>& constant_names)
      : ext_func_id_(id), bindings_(bindings), constant_names_(constant_names) {}

  void AddParm(Var param) {
    ext_func_args_.push_back(param);
    auto v_name = name_sup_->FreshName(param->name_hint());
    var_name_map_[param.get()] = v_name;
  }

  std::string JIT(const OutputType& out) final {
    std::vector<std::string> arg_names;
    for (const auto& arg : ext_func_args_) {
      arg_names.push_back(var_name_map_.at(arg.get()));
    }

    code_stream_ << EmitSignature(out, ext_func_id_, arg_names) << "{\n";
    this->EnterScope();
    for (auto decl : buf_decl_) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : ext_func_body_) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }
    this->ExitScope();
    code_stream_ << "}\n";

    code_stream_ << "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n";
    code_stream_ << "TVM_FFI_DLL_EXPORT int __tvm_ffi_" << ext_func_id_ << "(\n";
    code_stream_ << "    void* handle, const TVMFFIAny* args,\n";
    code_stream_ << "    int32_t num_args, TVMFFIAny* result) {\n";
    for (size_t i = 0; i < arg_names.size(); i++) {
      code_stream_ << "  DLTensor* arg" << i << " = GEMMINI_READ_TENSOR(&args[" << i << "]);\n";
    }
    for (size_t i = 0; i < out.size(); i++) {
      code_stream_ << "  DLTensor* out" << i << " = GEMMINI_READ_TENSOR(&args["
                   << (arg_names.size() + i) << "]);\n";
    }
    code_stream_ << "  " << ext_func_id_ << "_(";
    for (size_t i = 0; i < arg_names.size(); i++) {
      code_stream_ << "arg" << i << ", ";
    }
    for (size_t i = 0; i < out.size() - 1; i++) {
      code_stream_ << "out" << i << ", ";
    }
    code_stream_ << "out" << out.size() - 1 << ");\n";
    code_stream_ << "  return 0;\n}\n";
    code_stream_ << "#ifdef __cplusplus\n}\n#endif\n";

    return code_stream_.str();
  }

 protected:
  OutputType VisitExpr_(const VarNode* node) final {
    Output output;
    auto it = var_name_map_.find(node);
    TVM_FFI_ICHECK(it != var_name_map_.end());
    output.name = it->second;
    return {output};
  }

  OutputType VisitExpr_(const CallNode* call) final {
    const auto* fn_var = call->op.as<VarNode>();
    TVM_FFI_ICHECK(fn_var);
    const auto func = Downcast<Function>(bindings_[ffi::GetRef<Var>(fn_var)]);
    const auto pattern_name_opt = func->GetAttr<ffi::String>(attr::kComposite);
    TVM_FFI_ICHECK(pattern_name_opt) << "Only composite function is supported for Gemmini.";
    auto ret = GenerateBody(call, pattern_name_opt.value(), func);
    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

  OutputType VisitExpr_(const FunctionNode* fn) final {
    TVM_FFI_ICHECK(fn->GetAttr<ffi::String>(attr::kComposite).has_value())
        << "JSON runtime only supports composite functions";
    // FunctionNode should be handled by the caller.
    return {};
  }

  OutputType VisitBinding(const Binding& binding) {
    OutputType outputs;
    if (const auto* node = binding.as<VarBindingNode>()) {
      auto from_b = VisitBinding_(node);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented type: " << binding->GetTypeKey();
    }
    return outputs;
  }

  OutputType VisitBindingBlock(const BindingBlock& block) {
    OutputType outputs;
    if (const auto* node = block.as<DataflowBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      outputs.insert(outputs.end(), from_bb.begin(), from_bb.end());
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented type: " << block->GetTypeKey();
    }
    return outputs;
  }

  OutputType VisitBindingBlock_(const BindingBlockNode* block) {
    OutputType outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  OutputType VisitBindingBlock_(const DataflowBlockNode* block) {
    OutputType outputs;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      outputs.insert(outputs.end(), from_b.begin(), from_b.end());
    }
    return outputs;
  }

  OutputType VisitExpr_(const SeqExprNode* op) final {
    OutputType outputs;

    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }

    auto from_body = VisitExpr(op->body);
    outputs.insert(outputs.end(), from_body.begin(), from_body.end());

    return outputs;
  }

 private:
  ffi::Array<ffi::String> GetArgumentNames(const CallNode* call) {
    ffi::Array<ffi::String> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& f,
		  						  const Function& func) {
    //auto func_args = GetArgumentNames(call);
    auto struct_info = GetStructInfo(ffi::GetRef<Call>(call));
	//attrs = func->attrs->dict;

	const auto* out_sinfo = struct_info.as<TensorStructInfoNode>();

    const std::string out_name = "out" + std::to_string(buf_idx_++);
    Output output;
    output.name = out_name;
    output.dtype = GetDtypeString(out_sinfo);
    output.need_copy = false;
    GenerateBodyOutput ret;
    ret.outputs.push_back(output);

	if (f == "gemmini.conv2d" ||
		f == "gemmini.conv2d_leakyrelu") {
		ret.decl = EmitGemminiConv2d(call, func, out_name, f);
	} else if (f == "gemmini.matmul" ||
			   f == "gemmini.fc" ||
			   f == "gemmini.fc_leakyrelu") {
		ret.decl = EmitGemminiMatmul(call, func, out_name, f);
	} else if (f == "gemmini.matmul_self_attention") {
			ret.decl = EmitGemminiSelfAttention(call, func, out_name, f);
		} else if (f == "gemmini.resadd_leakyrelu") {
		ret.decl = EmitGemminiResadd(call, func, out_name, f);
	} else {
		TVM_FFI_THROW(InternalError) << "Unsupported Gemmini op: " << f;
	}

    return ret;
  }


  std::string EmitDebugDump(const std::string& out, const std::string& tag, int64_t numel) {
	  std::ostringstream ss;
	  ss << "\n  for (int64_t dbg_i = 0; dbg_i < " << numel << "; dbg_i++) { "
		 << "printf(\"[dbg] " << tag << " " << out << " e%lld=%d\\n\", (long long)dbg_i, "
		 << "(int)((elem_t*)GEMMINI_DATA(" << out << "))[dbg_i]); }";
	  return ss.str();
  }


  std::string EmitDebugDumpRaw(const std::string& arr, const std::string& tag, int64_t numel) {
	  std::ostringstream ss;
	  ss << "\n  for (int64_t dbg_i = 0; dbg_i < " << numel << "; dbg_i++) { "
		 << "printf(\"[dbg] " << tag << " " << arr << " e%lld=%d\\n\", (long long)dbg_i, "
		 << "(int)" << arr << "[dbg_i]); }";
	  return ss.str();
  }


  std::string EmitDebugDumpAcc(const std::string& out, const std::string& tag, int64_t numel) {
	  std::ostringstream ss;
	  ss << "\n  for (int64_t dbg_i = 0; dbg_i < " << numel << "; dbg_i++) { "
		 << "printf(\"[dbg] " << tag << " " << out << " e%lld=%d\\n\", (long long)dbg_i, "
		 << "(int)((acc_t*)GEMMINI_DATA(" << out << "))[dbg_i]); }";
	  return ss.str();
  }


  std::string EmitGemminiConv2d(const CallNode* call, const Function& func, const std::string& out, const std::string& func_name) {
      ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t i_idx = arg_idx["input"]->value;
	  int64_t w_idx = arg_idx["weight"]->value;
	  int64_t b_idx = arg_idx["bias"]->value;

	  const auto* in_sinfo = GetStructInfo(func->params[i_idx]).as<TensorStructInfoNode>();
	  auto in_shape = backend::GetIntShape(in_sinfo->shape.value().as<ShapeExprNode>()->values);

	  const auto* w_sinfo = GetStructInfo(func->params[w_idx]).as<TensorStructInfoNode>();
	  auto w_shape = backend::GetIntShape(w_sinfo->shape.value().as<ShapeExprNode>()->values);

	  int64_t batch_size   = in_shape[0];
	  int64_t in_channels  = in_shape[1];
	  int64_t in_rows      = in_shape[2];
	  int64_t in_cols      = in_shape[3];
	  int64_t out_channels = w_shape[3];

	  std::string act = attrs.at("act").try_cast<std::string>().value();
	  double acc_scale = attrs.at("acc_scale").try_cast<double>().value();
	  int stride = attrs.at("stride").try_cast<int>().value();
	  int padding = attrs.at("padding").try_cast<int>().value();
	  int kernel_dim = attrs.at("kernel_dim").try_cast<int>().value();

	  int64_t out_rows = (in_rows + 2 * padding - kernel_dim) / stride + 1;
	  int64_t out_cols = (in_cols + 2 * padding - kernel_dim) / stride + 1;

	  std::ostringstream ss;

	  // cause of a divergence, separately from the conv computation itself.
	  //int64_t input_numel = batch_size * in_channels * in_rows * in_cols;
	  //ss << EmitDebugDump(func_args[i_idx], func_name + ".input", input_numel);
	  //int64_t weight_numel = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3];
	  //ss << EmitDebugDump(func_args[w_idx], func_name + ".weight", weight_numel);
	  //ss << EmitDebugDumpAcc(func_args[b_idx], func_name + ".bias", out_channels);

	//tiled_conv_auto(int batch_size, int in_row_dim, int in_col_dim,
    //                int in_channels, int out_channels, int out_row_dim,
    //                int out_col_dim, int stride, int input_dilation,
    //                int kernel_dilation, int padding, int kernel_dim,
    //                bool wrot180, bool trans_output_1203,
    //                bool trans_input_3120, bool trans_weight_1203,
    //                bool trans_weight_0132,

    //                const elem_t *input, const elem_t *weights,
    //                const acc_t *bias, elem_t *output,

    //                int act, acc_scale_t scale, int pool_size,
    //                int pool_stride, int pool_padding,

    //                enum tiled_matmul_type_t tiled_conv_type) {
	  ss << "unsigned long long start = read_cycles();\n"
	     << "tiled_conv_auto("
		 << batch_size << ", "
		 << in_rows << ", "
		 << in_cols << ", "
		 << in_channels << ", "
		 << out_channels << ", "
		 << out_rows << ", "
		 << out_cols << ", "
		 << stride << ", "
		 << "1, "
		 << "1, "
		 << padding << ", "
		 << kernel_dim << ", "
		 << "false, false, false, false, false, "
	  	 << "(elem_t*)GEMMINI_DATA(" << func_args[i_idx] << "), "
	  	 << "(elem_t*)GEMMINI_DATA(" << func_args[w_idx] << "), "
	  	 << "(acc_t*)GEMMINI_DATA(" << func_args[b_idx] << "), "
	  	 << "(elem_t*)GEMMINI_DATA(" << out << "), "
		 << act << ", "
		 << acc_scale << ", "
		 << "0, 0, 0, "
		 << "WS);\n"
		 // tiled_conv_auto does NOT fence internally unlike tiled_matmul_auto /
		 // tiled_resadd_auto, which fence on exit.
		 << "gemmini_fence();"
	     << "unsigned long long end = read_cycles();\n"
		 << "gemmini_clock_cycles[gemmini_clock_cycle_idx++] = end - start;\n";

	  //int64_t numel = batch_size * out_channels * out_rows * out_cols;
	  //ss << EmitDebugDump(out, func_name, numel);

	  return ss.str();
  }

  std::string EmitGemminiMatmul(const CallNode* call, const Function& func, const std::string& out, const std::string func_name) {
      ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t in_idx = arg_idx["input"]->value;
	  int64_t w_idx = arg_idx["weight"]->value;

	  const auto* in_sinfo = GetStructInfo(func->params[in_idx]).as<TensorStructInfoNode>();
	  auto in_shape = backend::GetIntShape(in_sinfo->shape.value().as<ShapeExprNode>()->values);

	  const auto* w_sinfo = GetStructInfo(func->params[w_idx]).as<TensorStructInfoNode>();
	  auto w_shape = backend::GetIntShape(w_sinfo->shape.value().as<ShapeExprNode>()->values);

	  std::string bias = "NULL";
	  if (arg_idx.count("bias")) {
		  bias = "(acc_t*)GEMMINI_DATA(" + std::string(func_args[arg_idx["bias"]->value]) + ")";
	  }

	  //auto* out_sinfo = GetStructInfo(ffi::GetRef<Call>(call)).as<TensorStructInfoNode>();
	  //auto out_shape = backend::GetIntShape(out_sinfo->shape.value().as<ShapeExprNode>()->values);

	  // The reshape and permute_dims ops between the conv and the matmul were absorbed into this composite, so must set H*W as rows and C as cols
	  auto matrix_dims = [](const std::vector<int64_t>& s) -> std::pair<int64_t, int64_t> {
		  if (s.size() == 4) return {s[2] * s[3], s[1]};  // NHWC-physical conv output: (H*W, C)
		  return {s[s.size() - 2], s[s.size() - 1]};       // 2D/3D operand: (rows, cols)
	  };
	  std::pair<int64_t, int64_t> a_dims = matrix_dims(in_shape);
	  std::pair<int64_t, int64_t> b_dims = matrix_dims(w_shape);

	  int64_t dim_I = a_dims.first;
	  int64_t dim_K = a_dims.second;
	  bool transpose_B = (b_dims.first != dim_K) || (b_dims.first == b_dims.second);
	  int64_t dim_J = transpose_B ? b_dims.first : b_dims.second;
	  int64_t stride_A = dim_K;
	  int64_t stride_B = transpose_B ? dim_K : dim_J;
	  int64_t stride_C = dim_J;
	  int64_t stride_D = dim_J;

	  std::string act = attrs.at("act").try_cast<std::string>().value();
	  double acc_scale = attrs.at("acc_scale").try_cast<double>().value();
	  double output_scale = acc_scale;
	  if (act == "SOFTMAX"){
		  output_scale = 1.0;
	  }

	  std::ostringstream ss;

	  //int64_t weight_numel = w_shape[0] * w_shape[1];
	  //for (size_t i = 2; i < w_shape.size(); i++) weight_numel *= w_shape[i];
	  //ss << EmitDebugDump(func_args[w_idx], func_name + ".weight", weight_numel);
	  //if (arg_idx.count("bias")) {
	  //    ss << EmitDebugDumpAcc(func_args[arg_idx["bias"]->value], func_name + ".bias", dim_J);
	  //}

	//tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K, const elem_t *A,
	//                  const elem_t *B, const void *D, void *C, size_t stride_A,
	//                  size_t stride_B, size_t stride_D, size_t stride_C,
	//                  scale_t A_scale_factor, scale_t B_scale_factor,
	//                  scale_acc_t D_scale_factor, int act, acc_scale_t scale,
	//                  acc_scale_t bert_scale, bool repeating_bias, bool transpose_A,
	//                  bool transpose_B, bool full_C, bool low_D, uint8_t weightA,
	//                  enum tiled_matmul_type_t tiled_matmul_type) {
	  ss << "unsigned long long start = read_cycles();\n"
	     << "tiled_matmul_auto("
		 << dim_I << ", "
		 << dim_J << ", "
		 << dim_K << ", "
		 << "(elem_t*)GEMMINI_DATA(" << func_args[in_idx] << "), "
		 << "(elem_t*)GEMMINI_DATA(" << func_args[w_idx] << "), "
		 << bias << ", "
		 << "(elem_t*)GEMMINI_DATA(" << out << "), "
		 << stride_A << ", " // stride_A
		 << stride_B << ", " // stride_B
		 << stride_D << ", " // stride_D
		 << stride_C << ", " // stride_C
		 << "MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, "
		 << act << ", "
		 << output_scale << ", " // scale
		 << acc_scale << ", " // bert_scale
		 << "false, false, "
		 << transpose_B << ", "
		 << "false, false, 0, "
		 << "WS);\n"
	     << "unsigned long long end = read_cycles();\n"
		 << "gemmini_clock_cycles[gemmini_clock_cycle_idx++] = end - start;\n";

	  // --- debug instrumentation: dump a window of this layer's output ---
	  //int64_t numel = dim_I * dim_J;
	  //ss << EmitDebugDump(out, func_name, numel);

	  return ss.str();
  }


  std::string EmitGemminiSelfAttention(const CallNode* call, const Function& func, const std::string& out, const std::string& func_name) {
	  ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t theta_idx = arg_idx["input_theta"]->value;
	  int64_t phi_idx = arg_idx["input_phi"]->value;
	  int64_t g_idx = arg_idx["input_g"]->value;

	  const auto* theta_sinfo = GetStructInfo(func->params[theta_idx]).as<TensorStructInfoNode>();
	  auto theta_shape = backend::GetIntShape(theta_sinfo->shape.value().as<ShapeExprNode>()->values);
	  const auto* phi_sinfo = GetStructInfo(func->params[phi_idx]).as<TensorStructInfoNode>();
	  auto phi_shape = backend::GetIntShape(phi_sinfo->shape.value().as<ShapeExprNode>()->values);
	  const auto* g_sinfo = GetStructInfo(func->params[g_idx]).as<TensorStructInfoNode>();
	  auto g_shape = backend::GetIntShape(g_sinfo->shape.value().as<ShapeExprNode>()->values);

	  auto matrix_dims = [](const std::vector<int64_t>& s) -> std::pair<int64_t, int64_t> {
		  if (s.size() == 4) return {s[2] * s[3], s[1]};  // NHWC-physical conv output: (H*W, C)
		  return {s[s.size() - 2], s[s.size() - 1]};       // 2D/3D operand: (rows, cols)
	  };

	  // Stage 1 dims: raw logits = theta @ phi (contraction over the hidden channel dim).
	  std::pair<int64_t, int64_t> theta_dims = matrix_dims(theta_shape);
	  std::pair<int64_t, int64_t> phi_dims = matrix_dims(phi_shape);
	  int64_t dim_I0 = theta_dims.first;
	  int64_t dim_K0 = theta_dims.second;
	  bool transpose_B0 = (phi_dims.first != dim_K0);
	  int64_t dim_J0 = transpose_B0 ? phi_dims.first : phi_dims.second;
	  int64_t stride_A0 = dim_K0;
	  int64_t stride_B0 = transpose_B0 ? dim_K0 : dim_J0;

	  double acc_scale0 = attrs.at("acc_scale0").try_cast<double>().value();
	  double bert_scale = attrs.at("bert_scale").try_cast<double>().value();
	  double acc_scale1 = attrs.at("acc_scale1").try_cast<double>().value();

	  // Stage 2 dims: attention = logits @ identity_N (SOFTMAX). N = dim_J0, the softmax
	  // axis width -- tiled_matmul_auto's dim_K must equal this so the norm unit's
	  // row-wise reduction spans the full softmax row.
	  int64_t N = dim_J0;

	  // Stage 3 dims: out = attention @ g -- identical derivation to the old
	  // matmul_transpose composite (attention rows = dim_I0, contraction = N).
	  std::pair<int64_t, int64_t> g_dims = matrix_dims(g_shape);
	  int64_t dim_I2 = dim_I0;
	  int64_t dim_K2 = N;
	  bool transpose_B2 = (g_dims.first != dim_K2);
	  int64_t dim_J2 = transpose_B2 ? g_dims.first : g_dims.second;
	  int64_t stride_A2 = dim_K2;
	  int64_t stride_B2 = transpose_B2 ? dim_K2 : dim_J2;

	  std::string logits = out + "_logits";
	  std::string attnbuf = out + "_attn";
	  std::string ident = out + "_identity";

	  // The N x N identity matrix is a fixed 0/1 pattern known entirely at codegen time, so
	  // emit it as a compile-time literal initializer instead of a runtime double loop --
	  // matches the convention elsewhere in this file of emitting constant data as
	  // `static const <type> <name>[] = {...}` rather than computing it on-device.
	  std::ostringstream ident_init;
	  for (int64_t i = 0; i < N; ++i) {
		  for (int64_t j = 0; j < N; ++j) {
			  if (i != 0 || j != 0) ident_init << ",";
			  ident_init << (i == j ? "1" : "0");
		  }
	  }

	  std::ostringstream ss;
	  ss << "static elem_t " << logits << "[" << (dim_I0 * dim_J0) << "];\n  "
		 << "static elem_t " << attnbuf << "[" << (dim_I0 * N) << "];\n  "
		 << "static const elem_t " << ident << "[" << (N * N) << "] = {" << ident_init.str() << "};\n  ";

	  ss << "unsigned long long start = read_cycles();\n"
	     << "tiled_matmul_auto("
		 << dim_I0 << ", " << dim_J0 << ", " << dim_K0 << ", "
		 << "(elem_t*)GEMMINI_DATA(" << func_args[theta_idx] << "), "
		 << "(elem_t*)GEMMINI_DATA(" << func_args[phi_idx] << "), "
		 << "NULL, "
		 << "(elem_t*)" << logits << ", "
		 << stride_A0 << ", " << stride_B0 << ", " << dim_J0 << ", " << dim_J0 << ", "
		 << "MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, "
		 << "SOFTMAX, " << acc_scale0 << ", " << bert_scale << ", "
		 << "false, false, " << transpose_B0 << ", "
		 << "false, false, 0, WS);\n  ";

	  //ss << EmitDebugDumpRaw(logits, func_name + ".logits", dim_I0 * dim_J0);

	  ss << "tiled_matmul_auto("
		 << dim_I2 << ", " << dim_J2 << ", " << dim_K2 << ", "
		 << "(elem_t*)" << logits << ", "
		 << "(elem_t*)GEMMINI_DATA(" << func_args[g_idx] << "), "
		 << "NULL, "
		 << "(elem_t*)GEMMINI_DATA(" << out << "), "
		 << stride_A2 << ", " << stride_B2 << ", " << dim_J2 << ", " << dim_J2 << ", "
		 << "MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, "
		 << "NO_ACTIVATION, " << acc_scale1 << ", 0, "
		 << "false, false, " << transpose_B2 << ", "
		 << "false, false, 0, WS);"
	     << "unsigned long long end = read_cycles();\n"
		 << "gemmini_clock_cycles[gemmini_clock_cycle_idx++] = end - start;\n";

	  //int64_t numel = dim_I2 * dim_J2;
	  //ss << EmitDebugDump(out, func_name, numel);

	  return ss.str();
  }

  std::string EmitGemminiResadd(const CallNode* call, const Function& func, const std::string& out, const std::string func_name) {
      ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t i0_idx = arg_idx["input0"]->value;
	  int64_t i1_idx = arg_idx["input1"]->value;

	  const auto* i0_sinfo = GetStructInfo(func->params[i0_idx]).as<TensorStructInfoNode>();
	  auto i0_shape = backend::GetIntShape(i0_sinfo->shape.value().as<ShapeExprNode>()->values);

	  const auto* i1_sinfo = GetStructInfo(func->params[i1_idx]).as<TensorStructInfoNode>();
	  auto i1_shape = backend::GetIntShape(i1_sinfo->shape.value().as<ShapeExprNode>()->values);

	  int64_t I = i0_shape[0] * i0_shape[2] * i0_shape[3];
	  int64_t J = i0_shape[1];
	  //int64_t A_scale_idx = arg_idx["scale_in"]->value;
	  //int64_t B_scale_idx = arg_idx["scale_w"]->value;
	  //int64_t C_scale_idx = arg_idx["scale_out"]->value;

	  double A_scale = attrs.at("scale_in").try_cast<double>().value();
	  double B_scale = attrs.at("scale_w").try_cast<double>().value();
	  double C_scale = attrs.at("scale_out").try_cast<double>().value();
	  bool relu = attrs.at("relu").try_cast<bool>().value();

	  std::ostringstream ss;

//	_STATIC void tiled_resadd_auto(const size_t I, const size_t J,
//	                               const scale_t A_scale, const scale_t B_scale,
//	                               const acc_scale_t C_scale, const elem_t *A,
//	                               const elem_t *B, elem_t *C, bool relu,
//	                               enum tiled_matmul_type_t matadd_type) {
	  ss << "unsigned long long start = read_cycles();\n"
	     << "tiled_resadd_auto("
		 << I << ", "
		 << J << ", "
		 << A_scale << ", "
		 << B_scale << ", "
		 << C_scale << ", "
		 //<< "((scale_t*)GEMMINI_DATA(" << func_args[A_scale_idx] << "))[0], "
		 //<< "((scale_t*)GEMMINI_DATA(" << func_args[B_scale_idx] << "))[0], "
		 //<< "(1.0f / ((acc_scale_t*)GEMMINI_DATA(" << func_args[C_scale_idx] << "))[0]), "
	  	 << "(elem_t*)GEMMINI_DATA(" << func_args[i0_idx] << "), "
	  	 << "(elem_t*)GEMMINI_DATA(" << func_args[i1_idx] << "), "
		 << "(elem_t*)GEMMINI_DATA(" << out << "), "
		 << relu << ", "
		 << "WS);\n"
	     << "unsigned long long end = read_cycles();\n"
		 << "gemmini_clock_cycles[gemmini_clock_cycle_idx++] = end - start;\n";

	  //int64_t numel = I * J;
	  //ss << EmitDebugDump(out, func_name, numel);

	  return ss.str();
  }

  /*! \brief The id of the external gemmini ext_func. */
  std::string ext_func_id_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls Gemmini kernels. */
  ffi::Array<Var> ext_func_args_;
  /*! \brief The statements of the function that will be compiled using Gemmini kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The binding to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
  /*! \brief Map from Constant node to its string name (from RunCodegen). */
  ffi::Map<Constant, ffi::String> constant_names_;
  /*!
   * \brief A mapping from a variable to its unique name.
   * We use this since sometimes different parameters to the same function end up having the same
   * name_hint.
   */
  std::unordered_map<const VarNode*, std::string> var_name_map_;
  /*! \brief A name supply to generate a unique name for each parameter. */
  NameSupply name_sup_;
};

class GemminiModuleCodegen {
 public:
  explicit GemminiModuleCodegen(ffi::Map<Constant, ffi::String> constant_names)
      : constant_names_(std::move(constant_names)) {}

  ffi::Module CreateCSourceModule(ffi::Array<Function> functions,
                                  const ffi::Map<ffi::String, ffi::Any>& options) {
    std::string code = "";
    for (const auto& f : functions) {
      code += "\n" + GenGemminiFunc(f);
    }
    return Finalize(code, func_names_);
  }

 private:
  std::string GenGemminiFunc(const Function& function) {
    TVM_FFI_ICHECK(function.defined()) << "Input error: expect a Relax function.";

    auto sid = GetExtSymbol(function);
    func_names_.push_back(sid);

    CodegenGemmini builder(sid, AnalyzeVar2Value(function), constant_names_);

    for (const auto& p : function->params) {
      builder.AddParm(p);
    }

    auto out = builder.VisitExpr(function->body);
    return builder.JIT(out);
  }

  /*! \brief The accumulated function names. */
  ffi::Array<ffi::String> func_names_;
  /*! \brief Map from Constant node to its string name. */
  ffi::Map<Constant, ffi::String> constant_names_;
};

ffi::Array<ffi::Module> GemminiCompiler(ffi::Array<Function> functions,
                                        ffi::Map<ffi::String, ffi::Any> options,
                                        ffi::Map<Constant, ffi::String> constant_names) {
  auto source_mod = GemminiModuleCodegen(constant_names).CreateCSourceModule(functions, options);
  const auto pf = tvm::ffi::Function::GetGlobal("contrib.gemmini.compile");
  TVM_FFI_ICHECK(pf.has_value())
      << "The packed function contrib.gemmini.compile not found, please import "
         "tvm.contrib.gemmini.build";
  ffi::Module gemmini_mod = (*pf)(source_mod, options).cast<ffi::Module>();

  return {gemmini_mod};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.gemmini", GemminiCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm

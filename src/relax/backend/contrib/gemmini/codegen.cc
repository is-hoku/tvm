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

ffi::Module Finalize(const std::string& code, const ffi::Array<ffi::String>& func_names) {
  TVM_FFI_ICHECK(!func_names.empty())
      << "Should only create Gemmini CSourceModule if there is at least one Gemmini partition";

  std::ostringstream default_headers;
  //default_headers << "#include <tvm/ffi/function.h>\n";
  //default_headers << "#include <dlpack/dlpack.h>\n";
  default_headers << "#include <assert.h>\n";
  default_headers << "#include <stddef.h>\n";
  default_headers << "#include <stdint.h>\n";
  default_headers << "#include <stdio.h>\n";
  default_headers << "#include <stdlib.h>\n";
  default_headers << "#ifndef BAREMETAL\n";
  default_headers << "#include <sys/mman.h>\n";
  default_headers << "#endif\n";
  default_headers << "#include \"include/gemmini_params.h\"\n";
  default_headers << "#include \"include/gemmini.h\"\n";

  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.CSourceModuleCreate");
  VLOG(1) << "Generated Gemmini code:" << std::endl << code;
  return pf(default_headers.str() + code, "c", func_names,
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
    code_stream_ << "void " << ext_func_id_ << "_(";

    for (const auto& arg : ext_func_args_) {
      const auto& dtype_str = GetDtypeString(arg);
      code_stream_ << dtype_str << "* " << arg->name_hint() << ", ";
    }
    for (size_t i = 0; i < out.size() - 1; ++i) {
      code_stream_ << out[i].dtype << "* out" << i << ", ";
    }
    code_stream_ << out.back().dtype << "* out" << out.size() - 1 << ") {\n";
    this->EnterScope();

    // Function body
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
			   f == "gemmini.matmul_softmax" ||
			   f == "gemmini.matmul_transpose" ||
			   f == "gemmini.fc" ||
			   f == "gemmini.fc_leakyrelu") {
		ret.decl = EmitGemminiMatmul(call, func, out_name, f);
	} else if (f == "gemmini.resadd_leakyrelu") {
		ret.decl = EmitGemminiResadd(call, func, out_name, f);
	} else {
		TVM_FFI_THROW(InternalError) << "Unsupported Gemmini op: " << f;
	}

    return ret;
  }

  std::string EmitGemminiConv2d(const CallNode* call, const Function& func, const std::string& out, const std::string& func_name) {
      ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t i_idx = arg_idx["input"]->value;
	  int64_t w_idx = arg_idx["weight"]->value;
	  int64_t b_idx = arg_idx["bias"]->value;

	  // NCHW
	  const auto* in_sinfo = GetStructInfo(func->params[i_idx]).as<TensorStructInfoNode>();
	  auto in_shape = backend::GetIntShape(in_sinfo->shape.value().as<ShapeExprNode>()->values);

	  // OHWI
	  const auto* w_sinfo = GetStructInfo(func->params[w_idx]).as<TensorStructInfoNode>();
	  auto w_shape = backend::GetIntShape(w_sinfo->shape.value().as<ShapeExprNode>()->values);

	  // NCHW
	  //auto* out_sinfo = GetStructInfo(ffi::GetRef<Call>(call)).as<TensorStructInfoNode>();
	  //auto out_shape = backend::GetIntShape(out_sinfo->shape.value().as<ShapeExprNode>()->values);

	  int64_t batch_size = in_shape[0];
	  int64_t in_rows = in_shape[2];
	  int64_t in_cols = in_shape[3];
	  int64_t in_channels = in_shape[1];
	  int64_t out_channels = w_shape[0];

	  std::string act = attrs.at("act").try_cast<std::string>().value();
	  double acc_scale = attrs.at("acc_scale").try_cast<double>().value();
	  int stride = attrs.at("stride").try_cast<int>().value();
	  int padding = attrs.at("padding").try_cast<int>().value();
	  int kernel_dim = attrs.at("kernel_dim").try_cast<int>().value();

	  int64_t out_rows = (in_rows + 2 * padding - kernel_dim) / stride + 1;
	  int64_t out_cols = (in_cols + 2 * padding - kernel_dim) / stride + 1;

	  std::ostringstream ss;

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
	  ss << "tiled_conv_auto("
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
	  	 << "(elem_t*)" << func_args[i_idx] << ", "
	  	 << "(elem_t*)" << func_args[w_idx] << ", "
	  	 << "(acc_t*)" << func_args[b_idx] << ", "
	  	 << "(elem_t*)" << out << ", "
		 << act << ", "
		 << acc_scale << ", "
		 << "0, 0, 0, "
		 << "WS);";

	  return ss.str();
  }

  std::string EmitGemminiMatmul(const CallNode* call, const Function& func, const std::string& out, const std::string func_name) {
      ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t in_idx = arg_idx["input"]->value;
	  int64_t w_idx = arg_idx["weight"]->value;

	  // NCHW
	  const auto* in_sinfo = GetStructInfo(func->params[in_idx]).as<TensorStructInfoNode>();
	  auto in_shape = backend::GetIntShape(in_sinfo->shape.value().as<ShapeExprNode>()->values);

	  // NCHW
	  const auto* w_sinfo = GetStructInfo(func->params[w_idx]).as<TensorStructInfoNode>();
	  auto w_shape = backend::GetIntShape(w_sinfo->shape.value().as<ShapeExprNode>()->values);

	  std::string bias = "NULL";
	  if (arg_idx.count("bias")) {
		  bias = func_args[arg_idx["bias"]->value];
	  }

	  // NCHW
	  //auto* out_sinfo = GetStructInfo(ffi::GetRef<Call>(call)).as<TensorStructInfoNode>();
	  //auto out_shape = backend::GetIntShape(out_sinfo->shape.value().as<ShapeExprNode>()->values);

	  // TODO: handling transpose tensors
	  int64_t dim_I = in_shape[in_shape.size() - 2]; // 2
	  int64_t dim_J = w_shape[w_shape.size() - 1]; // 3
	  int64_t dim_K = in_shape.back();
	  bool transpose_B = (in_shape[in_shape.size() - 1] != w_shape[w_shape.size() - 2] ? true : false);
	  int64_t stride_A = dim_K;
	  int64_t stride_B = in_shape.back();
	  int64_t stride_C = dim_J;
	  int64_t stride_D = dim_J;

	  std::string act = attrs.at("act").try_cast<std::string>().value();
	  double acc_scale = attrs.at("acc_scale").try_cast<double>().value();

	  std::ostringstream ss;

	//tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K, const elem_t *A,
	//                  const elem_t *B, const void *D, void *C, size_t stride_A,
	//                  size_t stride_B, size_t stride_D, size_t stride_C,
	//                  scale_t A_scale_factor, scale_t B_scale_factor,
	//                  scale_acc_t D_scale_factor, int act, acc_scale_t scale,
	//                  acc_scale_t bert_scale, bool repeating_bias, bool transpose_A,
	//                  bool transpose_B, bool full_C, bool low_D, uint8_t weightA,
	//                  enum tiled_matmul_type_t tiled_matmul_type) {
	  ss << "tiled_matmul_auto("
		 << dim_I << ", "
		 << dim_J << ", "
		 << dim_K << ", "
		 << "(elem_t*)" << func_args[in_idx] << ", "
		 << "(elem_t*)" << func_args[w_idx] << ", "
		 << bias << ", "
		 << "(elem_t*)" << out << ", "
		 << stride_A << ", " // stride_A
		 << stride_B << ", " // stride_B
		 << stride_D << ", " // stride_D
		 << stride_C << ", " // stride_C
		 << "MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, "
		 << act << ", "
		 << acc_scale << ", " // scale
		 << "0, " // bert_scale
		 << "false, false, "
		 << transpose_B << ", "
		 << "false, false, 0, "
		 << "WS);";

	  return ss.str();
	  return "tiled_matmul_auto()";
  }

  std::string EmitGemminiResadd(const CallNode* call, const Function& func, const std::string& out, const std::string func_name) {
      ffi::Array<ffi::String> func_args = GetArgumentNames(call);
	  ffi::Map<ffi::String, ffi::Any> attrs = func->attrs->dict;

	  auto arg_idx = backend::ExtractArgIdx(func_name, func);

	  int64_t i0_idx = arg_idx["input0"]->value;
	  int64_t i1_idx = arg_idx["input1"]->value;

	  // NCHW
	  const auto* i0_sinfo = GetStructInfo(func->params[i0_idx]).as<TensorStructInfoNode>();
	  auto i0_shape = backend::GetIntShape(i0_sinfo->shape.value().as<ShapeExprNode>()->values);

	  // NCHW
	  const auto* i1_sinfo = GetStructInfo(func->params[i1_idx]).as<TensorStructInfoNode>();
	  auto i1_shape = backend::GetIntShape(i1_sinfo->shape.value().as<ShapeExprNode>()->values);

	  int64_t I = i0_shape[2];
	  int64_t J = i0_shape[3];
	  std::string A_scale = func_args[arg_idx["scale_in"]->value];
	  std::string B_scale = func_args[arg_idx["scale_w"]->value];
	  std::string C_scale = func_args[arg_idx["scale_out"]->value];

	  bool relu = attrs.at("relu").try_cast<bool>().value();

	  std::ostringstream ss;

//	_STATIC void tiled_resadd_auto(const size_t I, const size_t J,
//	                               const scale_t A_scale, const scale_t B_scale,
//	                               const acc_scale_t C_scale, const elem_t *A,
//	                               const elem_t *B, elem_t *C, bool relu,
//	                               enum tiled_matmul_type_t matadd_type) {
	  ss << "tiled_resadd_auto("
		 << I << ", "
		 << J << ", "
		 << "*" << A_scale << ", "
		 << "*" << B_scale << ", "
		 << "*" << C_scale << ", "
	  	 << "(elem_t*)" << func_args[i0_idx] << ", "
	  	 << "(elem_t*)" << func_args[i1_idx] << ", "
		 << "(elem_t*)" << out << ", "
		 << relu << ", "
		 << "WS);";

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

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

#include <memory>
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
  CodegenGemmini(const std::string& id, const ffi::Map<Var, Expr>& bindings)
      : ext_func_id_(id), bindings_(bindings) {}

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
    auto ret = GenerateBody(call, pattern_name_opt.value());
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

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name) {
    auto func_args = GetArgumentNames(call);
    auto struct_info = GetStructInfo(ffi::GetRef<Call>(call));

	const auto* out_sinfo = struct_info.as<TensorStructInfoNode>();

    const std::string out_name = "out" + std::to_string(buf_idx_++);
    Output output;
    output.name = out_name;
    output.dtype = GetDtypeString(out_sinfo);
    output.need_copy = false;
    GenerateBodyOutput ret;
    ret.outputs.push_back(output);

	std::cout << func_name;
	if (func_name == "gemmini.matmul") {
		ret.decl = EmitGemminiMatmul(call, func_args, out_name);
	} else if (func_name == "gemmini.conv2d") {
		ret.decl = EmitGemminiConv2d(call, func_args, out_name);
	} else {
		TVM_FFI_THROW(InternalError) << "Unsupported Gemmini op: " << func_name;
	}

    return ret;
  }

  std::string EmitGemminiMatmul(const CallNode* call, const ffi::Array<ffi::String>& args, const std::string& out) {
	  return "tiled_matmul_auto()";
  }

  std::string EmitGemminiConv2d(const CallNode* call, const ffi::Array<ffi::String>& args, const std::string& out) {
	  return "tiled_conv2d_auto()";
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
  /*! \brief Required header-file names.
  ffi::Array<ffi::String> headers_; */
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

    CodegenGemmini builder(sid, AnalyzeVar2Value(function));

    for (const auto& p : function->params) {
      builder.AddParm(p);
    }

    auto out = builder.VisitExpr(function->body);
    return builder.JIT(out);
  }

  /*! \brief The accumulated function names. */
  ffi::Array<ffi::String> func_names_;
};

ffi::Array<ffi::Module> GemminiCompiler(ffi::Array<Function> functions,
                                        ffi::Map<ffi::String, ffi::Any> options,
                                        ffi::Map<Constant, ffi::String> /*unused*/) {
  auto source_mod = GemminiModuleCodegen().CreateCSourceModule(functions, options);
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

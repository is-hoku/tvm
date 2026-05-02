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
 * \brief Implementation of the Gemmini C code generator for Relax.
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

ffi::Module Finalize(const std::string& code, const ffi::Array<ffi::String>& func_names) {
  TVM_FFI_ICHECK(!func_names.empty())
      << "Should only create Gemmini CSourceModule if there is at least one Gemmini partition";

  std::ostringstream default_headers;
  default_headers << "#include <tvm/ffi/function.h>\n";
  default_headers << "#include <dlpack/dlpack.h>\n";
  default_headers << "#include <gemmini.h>\n";

  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.CSourceModuleCreate");
  VLOG(1) << "Generated Gemmini code:" << std::endl << code;
  return pf(default_headers.str() + code, "c", func_names,
            /*const_vars=*/ffi::Array<ffi::String>())
      .cast<ffi::Module>();
}

GenerateBodyOutput GenerateBody(const std::string& func_name, const std::string& ext_func_id,
                                const std::vector<std::string>& output_types,
                                const ffi::Array<ffi::String>& func_args,
                                const ffi::Map<ffi::String, ffi::Any>& attrs, int* buf_idx) {
  TVM_FFI_ICHECK_GT(func_args.size(), 0);
  std::ostringstream decl_stream;
  decl_stream << "(" << func_args[0];
  for (size_t i = 1; i < func_args.size(); ++i) {
    decl_stream << ", " << func_args[i];
  }

  GenerateBodyOutput ret;
  for (const auto& out_type : output_types) {
    const std::string out = "out" + std::to_string(*buf_idx++);
    decl_stream << ", " << out;
    Output output;
    output.name = out;
    output.dtype = out_type;
    output.need_copy = false;
    ret.outputs.push_back(output);
  }
  decl_stream << ");";

  // TODO: implement Gemmini C code generation per op (func_name / attrs を見て分岐)
  //   if (func_name == "gemmini.matmul") { ... tiled_matmul_auto(...) ... }
  //   if (func_name == "gemmini.conv2d") { ... tiled_conv_auto(...) ... }
  if (func_name == "gemmini.matmul") {
    ret.decl << "tiled_matmul_auto()"
  }
  if (func_name == "gemmini.conv2d") {
    ret.decl << "tiled_conv_auto()"
  }
  ret.headers = {};

  return ret;
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
    std::vector<std::string> arg_types, arg_names;

    for (const auto& arg : ext_func_args_) {
      auto sinfo = GetStructInfo(arg);
      if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
        arg_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
      } else if (const auto* shape_sinfo = sinfo.as<ShapeStructInfoNode>()) {
        arg_types.emplace_back(backend::DType2String(shape_sinfo->values.value()[0]->dtype));
      } else {
        TVM_FFI_THROW(InternalError) << "Unimplemented";
      }
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

    this->GenerateBackendCFunc(ext_func_id_, arg_types, /*const_arr_name=*/"", out, true);
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
    auto ret = GenerateBody(call, pattern_name_opt.value(), func->attrs->dict);
    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

  OutputType VisitExpr_(const FunctionNode* fn) final {
    TVM_FFI_ICHECK(fn->GetAttr<ffi::String>(attr::kComposite).has_value())
        << "JSON runtime only supports composite functions";
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

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name,
                                  const ffi::Map<ffi::String, ffi::Any>& attrs) {
    auto func_args = GetArgumentNames(call);
    auto struct_info = GetStructInfo(ffi::GetRef<Call>(call));

    std::vector<std::string> out_types;
    if (const auto* tensor_sinfo = struct_info.as<TensorStructInfoNode>()) {
      out_types.emplace_back(backend::DType2String(tensor_sinfo->dtype));
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented sinfo type: " << struct_info;
    }

    return contrib::GenerateBody(func_name, ext_func_id_, out_types, func_args, attrs, &buf_idx_);
  }

  /*! \brief The id of the external Gemmini function. */
  std::string ext_func_id_;
  /*! \brief The index to track the output buffer. */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls Gemmini kernels. */
  ffi::Array<Var> ext_func_args_;
  /*! \brief The statements of the function that will be compiled using Gemmini. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The binding to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
  /*!
   * \brief A mapping from a variable to its unique name.
   */
  std::unordered_map<const VarNode*, std::string> var_name_map_;
  /*! \brief A name supply to generate a unique name for each parameter. */
  NameSupply name_sup_;
};

class GemminiModuleCodegen {
 public:
  ffi::Module CreateCSourceModule(ffi::Array<Function> functions) {
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
  auto source_mod = GemminiModuleCodegen().CreateCSourceModule(functions);

  // TODO: RISC-V cross-compilation
  // const auto pf = tvm::ffi::Function::GetGlobal("contrib.gemmini.compile");
  // if (pf.has_value()) {
  //   return {(*pf)(source_mod, options).cast<ffi::Module>()};
  // }

  return {source_mod};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.gemmini", GemminiCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm

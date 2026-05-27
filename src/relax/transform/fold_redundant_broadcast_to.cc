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
 * \file src/relax/transform/fold_broadcast.cc
 * \brief Remove broadcast_to(x, shape) when x already has that shape.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class BroadcastFolder : public ExprMutator {
 public:
  static Function Fold(Function func, IRModule ctx_module) {
    BroadcastFolder folder(std::move(ctx_module));
    return Downcast<Function>(folder(func));
  }

 private:
  explicit BroadcastFolder(IRModule ctx_module) : ExprMutator(ctx_module) {}

  Expr VisitExpr_(const CallNode* call) final {
    static const Op& broadcast_to_op = Op::Get("relax.broadcast_to");

    // Recursively transform children first (same pattern as FoldConstant)
    Expr new_expr = ExprMutator::VisitExpr_(call);
    const auto* new_call = new_expr.as<CallNode>();
    if (!new_call) return new_expr;

    if (!new_call->op.same_as(broadcast_to_op)) return new_expr;

    // broadcast_to(src, target_shape)
    Expr src = new_call->args[0];
    Expr target_shape = new_call->args[1];  // ShapeExpr passed as second arg

    const auto* src_sinfo = src->struct_info_.as<TensorStructInfoNode>();
    if (!src_sinfo || !src_sinfo->shape.defined()) return new_expr;

    // Remove if src.shape == target_shape (structural equality)
    ffi::StructuralEqual struct_equal;
    if (struct_equal(src_sinfo->shape.value(), target_shape)) {
      return src;
    }

    return new_expr;
  }
};

namespace transform {

Pass FoldRedundantBroadcastTo() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return BroadcastFolder::Fold(f, m);
  };
  return CreateFunctionPass(pass_func, 0, "FoldRedundantBroadcastTo", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FoldRedundantBroadcastTo", FoldRedundantBroadcastTo);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm

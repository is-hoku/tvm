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

#include <cstring>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include "tvm/relax/attrs/manipulate.h"
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace relax {

class PermuteDimsFolder : public ExprMutator {
 public:
  static Function Fold(Function func, IRModule ctx_module) {
    PermuteDimsFolder folder(std::move(ctx_module));
    return Downcast<Function>(folder(func));
  }

 private:
  explicit PermuteDimsFolder(IRModule ctx_module) : ExprMutator(ctx_module) {}

  Expr VisitExpr_(const CallNode* call) final {
    // post-order mutation
    static const Op& permute_dims_op = Op::Get("relax.permute_dims");
    static const Op& dequantize_op = Op::Get("relax.dequantize");

	Expr new_expr = ExprMutator::VisitExpr_(call);
	const auto* new_call = new_expr.as<CallNode>();
	if (!new_call || !new_call->op.same_as(permute_dims_op)) {
		return new_expr;
	}

	// lv: R.dequantize(weight, ...)
	// permute_dims(lv, axes=[0, 2, 3, 1])
	if (!new_call->args[0]->IsInstance<VarNode>()) {
		return new_expr;
	}
	ffi::Optional<Expr> val = LookupBinding(Downcast<Var>(new_call->args[0]));
	if (!val) {
		return new_expr;
	}
	const auto* dq = val.value().as<CallNode>();
	if (!dq || !dq->op.same_as(dequantize_op)) {
		return new_expr;
	}
	const auto* weight = dq->args[0].as<ConstantNode>();
	if (!weight) {
		return new_expr;
	}
	if (weight->data->dtype.code != kDLInt || weight->data->dtype.bits != 8){
		return new_expr;
	}
	if (weight->data->ndim != 4) {
		return new_expr;
	}
	const auto* perm_attrs = new_call->attrs.as<PermuteDimsAttrs>();
	if (!perm_attrs || !perm_attrs->axes.defined()) {
		return new_expr;
	}

	// Transpose tensor
	const runtime::Tensor& src = weight->data;
	const ffi::Array<Integer> axes = perm_attrs->axes.value();
	int ndim = static_cast<int>(axes.size());
	std::vector<int64_t> new_shape(ndim);
	std::vector<int> ax(ndim);
	for (int i = 0; i < ndim; ++i) {
		ax[i] = static_cast<int>(axes[i]->value);
		new_shape[i] = src->shape[ax[i]];
	}
	DLDevice cpu_dev{kDLCPU, 0};
	runtime::Tensor dst = runtime::Tensor::Empty(ffi::Shape(new_shape), src->dtype, cpu_dev);
	std::vector<int64_t> src_stride(ndim);
	src_stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; --i) {
		src_stride[i] = src_stride[i + 1] * src->shape[i + 1];
	}
	int64_t total = 1;
	for (int i = 0; i < ndim; ++i) {
		total *= src->shape[i];
	}
	int elem_bytes = (src->dtype.bits + 7) / 8;
	const uint8_t* sp = static_cast<const uint8_t*>(src->data);
	uint8_t* dp = static_cast<uint8_t*>(dst->data);
	for (int64_t flat = 0; flat < total; ++flat) {
		int64_t src_off = 0, tmp = flat;
		for (int i = ndim - 1; i >= 0; --i) {
			int64_t idx = tmp % new_shape[i];
			tmp /= new_shape[i];
			src_off += idx * src_stride[ax[i]];
		}
		std::memcpy(dp + flat * elem_bytes, sp + src_off * elem_bytes, elem_bytes);
	}

	return Call(dequantize_op, {Constant(dst), dq->args[1], dq->args[2]}, dq->attrs);
  }

  Expr VisitExpr_(const VarNode* op) final {
    ffi::Optional<Expr> opt = LookupBinding(ffi::GetRef<Var>(op));
    // `as` check checks if opt is not null and is instance of constant
    if (opt.as<relax::ConstantNode>()) {
      return opt.value();
    }
    return ExprMutator::VisitExpr_(op);
  }

  // cache for function build, via structural equality
  std::unordered_map<tirx::PrimFunc, ffi::Optional<ffi::Function>, ffi::StructuralHash,
                     ffi::StructuralEqual>
      func_build_cache_;
};

namespace transform {

Pass FoldPermuteDims() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return PermuteDimsFolder::Fold(f, m);
  };
  return CreateFunctionPass(pass_func, 0, "FoldPermuteDims", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FoldPermuteDims", FoldPermuteDims);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm

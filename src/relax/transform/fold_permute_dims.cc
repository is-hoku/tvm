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
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include "tvm/relax/attrs/manipulate.h"
#include <tvm/runtime/tensor.h>

#include "../backend/contrib/utils.h"

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
    static const Op& matmul_op = Op::Get("relax.matmul");

	Expr new_expr = ExprMutator::VisitExpr_(call);
	const auto* new_call = new_expr.as<CallNode>();
	if (!new_call) {
		return new_expr;
	}
	if (new_call->op.same_as(permute_dims_op)) {
		return FoldConvWeightPermute(new_call, new_expr);
	}
	if (new_call->op.same_as(matmul_op)) {
		return FixFCFlattenLayout(new_call, new_expr);
	}
	return new_expr;
  }

  // Physically bakes an OIHW->HWIO (or similar) weight transpose into the constant's data,
  // for conv2d weights: ConvertLayout inserts a real `relax.permute_dims` on the
  // dequantized weight constant to request the new layout; this replaces
  // `permute_dims(dequantize(Constant, ...))` with `dequantize(Constant(pre-transposed), ...)`
  // so no runtime transpose op remains.
  Expr FoldConvWeightPermute(const CallNode* new_call, Expr new_expr) {
    static const Op& dequantize_op = Op::Get("relax.dequantize");

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

  // Given a Var, repeatedly unwraps unary passthrough ops (dequantize/quantize/leakyrelu)
  // that never reorder elements, to find the nearest "structural" producer -- e.g. the
  // `relax.reshape` that flattens a conv's spatial output before it reaches a matmul/FC.
  const CallNode* SkipUnaryPassthrough(Expr e) {
    static const Op& dequantize_op = Op::Get("relax.dequantize");
    static const Op& quantize_op = Op::Get("relax.quantize");
    static const Op& leakyrelu_op = Op::Get("relax.nn.leakyrelu");
    for (int guard = 0; guard < 16; ++guard) {
      const auto* var = e.as<VarNode>();
      if (!var) return nullptr;
      ffi::Optional<Expr> bound = LookupBinding(ffi::GetRef<Var>(var));
      if (!bound) return nullptr;
      const auto* bcall = bound.value().as<CallNode>();
      if (!bcall) return nullptr;
      if (bcall->op.same_as(dequantize_op) || bcall->op.same_as(quantize_op) ||
          bcall->op.same_as(leakyrelu_op)) {
        e = bcall->args[0];
        continue;
      }
      return bcall;
    }
    return nullptr;
  }

  // Fixes a layout mismatch at the CNN->FC boundary. After `ConvertLayout` moves all
  // `relax.nn.conv2d` ops to NHWC/HWIO, Gemmini's conv2d codegen genuinely produces
  // physically NHWC (H,W,C) output. To keep the rest of the graph in its original NCHW
  // convention, ConvertLayout inserts a `relax.permute_dims(axes=[0,3,1,2])` right after
  // the conv (NHWC->NCHW) -- but that permute is absorbed into the conv2d_leakyrelu
  // composite (its pattern's optional `permute_dims(add)`) and is NEVER physically
  // executed by the Gemmini codegen. So by the time the `relax.reshape` flattens the last
  // conv output before the FC/matmul, the tensor is *declared* NCHW `(N,C,H,W)` (and the
  // flatten's C,H,W order matches the imported PyTorch `nn.Linear.weight`, which
  // `ConvertLayout` never touches), yet the actual bytes in the buffer are still NHWC
  // (H,W,C order). Real hardware therefore multiplies a physically H,W,C-ordered input
  // against a weight whose columns are indexed C,H,W -- a genuine layout mismatch, not a
  // rounding difference (see nchw_nhwc.md "Finding #3" for the real-hardware-log
  // triangulation). This permutes the weight's columns at compile time from the logical
  // C,H,W order to the physical H,W,C order, so the matmul as generated is numerically
  // correct -- mirroring how FoldConvWeightPermute bakes the OIHW->HWIO permutation into
  // conv weights above.
  Expr FixFCFlattenLayout(const CallNode* matmul_call, Expr matmul_expr) {
    static const Op& permute_dims_op = Op::Get("relax.permute_dims");
    static const Op& dequantize_op = Op::Get("relax.dequantize");
    static const Op& reshape_op = Op::Get("relax.reshape");

    if (matmul_call->args.size() < 2) return matmul_expr;

    // LHS: expect (a Var bound to) `dequantize(input, scale, zp)`.
    Expr lhs_data;
    if (const auto* lhs_var = matmul_call->args[0].as<VarNode>()) {
      ffi::Optional<Expr> lhs_bound = LookupBinding(ffi::GetRef<Var>(lhs_var));
      if (lhs_bound) {
        const auto* lhs_dq = lhs_bound.value().as<CallNode>();
        if (lhs_dq && lhs_dq->op.same_as(dequantize_op)) {
          lhs_data = lhs_dq->args[0];
        }
      }
    }
    if (!lhs_data.defined()) return matmul_expr;

    const CallNode* reshape_call = SkipUnaryPassthrough(lhs_data);
    if (!reshape_call || !reshape_call->op.same_as(reshape_op)) return matmul_expr;

    // The reshape's own input must be a 4D tensor collapsing to a flat vector -- i.e. the
    // CNN->FC flatten. It is declared NCHW `(N, C, H, W)` (the NHWC->NCHW back-permute
    // described above has already relabeled it, even though the physical bytes remain
    // NHWC), so C = axis 1, H = axis 2, W = axis 3.
    StructInfo pre_sinfo = GetStructInfo(reshape_call->args[0]);
    const auto* pre_tsinfo = pre_sinfo.as<TensorStructInfoNode>();
    if (!pre_tsinfo || !pre_tsinfo->shape.defined()) return matmul_expr;
    const auto* pre_shape_expr = pre_tsinfo->shape.value().as<ShapeExprNode>();
    if (!pre_shape_expr || pre_shape_expr->values.size() != 4) return matmul_expr;
    std::vector<int64_t> pre_shape = backend::GetIntShape(pre_shape_expr->values);
    int64_t dim_c = pre_shape[1], dim_h = pre_shape[2], dim_w = pre_shape[3];
    if (dim_h <= 0 || dim_w <= 0 || dim_c <= 0) return matmul_expr;

    // RHS: expect (a Var bound to) `permute_dims(dequantize(weight_const, scale, zp))`,
    // the FC weight's ordinary matmul-transpose (untouched by FoldConvWeightPermute above
    // since the weight is 2D, not 4D). In normalized/ANF form the matmul's arg is a Var,
    // so unwrap it via LookupBinding rather than expecting an inline Call.
    Expr rhs_expr = matmul_call->args[1];
    if (const auto* rhs_var = rhs_expr.as<VarNode>()) {
      ffi::Optional<Expr> rhs_bound = LookupBinding(ffi::GetRef<Var>(rhs_var));
      if (!rhs_bound) return matmul_expr;
      rhs_expr = rhs_bound.value();
    }
    const auto* rhs_permute = rhs_expr.as<CallNode>();
    if (!rhs_permute || !rhs_permute->op.same_as(permute_dims_op)) return matmul_expr;
    if (!rhs_permute->args[0]->IsInstance<VarNode>()) return matmul_expr;
    ffi::Optional<Expr> w_val = LookupBinding(Downcast<Var>(rhs_permute->args[0]));
    if (!w_val) return matmul_expr;
    const auto* w_dq = w_val.value().as<CallNode>();
    if (!w_dq || !w_dq->op.same_as(dequantize_op)) return matmul_expr;
    const auto* weight = w_dq->args[0].as<ConstantNode>();
    if (!weight) return matmul_expr;
    if (weight->data->dtype.code != kDLInt || weight->data->dtype.bits != 8) return matmul_expr;
    if (weight->data->ndim != 2) return matmul_expr;

    const runtime::Tensor& src = weight->data;
    int64_t out_features = src->shape[0];
    int64_t in_features = src->shape[1];
    if (in_features != dim_h * dim_w * dim_c) return matmul_expr;

    DLDevice cpu_dev{kDLCPU, 0};
    runtime::Tensor dst =
        runtime::Tensor::Empty(ffi::Shape({out_features, in_features}), src->dtype, cpu_dev);
    int elem_bytes = (src->dtype.bits + 7) / 8;
    const uint8_t* sp = static_cast<const uint8_t*>(src->data);
    uint8_t* dp = static_cast<uint8_t*>(dst->data);
    // dst column = h*(W*C) + w*C + c  (physical H,W,C order)
    // src column = c*(H*W) + h*W + w  (original PyTorch C,H,W-flatten order)
    for (int64_t o = 0; o < out_features; ++o) {
      for (int64_t h = 0; h < dim_h; ++h) {
        for (int64_t w = 0; w < dim_w; ++w) {
          for (int64_t c = 0; c < dim_c; ++c) {
            int64_t dst_col = h * (dim_w * dim_c) + w * dim_c + c;
            int64_t src_col = c * (dim_h * dim_w) + h * dim_w + w;
            std::memcpy(dp + (o * in_features + dst_col) * elem_bytes,
                        sp + (o * in_features + src_col) * elem_bytes, elem_bytes);
          }
        }
      }
    }

    Expr new_dq = Call(dequantize_op, {Constant(dst), w_dq->args[1], w_dq->args[2]}, w_dq->attrs);
    Expr new_rhs = Call(permute_dims_op, {new_dq}, rhs_permute->attrs);
    return Call(matmul_call->op, {matmul_call->args[0], new_rhs}, matmul_call->attrs);
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

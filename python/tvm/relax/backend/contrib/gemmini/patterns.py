# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Gemmini Pattern Table with Architectural Concepts
"""

from typing import ClassVar

from tvm import TVMError
from tvm.ir import Op
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr import Var
from tvm.relax.transform import PatternCheckContext

from ...pattern_registry import register_patterns

from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern

# Gemmini-specific configuration constants
class GemminiConfig:
    SYSTOLIC_SIZE=16


def _check_gemmini_memory_constraints(
    context: PatternCheckContext,  # pylint: disable=unused-argument
) -> bool:
    """
    Placeholder for Gemmini memory hierarchy constraint checking.

    A real implementation would inspect the annotated expression's
    TensorStructInfo to verify the tensor fits within the NPU's
    on-chip SRAM (L1) or compute memory (L2/CMX). Tensors that
    exceed on-chip capacity require tiling before offload.
    """
    return True


def _check_gemmini_quantization(
    context: PatternCheckContext,  # pylint: disable=unused-argument
) -> bool:
    """
    Placeholder for Gemmini quantization requirement checking.

    A real implementation would verify the op's dtype falls within
    the set supported by the Gemmini (e.g. int8, int16, float16, float32)
    and reject ops with unsupported dtypes so they fall back to CPU.
    """
    return True


def conv2d_pattern():
    """
    input -> DQ, weight -> DQ -> conv2d(input, weight), bias -> reshape -> add(conv2d, reshape) -> Q
    """

    def _make_conv2d_pattern():
        x0 = wildcard()
        s0 = wildcard()
        z0 = wildcard()
        d = is_op("relax.dequantize")(x0, s0, z0)
        d_t = is_op("relax.permute_dims")(d) | d

        wi8 = wildcard()
        sw = wildcard()
        zpw = wildcard()
        weight = is_op("relax.dequantize")(wi8, sw, zpw)
        # When putting ConvertLayout before FuseOpsByPattern inserts the relax.permute_dims to transpose weights as NHWC that is required by Gemmini from NCHW that is a representation in PyTorch
        weight_t = is_op("relax.permute_dims")(weight) | weight

        conv = is_op("relax.nn.conv2d")(d_t, weight_t)

        bias = wildcard()
        reshape = is_op("relax.reshape")(bias, wildcard())
        bias_t = is_op("relax.permute_dims")(reshape) | reshape
        add = is_op("relax.add")(conv, bias_t)

        add_t = is_op("relax.permute_dims")(add) | add
        s1 = wildcard()
        z1 = wildcard()
        output = is_op("relax.quantize")(add_t, s1, z1)

        annotations = {
            "input": x0,
            "scale_in": s0,
            "scale_w": sw,
            "scale_out": s1,
            "weight": wi8,
            "conv": conv,
            "bias": bias,
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_in = annotated_expr["scale_in"].data.numpy().item()
        scale_w = annotated_expr["scale_w"].data.numpy().item()
        scale_out = annotated_expr["scale_out"].data.numpy().item()
        conv_attrs = annotated_expr["conv"].attrs
        stride = int(conv_attrs.strides[0])
        padding = int(conv_attrs.padding[0])
        kernel_dim = int(annotated_expr["weight"].struct_info.shape[2])
        return {
            "act": "NO_ACTIVATION",
            "acc_scale":  float(scale_in * scale_w / scale_out),
            "stride":     stride,
            "padding":    padding,
            "kernel_dim": kernel_dim,
        }

    def _check_conv2d_pattern(
        context: PatternCheckContext,  # pylint: disable=unused-argument
    ) -> bool:
        return True

    return ("gemmini.conv2d", *_make_conv2d_pattern(), _check_conv2d_pattern, _attrs_getter)


def matmul_softmax_pattern():
    """
    LHS (theta): int8 -> DQ -> reshape -> Q -> DQ -> permute_dims -> Q -> DQ -> broadcast_to
    RHS (phi):   int8 -> DQ -> reshape -> Q -> DQ -> broadcast_to
    matmul(LHS, RHS) -> softmax -> broadcast_to  (NO QUANTIZE at output)
    """

    def _make_matmul_softmax_pattern():
        x0 = wildcard()
        s0 = wildcard()
        z0 = wildcard()
        d = is_op("relax.dequantize")(x0, s0, z0)
        reshape = is_op("relax.reshape")(d, wildcard())
        s2 = wildcard()
        z2 = wildcard()
        q1 = is_op("relax.quantize")(reshape, s2, z2)
        s3 = wildcard()
        z3 = wildcard()
        d1 = is_op("relax.dequantize")(q1, s3, z3)
        transposed = is_op("relax.permute_dims")(d1)
        s4 = wildcard()
        z4 = wildcard()
        q2 = is_op("relax.quantize")(transposed, s4, z4)
        s5 = wildcard()
        z5 = wildcard()
        d2 = is_op("relax.dequantize")(q2, s5, z5)
        lhs = is_op("relax.broadcast_to")(d2, wildcard()) | d2

        # RHS: phi branch - int8 -> DQ -> reshape -> Q -> DQ
        input_phi = wildcard()
        scale_phi_dq = wildcard()
        zp_phi_dq = wildcard()
        dq_phi = is_op("relax.dequantize")(input_phi, scale_phi_dq, zp_phi_dq)
        reshape_phi = is_op("relax.reshape")(dq_phi, wildcard())
        s6 = wildcard()
        z6 = wildcard()
        q3 = is_op("relax.quantize")(reshape_phi, s6, z6)
        s7 = wildcard()
        z7 = wildcard()
        d3 = is_op("relax.dequantize")(q3, s7, z7)
        rhs = is_op("relax.broadcast_to")(d3, wildcard()) | d3

        matmul = is_op("relax.matmul")(lhs, rhs)
        softmax = is_op("relax.nn.softmax")(matmul)
        output = is_op("relax.broadcast_to")(softmax, wildcard()) | softmax

        annotations = {
            "input":    x0,
            "weight":    input_phi,
            "scale_in": s0,            # theta int8 input DQ scale
            "scale_w": scale_phi_dq,  # phi   int8 input DQ scale
            #"scale_out": 1.0,
            #"root":      output,
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_lhs = annotated_expr["scale_in"].data.numpy().item()
        scale_rhs = annotated_expr["scale_w"].data.numpy().item()
        return {
            "act": "SOFTMAX",
            "acc_scale": float(scale_lhs * scale_rhs / 1.0),
        }

    def _check_matmul_softmax_pattern(context: PatternCheckContext) -> bool:
        return True

    return ("gemmini.matmul_softmax", *_make_matmul_softmax_pattern(), _check_matmul_softmax_pattern, _attrs_getter)


def matmul_transpose_pattern():
    """
    wildcard (e.g. softmax), int8_input -> DQ -> reshape -> Q -> DQ -> permute_dims -> Q -> DQ
    -> matmul(wildcard, DQ) -> permute_dims -> reshape -> Q
    """

    def _make_matmul_transpose_pattern():
        # LHS: any float input (e.g. softmax attention weights)
        input0 = wildcard()
        lhs = is_op("relax.broadcast_to")(input0, wildcard()) | input0

        # RHS: full g-branch chain: int8 -> DQ -> reshape -> Q -> DQ -> permute_dims -> Q -> DQ
        input1 = wildcard()
        scale_g0_dq = wildcard()
        zp_g0_dq = wildcard()
        dq_g0 = is_op("relax.dequantize")(input1, scale_g0_dq, zp_g0_dq)

        reshape_g = is_op("relax.reshape")(dq_g0, wildcard())

        scale_g1_q = wildcard()
        zp_g1_q = wildcard()
        q_g1 = is_op("relax.quantize")(reshape_g, scale_g1_q, zp_g1_q)

        scale_g1_dq = wildcard()
        zp_g1_dq = wildcard()
        dq_g1 = is_op("relax.dequantize")(q_g1, scale_g1_dq, zp_g1_dq)

        permute_g = is_op("relax.permute_dims")(dq_g1)

        scale_g2_q = wildcard()
        zp_g2_q = wildcard()
        q_g2 = is_op("relax.quantize")(permute_g, scale_g2_q, zp_g2_q)

        scale_g2_dq = wildcard()
        zp_g2_dq = wildcard()
        requant_g = is_op("relax.dequantize")(q_g2, scale_g2_dq, zp_g2_dq)
        rhs = is_op("relax.broadcast_to")(requant_g, wildcard()) | requant_g

        matmul = is_op("relax.matmul")(lhs, rhs)
        transposed = is_op("relax.permute_dims")(matmul)
        reshape = is_op("relax.reshape")(transposed, wildcard())
        scale0_q = wildcard()
        zp0_q = wildcard()
        output = is_op("relax.quantize")(reshape, scale0_q, zp0_q)

        annotations = {
            "input":    input0,
            "weight":    input1,
            #"scale_in": 1.0,
            "scale_w": scale_g0_dq,  # g-branch int8 input DQ scale
            "scale_out": scale0_q,     # output Q scale
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_rhs = annotated_expr["scale_w"].data.numpy().item()
        scale_out = annotated_expr["scale_out"].data.numpy().item()
        return {
            "act": "NO_ACTIVATION",
            "acc_scale": float(1.0 * scale_rhs / scale_out),
        }

    def _check_matmul_transpose_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.matmul_transpose", *_make_matmul_transpose_pattern(), _check_matmul_transpose_pattern, _attrs_getter)


def resadd_leakyrelu_pattern():
    """
    int8_input0 -> DQ, int8_input1 -> DQ, resadd(requant0, requant1) -> Q/DQ -> leakyrelu -> Q
    """

    def _make_resadd_leakyrelu_pattern():
        input0 = wildcard()
        scale0_dq = wildcard()
        zp0_dq = wildcard()
        lhs = is_op("relax.dequantize")(input0, scale0_dq, zp0_dq)
        input1 = wildcard()
        scale1_dq = wildcard()
        zp1_dq = wildcard()
        rhs = is_op("relax.dequantize")(input1, scale1_dq, zp1_dq)

        add = is_op("relax.add")(lhs, rhs)
        scale2_q = wildcard()
        zp2_q = wildcard()
        scale2_dq = wildcard()
        zp2_dq = wildcard()
        q2 = is_op("relax.quantize")(add, scale2_q, zp2_q)
        _r = is_op("relax.dequantize")(q2, scale2_dq, zp2_dq)
        act = is_op("relax.nn.leakyrelu")(_r)
        scale0_q = wildcard()
        zp0_q = wildcard()
        output = is_op("relax.quantize")(act, scale0_q, zp0_q)

        annotations = {
            "input0": input0,
            "input1": input1,
            "scale0_dq": scale0_dq,
            "scale2_q": scale2_q,
            "scale1_dq": scale1_dq,
            "scale0_q": scale0_q,
            #"root": output,
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale0_dq = annotated_expr["scale0_dq"].data.numpy().item()
        scale2_q = annotated_expr["scale2_q"].data.numpy().item()
        scale1_dq = annotated_expr["scale1_dq"].data.numpy().item()
        scale0_q = annotated_expr["scale0_q"].data.numpy().item()
        return {
            "scale_in":  scale0_dq / scale2_q,
            "scale_w":   scale1_dq / scale2_q,
            "scale_out": scale2_q / scale0_q,
            "relu": True,
        }

    def _check_resadd_leakyrelu_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.resadd_leakyrelu", *_make_resadd_leakyrelu_pattern(), _check_resadd_leakyrelu_pattern, _attrs_getter)


def conv2d_leakyrelu_pattern():
    """
    input -> DQ, weight -> DQ -> conv2d(dq, weight), bias -> reshape, add(conv2d, reshape) -> Q/DQ -> leakyrelu -> reshape -> Q
    """

    def _make_conv2d_leakyrelu_pattern():
        input = wildcard()
        scale_dq = wildcard()
        zp_dq = wildcard()
        requant = is_op("relax.dequantize")(input, scale_dq, zp_dq)
        requant_t = is_op("relax.permute_dims")(requant) | requant

        wi8 = wildcard()
        sw = wildcard()
        zpw = wildcard()
        weight = is_op("relax.dequantize")(wi8, sw, zpw)
        # When putting ConvertLayout before FuseOpsByPattern inserts the relax.permute_dims to transpose weights as NHWC that is required by Gemmini from NCHW that is a representation in PyTorch
        weight_t = is_op("relax.permute_dims")(weight) | weight

        conv = is_op("relax.nn.conv2d")(requant_t, weight_t)

        bias = wildcard()
        bias_reshape = is_op("relax.reshape")(bias, wildcard())
        bias_t = is_op("relax.permute_dims")(bias_reshape) | bias_reshape
        add = is_op("relax.add")(conv, bias_t)

        add_t = is_op("relax.permute_dims")(add) | add
        scale1_q = wildcard()
        zp1_q = wildcard()
        scale1_dq = wildcard()
        zp1_dq = wildcard()
        q1 = is_op("relax.quantize")(add_t, scale1_q, zp1_q)
        _r = is_op("relax.dequantize")(q1, scale1_dq, zp1_dq)
        act = is_op("relax.nn.leakyrelu")(_r)
        act_reshape = is_op("relax.reshape")(act, wildcard()) | act
        scale_q = wildcard()
        zp_q = wildcard()
        output = is_op("relax.quantize")(act_reshape, scale_q, zp_q)

        annotations = {
            "input":     input,
            "scale_in":  scale_dq,
            "scale_w":   sw,
            "scale_out": scale_q,
            "weight":    wi8,
            "bias":      bias,
            "conv":      conv,
            #"root":      output,
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_in = annotated_expr["scale_in"].data.numpy().item()
        scale_w = annotated_expr["scale_w"].data.numpy().item()
        scale_out = annotated_expr["scale_out"].data.numpy().item()
        conv_attrs = annotated_expr["conv"].attrs
        stride = int(conv_attrs.strides[0])
        padding = int(conv_attrs.padding[0])
        kernel_dim = int(annotated_expr["weight"].struct_info.shape[2])
        return {
            "act": "RELU",
            "acc_scale": float(scale_in * scale_w / scale_out),
            "stride": stride,
            "padding": padding,
            "kernel_dim": kernel_dim,
        }

    def _check_conv2d_leakyrelu_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.conv2d_leakyrelu", *_make_conv2d_leakyrelu_pattern(), _check_conv2d_leakyrelu_pattern, _attrs_getter)


def fc_leakyrelu_pattern():
    """
    input -> DQ, weight -> DQ -> permute_dims, -> matmul(requant, permute_dims), bias -> add(bias, matmul) -> requant -> leakyrelu -> Q
    """

    def _make_fc_leakyrelu_pattern():
        input = wildcard()
        scale_dq = wildcard()
        zp_dq = wildcard()
        lhs = is_op("relax.dequantize")(input, scale_dq, zp_dq)

        wi8 = wildcard()
        sw = wildcard()
        zpw = wildcard()
        weight = is_op("relax.dequantize")(wi8, sw, zpw)
        rhs = is_op("relax.permute_dims")(weight)

        matmul = is_op("relax.matmul")(lhs, rhs)

        bias = wildcard()
        add = is_op("relax.add")(bias, matmul)

        scale1_q = wildcard()
        zp1_q = wildcard()
        scale1_dq = wildcard()
        zp1_dq = wildcard()
        q1 = is_op("relax.quantize")(add, scale1_q, zp1_q)
        _r = is_op("relax.dequantize")(q1, scale1_dq, zp1_dq)
        act = is_op("relax.nn.leakyrelu")(_r)
        scale_q = wildcard()
        zp_q = wildcard()
        output = is_op("relax.quantize")(act, scale_q, zp_q)

        annotations = {
            "input":     input,
            "scale_in":  scale_dq,
            "scale_w":   sw,
            "scale_out": scale_q,
            "weight":    wi8,
            "bias":      bias,
            #"root":      output,
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_in = annotated_expr["scale_in"].data.numpy().item()
        scale_w = annotated_expr["scale_w"].data.numpy().item()
        scale_out = annotated_expr["scale_out"].data.numpy().item()
        return {
            "act": "RELU",
            "acc_scale": float(scale_in * scale_w / scale_out),
        }

    def _check_fc_leakyrelu_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.fc_leakyrelu", *_make_fc_leakyrelu_pattern(), _check_fc_leakyrelu_pattern, _attrs_getter)


def fc_pattern():
    """
    input -> DQ, weight -> DQ -> permute_dims, -> matmul(requant, permute_dims), bias -> add(bias, matmul) -> Q
    """

    def _make_fc_pattern():
        input = wildcard()
        scale_dq = wildcard()
        zp_dq = wildcard()
        lhs = is_op("relax.dequantize")(input, scale_dq, zp_dq)

        wi8 = wildcard()
        sw = wildcard()
        zpw = wildcard()
        weight = is_op("relax.dequantize")(wi8, sw, zpw)
        rhs = is_op("relax.permute_dims")(weight)

        matmul = is_op("relax.matmul")(lhs, rhs)

        bias = wildcard()
        add = is_op("relax.add")(bias, matmul)

        scale_q = wildcard()
        zp_q = wildcard()
        output = is_op("relax.quantize")(add, scale_q, zp_q)

        annotations = {
            "input":     input,
            "scale_in":  scale_dq,
            "scale_w":   sw,
            "scale_out": scale_q,
            "weight":    wi8,
            "bias":      bias,
            #"root":      output,
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_in = annotated_expr["scale_in"].data.numpy().item()
        scale_w = annotated_expr["scale_w"].data.numpy().item()
        scale_out = annotated_expr["scale_out"].data.numpy().item()
        return {
            "act": "NO_ACTIVATION",
            "acc_scale": float(scale_in * scale_w / scale_out),
        }

    def _check_fc_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.fc", *_make_fc_pattern(), _check_fc_pattern, _attrs_getter)


register_patterns(
    [
        conv2d_pattern(), # Fused patterns first (higher priority)
        matmul_softmax_pattern(),
        matmul_transpose_pattern(),
        resadd_leakyrelu_pattern(),
        conv2d_leakyrelu_pattern(),
        fc_pattern(),
        fc_leakyrelu_pattern(),
    ]
)

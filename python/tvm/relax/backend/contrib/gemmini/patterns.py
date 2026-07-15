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
    input -> DQ, weight -> DQ -> conv2d(input, weight), bias -> DQ -> reshape -> add(conv2d, reshape) -> Q
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
        sb = wildcard()
        zb = wildcard()
        bias_d = is_op("relax.dequantize")(bias, sb, zb)
        reshape = is_op("relax.reshape")(bias_d, wildcard())
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
        kernel_dim = int(annotated_expr["weight"].struct_info.shape[0])
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


def matmul_self_attention_pattern():
    """
    LHS (theta): int8 -> DQ -> reshape -> Q -> DQ -> permute_dims -> Q -> DQ -> broadcast_to
    RHS (phi):   int8 -> DQ -> reshape -> Q -> DQ -> broadcast_to
    matmul(LHS, RHS) -> softmax -> broadcast_to  (NO QUANTIZE at output)

    LHS (from phi): wildcard (softmax)
    RHS (g): int8_input -> DQ -> reshape -> Q -> DQ -> permute_dims -> Q -> DQ
    matmul(wildcard, DQ) -> permute_dims -> reshape -> Q
    """

    def _make_matmul_self_attention_pattern():
        # LHS: theta branch
        input_theta = wildcard()
        scale_theta_dq = wildcard()
        zp_theta_dq = wildcard()
        dq_theta = is_op("relax.dequantize")(input_theta, scale_theta_dq, zp_theta_dq)
        reshape_theta = is_op("relax.reshape")(dq_theta, wildcard())
        s2 = wildcard()
        z2 = wildcard()
        q1 = is_op("relax.quantize")(reshape_theta, s2, z2)
        s3 = wildcard()
        z3 = wildcard()
        d1 = is_op("relax.dequantize")(q1, s3, z3)
        t1 = is_op("relax.permute_dims")(d1)
        s4 = wildcard()
        z4 = wildcard()
        q2 = is_op("relax.quantize")(t1, s4, z4)
        s5 = wildcard()
        z5 = wildcard()
        d2 = is_op("relax.dequantize")(q2, s5, z5)
        lhs_theta = is_op("relax.broadcast_to")(d2, wildcard()) | d2

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
        rhs_phi = is_op("relax.broadcast_to")(d3, wildcard()) | d3

        matmul0 = is_op("relax.matmul")(lhs_theta, rhs_phi)
        softmax = is_op("relax.nn.softmax")(matmul0)
        output_phi = is_op("relax.broadcast_to")(softmax, wildcard()) | softmax


        # LHS: softmax attention weights
        lhs_g = is_op("relax.broadcast_to")(output_phi, wildcard()) | output_phi

        # RHS: full g-branch chain: int8 -> DQ -> reshape -> Q -> DQ -> permute_dims -> Q -> DQ
        input_g = wildcard()
        scale_g_dq = wildcard()
        zp_g_dq = wildcard()
        dq_g0 = is_op("relax.dequantize")(input_g, scale_g_dq, zp_g_dq)

        r1 = is_op("relax.reshape")(dq_g0, wildcard())

        s8 = wildcard()
        z8 = wildcard()
        q4 = is_op("relax.quantize")(r1, s8, z8)

        s9 = wildcard()
        z9 = wildcard()
        d4 = is_op("relax.dequantize")(q4, s9, z9)

        t2 = is_op("relax.permute_dims")(d4)

        s9 = wildcard()
        z9 = wildcard()
        q5 = is_op("relax.quantize")(t2, s9, z9)

        s10 = wildcard()
        z10 = wildcard()
        d5 = is_op("relax.dequantize")(q5, s10, z10)
        rhs_g = is_op("relax.broadcast_to")(d5, wildcard()) | d5

        matmul1 = is_op("relax.matmul")(lhs_g, rhs_g)
        t3 = is_op("relax.permute_dims")(matmul1)
        r2 = is_op("relax.reshape")(t3, wildcard())
        s11 = wildcard()
        z11 = wildcard()
        output = is_op("relax.quantize")(r2, s11, z11)

        annotations = {
            "input_theta":    input_theta,
            "input_phi":    input_phi,
            "scale_theta": s5,
            "scale_phi": s7,
            "input_g":    input_g,
            "scale_g": s10,
            "scale_out": s11,     # output Q scale
        }
        return output, annotations

    def _attrs_getter(annotated_expr):
        scale_theta = annotated_expr["scale_theta"].data.numpy().item()
        scale_phi = annotated_expr["scale_phi"].data.numpy().item()
        scale_g = annotated_expr["scale_g"].data.numpy().item()
        scale_out = annotated_expr["scale_out"].data.numpy().item()
        SOFTMAX_OUTPUT_SCALE = 1.0 / 127.0

        return {
            "acc_scale0": 1.0,
            "bert_scale": scale_theta * scale_phi,
            "acc_scale1": float(scale_g * SOFTMAX_OUTPUT_SCALE / scale_out),
        }

    def _check_matmul_self_attention_pattern(context: PatternCheckContext) -> bool:
        return True

    return ("gemmini.matmul_self_attention", *_make_matmul_self_attention_pattern(), _check_matmul_self_attention_pattern, _attrs_getter)


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
    input -> DQ, weight -> DQ -> conv2d(dq, weight), bias -> DQ -> reshape, add(conv2d, reshape) -> Q/DQ -> leakyrelu -> reshape -> Q
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
        sb = wildcard()
        zb = wildcard()
        bias_d = is_op("relax.dequantize")(bias, sb, zb)
        bias_reshape = is_op("relax.reshape")(bias_d, wildcard())
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
        kernel_dim = int(annotated_expr["weight"].struct_info.shape[0])
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
        sb = wildcard()
        zb = wildcard()
        bias_d = is_op("relax.dequantize")(bias, sb, zb)
        add = is_op("relax.add")(bias_d, matmul)

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
        sb = wildcard()
        zb = wildcard()
        bias_d = is_op("relax.dequantize")(bias, sb, zb)
        add = is_op("relax.add")(bias_d, matmul)

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
        matmul_self_attention_pattern(),
        resadd_leakyrelu_pattern(),
        conv2d_leakyrelu_pattern(),
        fc_pattern(),
        fc_leakyrelu_pattern(),
    ]
)

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
from tvm.relax.transform import PatternCheckContext

from ...pattern_registry import register_patterns


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
    input -> DQ, int8_weight -> DQ -> conv2d(input, weight), bias -> reshape -> add(conv2d, reshape) -> Q
    """

    def _make_conv2d_pattern():
        x0 = wildcard()
        s0 = wildcard()
        z0 = wildcard()
        d = is_op("relax.dequantize")(x0, s0, z0)

        weight = wildcard()
        #NOTE: It ignores stride and padding and does not check data layout and out_dtype
        conv = is_op("relax.nn.conv2d")(d, weight)

        bias = wildcard()
        #NOTE: It does not check data layout
        reshape = is_op("relax.reshape")(bias, wildcard())
        add = is_op("relax.add")(conv, reshape)
        s1 = wildcard()
        z1 = wildcard()
        output = is_op("relax.quantize")(add, s1, z1)

        annotations = {
            "input": x0,
            "scale": s0,
            "zp": z0,
            "weight": weight,
            "bias": reshape,
            "root": output,
        }
        return output, annotations

    def _check_conv2d_pattern(
        context: PatternCheckContext,  # pylint: disable=unused-argument
    ) -> bool:
        # TODO: CHECK Q/DQ scaling factors equality
        # NOTE: It does not check output dtype
        #if context.annotated_expr["s0"] == context.annotated_expr["s1"]:
        #    if context.annotated_expr["z0"] == context.annotated_expr["z1"]:
        #        return True
        #return False
        return True

    return ("gemmini.conv2d", *_make_conv2d_pattern(), _check_conv2d_pattern)


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
            "input0": lhs,
            "input1": rhs,
            "matmul": matmul,
            "root": output,
        }
        return output, annotations

    def _check_matmul_softmax_pattern(context: PatternCheckContext) -> bool:
        # NOTE: It does not check output dtype
        #if context.annotated_expr["s0"] == context.annotated_expr["s1"]:
        #    if context.annotated_expr["z0"] == context.annotated_expr["z1"]:
        #        if context.annotated_expr["s2"] == context.annotated_expr["s3"]:
        #            if context.annotated_expr["z2"] == context.annotated_expr["z3"]:
        #                if context.annotated_expr["s4"] == context.annotated_expr["s5"]:
        #                    if context.annotated_expr["z4"] == context.annotated_expr["z5"]:
        #                        if context.annotated_expr["s6"] == context.annotated_expr["s7"]:
        #                            if context.annotated_expr["z6"] == context.annotated_expr["z7"]:
        #                                return True
        #return False
        return True

    return ("gemmini.matmul_softmax", *_make_matmul_softmax_pattern(), _check_matmul_softmax_pattern)


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
            "input0": input0,
            "input1": input1,
            "matmul": matmul,
            "root": output,
        }
        return output, annotations

    def _check_matmul_transpose_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.matmul", *_make_matmul_transpose_pattern(), _check_matmul_transpose_pattern)


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
            "scale0": scale0_dq,
            "zp0": zp0_dq,
            "input1": input1,
            "scale1": scale1_dq,
            "zp1": zp1_dq,
            "add": add,
            "root": output,
        }
        return output, annotations

    def _check_resadd_leakyrelu_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.resadd_leakyrelu", *_make_resadd_leakyrelu_pattern(), _check_resadd_leakyrelu_pattern)


def conv2d_leakyrelu_pattern():
    """
    DQ -> conv2d(dq, int8_weight), bias -> reshape, add(conv2d, reshape) -> Q/DQ -> leakyrelu -> reshape -> Q
    """

    def _make_conv2d_leakyrelu_pattern():
        input = wildcard()
        scale_dq = wildcard()
        zp_dq = wildcard()
        q = is_op("relax.quantize")(input, scale_dq, zp_dq)
        requant = is_op("relax.dequantize")(input, scale_dq, zp_dq)

        weight = wildcard()
        #NOTE: It ignores stride and padding and does not check data layout and out_dtype
        conv = is_op("relax.nn.conv2d")(requant, weight)

        bias = wildcard()
        #NOTE: It does not check data layout
        reshape = is_op("relax.reshape")(bias, wildcard())
        add = is_op("relax.add")(conv, reshape)
        scale1_q = wildcard()
        zp1_q = wildcard()
        scale1_dq = wildcard()
        zp1_dq = wildcard()
        q1 = is_op("relax.quantize")(add, scale1_q, zp1_q)
        _r = is_op("relax.dequantize")(q1, scale1_dq, zp1_dq)
        act = is_op("relax.nn.leakyrelu")(_r)
        reshape = is_op("relax.reshape")(act, wildcard()) | act
        scale_q = wildcard()
        zp_q = wildcard()
        output = is_op("relax.quantize")(reshape, scale_q, zp_q)

        annotations = {
            "input": input,
            "scale": scale_q,
            "zp": zp_q,
            "weight": weight,
            "bias": reshape,
            "root": output,
        }
        return output, annotations

    def _check_conv2d_leakyrelu_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.conv2d_leakyrelu", *_make_conv2d_leakyrelu_pattern(), _check_conv2d_leakyrelu_pattern)


def fc_leakyrelu_pattern():
    """
    DQ, int8_weight -> permute_dims, -> matmul(requant, permute_dims), bias -> add(bias, matmul) -> requant -> leakyrelu -> Q
    """

    def _make_fc_leakyrelu_pattern():
        input = wildcard()
        scale_dq = wildcard()
        zp_dq = wildcard()
        lhs = is_op("relax.dequantize")(input, scale_dq, zp_dq)

        weight = wildcard()
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
            "input": input,
            "scale": scale_q,
            "zp": zp_q,
            "weight": rhs,
            "matmul": matmul,
            "bias": bias,
            "add": add,
            "root": output,
        }
        return output, annotations

    def _check_fc_leakyrelu_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.fc_leakyrelu", *_make_fc_leakyrelu_pattern(), _check_fc_leakyrelu_pattern)


def fc_pattern():
    """
    DQ, int8_weight -> permute_dims, -> matmul(requant, permute_dims), bias -> add(bias, matmul) -> Q
    """

    def _make_fc_pattern():
        input = wildcard()
        scale_dq = wildcard()
        zp_dq = wildcard()
        lhs = is_op("relax.dequantize")(input, scale_dq, zp_dq)

        weight = wildcard()
        rhs = is_op("relax.permute_dims")(weight)

        matmul = is_op("relax.matmul")(lhs, rhs)

        bias = wildcard()
        add = is_op("relax.add")(bias, matmul)

        scale_q = wildcard()
        zp_q = wildcard()
        output = is_op("relax.quantize")(add, scale_q, zp_q)

        annotations = {
            "input": input,
            "scale": scale_q,
            "zp": zp_q,
            "weight": rhs,
            "matmul": matmul,
            "bias": bias,
            "add": add,
            "root": output,
        }
        return output, annotations

    def _check_fc_pattern(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return ("gemmini.fc", *_make_fc_pattern(), _check_fc_pattern)


#def conv2d_transpose_fused_pattern():
#    """
#    Conv2d+Transpose fusion pattern.
#    """
#
#    def _make_conv2d_transpose_pattern():
#        input_tensor = wildcard()
#        weight = wildcard()
#        conv = is_op("relax.nn.conv2d")(input_tensor, weight)
#        transpose = is_op("relax.permute_dims")(conv)
#
#        annotations = {
#            "input": input_tensor,
#            "weight": weight,
#            "conv": conv,
#            "root": transpose,
#        }
#        return transpose, annotations
#
#    def _check_conv2d_transpose(context: PatternCheckContext) -> bool:
#        if not _check_gemmini_memory_constraints(context):
#            return False
#        if not _check_gemmini_quantization(context):
#            return False
#        return True
#
#    return ("gemmini.conv2d_transpose_fused", *_make_conv2d_transpose_pattern(), _check_conv2d_transpose)
#
#
#def conv2d_relu_fused_pattern():
#    """
#    Conv2D+ReLU fusion pattern.
#    """
#
#    def _make_conv2d_relu_pattern():
#        input_tensor = wildcard()
#        weight = wildcard()
#        conv = is_op("relax.nn.conv2d")(input_tensor, weight)
#        relu = is_op("relax.nn.relu")(conv)
#
#        annotations = {
#            "input": input_tensor,
#            "weight": weight,
#            "conv": conv,
#            "root": relu,
#        }
#        return relu, annotations
#
#    def _check_conv2d_relu(context: PatternCheckContext) -> bool:
#        if not _check_gemmini_memory_constraints(context):
#            return False
#        if not _check_gemmini_quantization(context):
#            return False
#        return True
#
#    return ("gemmini.conv2d_relu_fused", *_make_conv2d_relu_pattern(), _check_conv2d_relu)
#
#
#def matmul_patterns():
#    """
#    Systolic array based matrix multiplication patterns.
#    """
#
#    def _make_matmul_pattern():
#        input_tensor = wildcard()
#        weight = wildcard()
#        output = is_op("relax.matmul")(input_tensor, weight)
#
#        annotations = {
#            "input": input_tensor,
#            "weight": weight,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_matmul(context: PatternCheckContext) -> bool:
#        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)
#
#    def _matmul_pattern(pattern_name):
#        return (pattern_name, *_make_matmul_pattern(), _check_matmul)
#
#    # Register both common names used for matrix multiplication in patterns/tests
#    # return [
#    #     _matmul_pattern("gemmini.dense"),
#    #     _matmul_pattern("gemmini.matmul"),
#    # ]
#    return [_matmul_pattern("gemmini.matmul")]
#
#
#def conv1d_patterns():
#    """
#    1D Convolution patterns optimized for Gemmini execution.
#    """
#
#    def _make_conv1d_pattern():
#        input_tensor = wildcard()
#        weight = wildcard()
#        output = is_op("relax.nn.conv1d")(input_tensor, weight)
#
#        annotations = {
#            "input": input_tensor,
#            "weight": weight,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_conv1d(context: PatternCheckContext) -> bool:
#        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)
#
#    def _conv1d_pattern(pattern_name):
#        return (pattern_name, *_make_conv1d_pattern(), _check_conv1d)
#
#    return [_conv1d_pattern("gemmini.conv1d")]
#
#
#def conv2d_patterns():
#    """
#    2D Convolution patterns
#    """
#
#    def _make_conv2d_pattern():
#        input_tensor = wildcard()
#        weight = wildcard()
#        output = is_op("relax.nn.conv2d")(input_tensor, weight)
#
#        annotations = {
#            "input": input_tensor,
#            "weight": weight,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_conv2d(context: PatternCheckContext) -> bool:
#        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)
#
#    def _conv2d_pattern(pattern_name):
#        return (pattern_name, *_make_conv2d_pattern(), _check_conv2d)
#
#    return [_conv2d_pattern("gemmini.conv2d")]
#
#
#def depthwise_conv2d_patterns():
#    """
#    Depthwise convolution - critical for mobile NPUs.
#
#    Many NPUs have specialized units for depthwise operations
#    used in MobileNet-style architectures.
#    """
#
#    def _make_depthwise_pattern():
#        input_tensor = wildcard()
#        weight = wildcard()
#        output = is_op("relax.nn.conv2d")(input_tensor, weight)
#
#        annotations = {
#            "input": input_tensor,
#            "weight": weight,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_depthwise(context: PatternCheckContext) -> bool:
#        conv_call = context.annotated_expr["root"]
#        # groups > 1 distinguishes depthwise/grouped conv from standard conv2d.
#        # True depthwise has groups == in_channels; we accept any grouped variant
#        # here since the NPU's depthwise unit handles all grouped convolutions.
#        if conv_call.attrs.groups <= 1:
#            return False
#        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)
#
#    return [("gemmini.depthwise_conv2d", *_make_depthwise_pattern(), _check_depthwise)]
#
#
#def pooling_patterns():
#    """
#    Pooling operations
#    """
#
#    def _make_maxpool2d_pattern():
#        input_tensor = wildcard()
#        output = is_op("relax.nn.max_pool2d")(input_tensor)
#
#        annotations = {
#            "input": input_tensor,
#            "root": output,
#        }
#        return output, annotations
#
#    def _make_avgpool2d_pattern():
#        input_tensor = wildcard()
#        output = is_op("relax.nn.avg_pool2d")(input_tensor)
#
#        annotations = {
#            "input": input_tensor,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_pooling(context: PatternCheckContext) -> bool:
#        return _check_gemmini_memory_constraints(context)
#
#    return [
#        ("gemmini.max_pool2d", *_make_maxpool2d_pattern(), _check_pooling),
#        ("gemmini.avg_pool2d", *_make_avgpool2d_pattern(), _check_pooling),
#    ]
#
#
#def batch_norm_patterns():
#    """
#    Batch normalization - often fused with conv on NPUs.
#    """
#
#    def _make_batch_norm_pattern():
#        input_tensor = wildcard()
#        gamma = wildcard()
#        beta = wildcard()
#        moving_mean = wildcard()
#        moving_var = wildcard()
#
#        output = is_op("relax.nn.batch_norm")(input_tensor, gamma, beta, moving_mean, moving_var)
#
#        annotations = {
#            "input": input_tensor,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_batch_norm(context: PatternCheckContext) -> bool:
#        return _check_gemmini_quantization(context)
#
#    return [("gemmini.batch_norm", *_make_batch_norm_pattern(), _check_batch_norm)]
#
#
#def softmax_patterns():
#    """
#    Softmax
#    """
#
#    def _make_softmax_pattern():
#        input_tensor = wildcard()
#        output = is_op("relax.nn.softmax")(input_tensor)
#
#        annotations = {
#            "input": input_tensor,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_softmax(context: PatternCheckContext) -> bool:
#        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)
#
#    patterns = []
#    try:
#        Op.get("relax.nn.softmax")
#        patterns.append(("gemmini.softmax", *_make_softmax_pattern(), _check_softmax))
#    except TVMError:  # pylint: disable=broad-exception-caught
#        pass
#
#    return patterns
#
#
#def activation_patterns():
#    """
#    Gemmini activation functions with specialized hardware.
#    """
#
#    def _make_activation_pattern(op_name: str):
#        def _pattern():
#            input_tensor = wildcard()
#            output = is_op(op_name)(input_tensor)
#
#            annotations = {
#                "input": input_tensor,
#                "root": output,
#            }
#            return output, annotations
#
#        return _pattern
#
#    def _check_activation(context: PatternCheckContext) -> bool:
#        return _check_gemmini_quantization(context)
#
#    activations = [
#        ("gemmini.relu", "relax.nn.relu"),
#        ("gemmini.relu6", "relax.nn.relu6"),
#        ("gemmini.sigmoid", "relax.nn.sigmoid"),
#        ("gemmini.tanh", "relax.nn.tanh"),
#        ("gemmini.gelu", "relax.nn.gelu"),
#    ]
#
#    patterns = []
#    for pattern_name, op_name in activations:
#        try:
#            Op.get(op_name)
#        except TVMError:  # pylint: disable=broad-exception-caught
#            continue
#
#        pattern_fn = _make_activation_pattern(op_name)
#        patterns.append((pattern_name, *pattern_fn(), _check_activation))
#
#    return patterns
#
#
#def elementwise_patterns():
#    """
#    Element-wise operations
#    """
#
#    def _make_elementwise_pattern(op_name: str):
#        def _pattern():
#            input1 = wildcard()
#            input2 = wildcard()
#            output = is_op(op_name)(input1, input2)
#
#            annotations = {
#                "input1": input1,
#                "input2": input2,
#                "root": output,
#            }
#            return output, annotations
#
#        return _pattern
#
#    def _check_elementwise(context: PatternCheckContext) -> bool:
#        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)
#
#    ops = ["relax.add", "relax.multiply", "relax.subtract", "relax.divide"]
#    patterns = []
#    for op in ops:
#        try:
#            Op.get(op)
#        except TVMError:  # pylint: disable=broad-exception-caught
#            continue
#
#        op_short = op.split(".")[-1]
#        pattern_fn = _make_elementwise_pattern(op)
#        patterns.append((f"gemmini.{op_short}", *pattern_fn(), _check_elementwise))
#
#    return patterns


#def requant_pattern():
#    """
#    quantize -> dequantize, dequantize -> quantize (when they have same scale and zp)
#    """
#
#    def _make_requant_pattern0():
#        input = wildcard()
#        scale = wildcard()
#        zp = wildcard()
#        _q = is_op("relax.quantize")(input, scale, zp)
#        output = is_op("relax.dequantize")(_q, scale, zp)
#
#        annotations = {
#            "input": input,
#            "scale": scale,
#            "zp": zp,
#            "root": output,
#        }
#        return output, annotations
#
#    def _make_requant_pattern1():
#        input = wildcard()
#        scale = wildcard()
#        zp = wildcard()
#        _d = is_op("relax.dequantize")(input, scale, zp)
#        output = is_op("relax.quantize")(_d, scale, zp)
#
#        annotations = {
#            "input": input,
#            "scale": scale,
#            "zp": zp,
#            "root": output,
#        }
#        return output, annotations
#
#    def _check_requant(
#        context: PatternCheckContext,  # pylint: disable=unused-argument
#    ) -> bool:
#        ## NOTE: It does not check output dtype
#        #if context.annotated_expr["scale0"] == context.annotated_expr["scale1"]:
#        #    if context.annotated_expr["zp0"] == context.annotated_expr["zp1"]:
#        #        return True
#        #return False
#        return True
#
#    patterns = []
#
#    try:
#        Op.get("relax.quantize")
#        patterns.append(("gemmini.requant", *_make_requant_pattern0(), _check_requant))
#    except TVMError:  # pylint: disable=broad-exception-caught
#        pass
#
#    try:
#        Op.get("relax.dequantize")
#        patterns.append(
#            ("gemmini.requant", *_make_requant_pattern1(), _check_requant)
#        )
#    except TVMError:  # pylint: disable=broad-exception-caught
#        pass
#
#    return patterns

# Register all Gemmini patterns with architectural awareness
register_patterns(
    [
        conv2d_pattern(), # Fused patterns first (higher priority)
        matmul_softmax_pattern(),
        matmul_transpose_pattern(),
        resadd_leakyrelu_pattern(),
        conv2d_leakyrelu_pattern(),
        fc_pattern(),
        fc_leakyrelu_pattern(),
        #*requant_pattern(),
        #conv2d_transpose_fused_pattern(),
        #conv2d_relu_fused_pattern(),
        #*matmul_patterns(),
        #*conv1d_patterns(),
        #*conv2d_patterns(),
        #*depthwise_conv2d_patterns(),
        #*pooling_patterns(),
        #*batch_norm_patterns(),
        #*softmax_patterns(),
        #*activation_patterns(),
        #*elementwise_patterns(),
        #*quantization_patterns(),
    ]
)

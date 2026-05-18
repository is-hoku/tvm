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
    TILE_SIZE=16


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

def dequantize_conv2d_quantize_fused_pattern():
    """
    Dequantize+Conv2d+Quantize fusion pattern.
    """

    def _make_dequantize_conv2d_quantize_pattern():
        input = wildcard()
        scale = wildcard()
        zp = wildcard()
        # NOTE: It does not check out_dtype
        dequantized_tensor = is_op("relax.dequantize")(input, scale, zp)

        weight = wildcard()
        #NOTE: It ignores stride and padding and does not check data layout and out_dtype
        conv = is_op("relax.nn.conv2d")(dequantized_tensor, weight)

        bias = wildcard()
        #NOTE: It does not check data layout
        reshape = is_op("relax.reshape")(bias)
        resadd = is_op("relax.add")(conv, reshape)
        output = is_op("relax.quantize")(resadd)

        annotations = {
            "input": input,
            "scale": scale,
            "zp": zp,
            "weight": weight,
            "bias": reshape,
            "root": output,
        }
        return output, annotations


def conv2d_transpose_fused_pattern():
    """
    Conv2d+Transpose fusion pattern.
    """

    def _make_conv2d_transpose_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        conv = is_op("relax.nn.conv2d")(input_tensor, weight)
        transpose = is_op("relax.permute_dims")(conv)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "conv": conv,
            "root": transpose,
        }
        return transpose, annotations

    def _check_conv2d_transpose(context: PatternCheckContext) -> bool:
        if not _check_gemmini_memory_constraints(context):
            return False
        if not _check_gemmini_quantization(context):
            return False
        return True

    return ("gemmini.conv2d_transpose_fused", *_make_conv2d_transpose_pattern(), _check_conv2d_transpose)


def conv2d_relu_fused_pattern():
    """
    Conv2D+ReLU fusion pattern.
    """

    def _make_conv2d_relu_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        conv = is_op("relax.nn.conv2d")(input_tensor, weight)
        relu = is_op("relax.nn.relu")(conv)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "conv": conv,
            "root": relu,
        }
        return relu, annotations

    def _check_conv2d_relu(context: PatternCheckContext) -> bool:
        if not _check_gemmini_memory_constraints(context):
            return False
        if not _check_gemmini_quantization(context):
            return False
        return True

    return ("gemmini.conv2d_relu_fused", *_make_conv2d_relu_pattern(), _check_conv2d_relu)


def matmul_patterns():
    """
    Systolic array based matrix multiplication patterns.
    """

    def _make_matmul_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.matmul")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_matmul(context: PatternCheckContext) -> bool:
        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)

    def _matmul_pattern(pattern_name):
        return (pattern_name, *_make_matmul_pattern(), _check_matmul)

    # Register both common names used for matrix multiplication in patterns/tests
    # return [
    #     _matmul_pattern("gemmini.dense"),
    #     _matmul_pattern("gemmini.matmul"),
    # ]
    return [_matmul_pattern("gemmini.matmul")]


def conv1d_patterns():
    """
    1D Convolution patterns optimized for Gemmini execution.
    """

    def _make_conv1d_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv1d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_conv1d(context: PatternCheckContext) -> bool:
        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)

    def _conv1d_pattern(pattern_name):
        return (pattern_name, *_make_conv1d_pattern(), _check_conv1d)

    return [_conv1d_pattern("gemmini.conv1d")]


def conv2d_patterns():
    """
    2D Convolution patterns
    """

    def _make_conv2d_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_conv2d(context: PatternCheckContext) -> bool:
        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)

    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_conv2d_pattern(), _check_conv2d)

    return [_conv2d_pattern("gemmini.conv2d")]


def depthwise_conv2d_patterns():
    """
    Depthwise convolution - critical for mobile NPUs.

    Many NPUs have specialized units for depthwise operations
    used in MobileNet-style architectures.
    """

    def _make_depthwise_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_depthwise(context: PatternCheckContext) -> bool:
        conv_call = context.annotated_expr["root"]
        # groups > 1 distinguishes depthwise/grouped conv from standard conv2d.
        # True depthwise has groups == in_channels; we accept any grouped variant
        # here since the NPU's depthwise unit handles all grouped convolutions.
        if conv_call.attrs.groups <= 1:
            return False
        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)

    return [("gemmini.depthwise_conv2d", *_make_depthwise_pattern(), _check_depthwise)]


def pooling_patterns():
    """
    Pooling operations
    """

    def _make_maxpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.max_pool2d")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _make_avgpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.avg_pool2d")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_pooling(context: PatternCheckContext) -> bool:
        return _check_gemmini_memory_constraints(context)

    return [
        ("gemmini.max_pool2d", *_make_maxpool2d_pattern(), _check_pooling),
        ("gemmini.avg_pool2d", *_make_avgpool2d_pattern(), _check_pooling),
    ]


def batch_norm_patterns():
    """
    Batch normalization - often fused with conv on NPUs.
    """

    def _make_batch_norm_pattern():
        input_tensor = wildcard()
        gamma = wildcard()
        beta = wildcard()
        moving_mean = wildcard()
        moving_var = wildcard()

        output = is_op("relax.nn.batch_norm")(input_tensor, gamma, beta, moving_mean, moving_var)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_batch_norm(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    return [("gemmini.batch_norm", *_make_batch_norm_pattern(), _check_batch_norm)]


def softmax_patterns():
    """
    Softmax
    """

    def _make_softmax_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.softmax")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_softmax(context: PatternCheckContext) -> bool:
        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)

    patterns = []
    try:
        Op.get("relax.nn.softmax")
        patterns.append(("gemmini.softmax", *_make_softmax_pattern(), _check_softmax))
    except TVMError:  # pylint: disable=broad-exception-caught
        pass

    return patterns


def activation_patterns():
    """
    Gemmini activation functions with specialized hardware.
    """

    def _make_activation_pattern(op_name: str):
        def _pattern():
            input_tensor = wildcard()
            output = is_op(op_name)(input_tensor)

            annotations = {
                "input": input_tensor,
                "root": output,
            }
            return output, annotations

        return _pattern

    def _check_activation(context: PatternCheckContext) -> bool:
        return _check_gemmini_quantization(context)

    activations = [
        ("gemmini.relu", "relax.nn.relu"),
        ("gemmini.relu6", "relax.nn.relu6"),
        ("gemmini.sigmoid", "relax.nn.sigmoid"),
        ("gemmini.tanh", "relax.nn.tanh"),
        ("gemmini.gelu", "relax.nn.gelu"),
    ]

    patterns = []
    for pattern_name, op_name in activations:
        try:
            Op.get(op_name)
        except TVMError:  # pylint: disable=broad-exception-caught
            continue

        pattern_fn = _make_activation_pattern(op_name)
        patterns.append((pattern_name, *pattern_fn(), _check_activation))

    return patterns


def elementwise_patterns():
    """
    Element-wise operations
    """

    def _make_elementwise_pattern(op_name: str):
        def _pattern():
            input1 = wildcard()
            input2 = wildcard()
            output = is_op(op_name)(input1, input2)

            annotations = {
                "input1": input1,
                "input2": input2,
                "root": output,
            }
            return output, annotations

        return _pattern

    def _check_elementwise(context: PatternCheckContext) -> bool:
        return _check_gemmini_memory_constraints(context) and _check_gemmini_quantization(context)

    ops = ["relax.add", "relax.multiply", "relax.subtract", "relax.divide"]
    patterns = []
    for op in ops:
        try:
            Op.get(op)
        except TVMError:  # pylint: disable=broad-exception-caught
            continue

        op_short = op.split(".")[-1]
        pattern_fn = _make_elementwise_pattern(op)
        patterns.append((f"gemmini.{op_short}", *pattern_fn(), _check_elementwise))

    return patterns


def quantization_patterns():
    """
    Quantization/dequantization patterns
    """

    def _make_quantize_pattern():
        input_tensor = wildcard()
        output = is_op("relax.quantize")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _make_dequantize_pattern():
        input_tensor = wildcard()
        output = is_op("relax.dequantize")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_quantization(
        context: PatternCheckContext,  # pylint: disable=unused-argument
    ) -> bool:
        return True

    patterns = []

    try:
        Op.get("relax.quantize")
        patterns.append(("gemmini.quantize", *_make_quantize_pattern(), _check_quantization))
    except TVMError:  # pylint: disable=broad-exception-caught
        pass

    try:
        Op.get("relax.dequantize")
        patterns.append(
            ("gemmini.dequantize", *_make_dequantize_pattern(), _check_quantization)
        )
    except TVMError:  # pylint: disable=broad-exception-caught
        pass

    return patterns


# Register all Gemmini patterns with architectural awareness
register_patterns(
    [
        conv2d_transpose_fused_pattern(),  # Fused patterns first (higher priority)
        conv2d_relu_fused_pattern(),  # Fused patterns first (higher priority)
        *matmul_patterns(),
        *conv1d_patterns(),
        *conv2d_patterns(),
        *depthwise_conv2d_patterns(),
        *pooling_patterns(),
        *batch_norm_patterns(),
        *softmax_patterns(),
        *activation_patterns(),
        *elementwise_patterns(),
        *quantization_patterns(),
    ]
)

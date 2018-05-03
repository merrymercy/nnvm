from __future__ import absolute_import

import tvm
import topi

from . import registry as reg
from .registry import OpPattern

@reg.register_compute("_contrib_conv2d_winograd_6x6_3x3_weight_transform")
def compute_contrib_conv2d_winograd_6x6_3x3_weight_transform(attrs, inputs, _):
    return topi.contrib.conv2d_winograd_6x6_3x3_weight_transform(inputs[0])

@reg.register_schedule("_contrib_conv2d_winograd_6x6_3x3_weight_transform")
def schedule_contrib_conv2d_winograd_6x6_3x3_weight_transform(attrs, outs, target):
    with tvm.target.create(target):
        return topi.contrib.schedule_conv2d_winograd_6x6_3x3_weight_transform(outs)

reg.register_pattern("_contrib_conv2d_winograd_6x6_3x3_weight_transform", OpPattern.INJECTIVE)

@reg.register_compute("_contrib_conv2d_winograd_6x6_3x3_without_weight_transform")
def compute_contrib_conv2d_winograd_6x6_3x3_without_weight_transform(attrs, inputs, _):
    """Compute definition of conv2d"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]
    assert layout == "NCHW" or layout == "NHWC"
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    elif layout == "NCHW":
        kernel = topi.nn.dilate(inputs[1], [1, 1, dilation_h, dilation_w])
    else: #layout == NHWC
        kernel = topi.nn.dilate(inputs[1], [1, dilation_h, dilation_w, 1])

    assert groups == 1
    assert layout == 'NCHW'

    out = topi.contrib.conv2d_winograd_6x6_3x3_without_weight_transform(
        inputs[0], kernel, strides, padding, layout)

    if attrs.get_bool("use_bias"):
        bias = inputs[2]
        expand_axis = 1 if layout == "NCHW" else 0
        bias = topi.expand_dims(bias, axis=expand_axis, num_newaxis=2)
        out = topi.broadcast_add(out, bias)
    return out

@reg.register_schedule("_contrib_conv2d_winograd_6x6_3x3_without_weight_transform")
def schedule_contrib_conv2d_winograd_6x6_3x3_without_weight_transform(attrs, outs, target):
    with tvm.target.create(target):
        return topi.contrib.schedule_conv2d_winograd_6x6_3x3_without_weight_transform(outs)

reg.register_pattern("_contrib_conv2d_winograd_6x6_3x3_without_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_compute("_contrib_conv2d_spatial_pack_weight_prepack")
def compute_contrib_conv2d_spatial_pack_weight_prepack(attrs, inputs, _):
    block_factor = attrs.get_int("block_factor")
    return topi.contrib.conv2d_spatial_pack_weight_prepack(inputs[0], block_factor=block_factor)

@reg.register_schedule("_contrib_conv2d_spatial_pack_weight_prepack")
def schedule_contrib_conv2d_spatial_pack_weight_prepack(attrs, outs, target):
    with tvm.target.create(target):
        return topi.contrib.schedule_conv2d_spatial_pack_weight_prepack(outs)

reg.register_pattern("_contrib_conv2d_spatial_pack_weight_prepack", OpPattern.INJECTIVE)


@reg.register_compute("_contrib_conv2d_spatial_pack_without_weight_prepack")
def compute_contrib_conv2d_spatial_pack_without_weight_prepack(attrs, inputs, _):
    """Compute definition of conv2d"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]
    assert layout == "NCHW" or layout == "NHWC"
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    assert layout == "NCHW"
    kernel = topi.nn.dilate(inputs[1], [1, 1, dilation_h, dilation_w, 1])

    assert groups == 1
    assert layout == 'NCHW'

    out = topi.contrib.conv2d_spatial_pack_without_weight_prepack(
        inputs[0], kernel, strides, padding, layout)

    if attrs.get_bool("use_bias"):
        bias = inputs[2]
        expand_axis = 1 if layout == "NCHW" else 0
        bias = topi.expand_dims(bias, axis=expand_axis, num_newaxis=2)
        out = topi.broadcast_add(out, bias)
    return out

@reg.register_schedule("_contrib_conv2d_spatial_pack_without_weight_prepack")
def schedule_contrib_conv2d_spatial_pack_without_weight_prepack(attrs, outs, target):
    with tvm.target.create(target):
        return topi.contrib.schedule_conv2d_spatial_pack_without_weight_prepack(outs)

reg.register_pattern("_contrib_conv2d_spatial_pack_without_weight_prepack", OpPattern.OUT_ELEMWISE_FUSABLE)

# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from collections import OrderedDict
from typing import Dict, List

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from . import resnet
from .intermediate_layer_getter import IntermediateLayerGetter
from .position_encoding import build_position_encoding
from util.misc import NestedTensor

class FrozenBatchNorm2d(dg.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.module = dg.BatchNorm(n, use_global_stats=True, param_attr=F.ParamAttr(learning_rate=0))
    

    def forward(self, x):
        return self.module(x)


class BackboneBase(dg.Layer):

    def __init__(self, backbone: dg.Layer, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.stop_gradient = True
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            m = L.unsqueeze(m, 1) # [batch_size, h, w] -> [batch_size, 1, h, w]
            m = m.astype("float32")
            mask = L.image_resize(m, out_shape=x.shape[-2:], resample="NEAREST")
            mask = mask.astype("bool")
            mask = L.squeeze(mask, [1]) # [batch_size, 1, h, w] -> [batch_size, h, w]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(resnet, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(dg.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(("0", backbone),
                         ("1", position_embedding))
    

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x))
        
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

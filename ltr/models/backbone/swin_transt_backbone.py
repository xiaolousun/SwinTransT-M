# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
import ltr.models.backbone as backbones

from util.misc import NestedTensor

from ltr.models.neck.position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 pretrained,
                 frozen_layers):
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=pretrained,
                                      frozen_layers=frozen_layers)
        num_channels = 1024
        super().__init__(backbone, num_channels)

class SwintranstBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 frozen_stages,
                 pretrained_model_path,
                 **params):
        backbone = backbones.SwinTransformer(out_indices=output_layers, frozen_stages=frozen_stages, **params)
        backbone.init_weights(pretrained_model_path)
        num_channels = None
        super().__init__(backbone, num_channels)

class SwintranstCBAM_Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 frozen_stages,
                 pretrained_model_path,
                 **params):
        backbone = backbones.SwinTransformer_CBAM(out_indices=output_layers, frozen_stages=frozen_stages, **params)
        backbone.init_weights(pretrained_model_path)
        num_channels = None
        super().__init__(backbone, num_channels)

class SwintranstDiffcnn_Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 frozen_stages,
                 pretrained_model_path,
                 **params):
        backbone = backbones.SwinTransformer_diffcnn(out_indices=output_layers, frozen_stages=frozen_stages, **params)
        backbone.init_weights(pretrained_model_path)
        num_channels = None
        super().__init__(backbone, num_channels)

class SwintranstBigkernel_Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 frozen_stages,
                 pretrained_model_path,
                 **params):
        backbone = backbones.SwinTransformer_Bigkernel(out_indices=output_layers, frozen_stages=frozen_stages, **params)
        backbone.init_weights(pretrained_model_path)
        num_channels = None
        super().__init__(backbone, num_channels)

class SwintranstCvt_Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 output_layers,
                 frozen_stages,
                 pretrained_model_path,
                 **params):
        backbone = backbones.SwinTransformer_cvt(out_indices=output_layers, frozen_stages=frozen_stages, **params)
        backbone.init_weights(pretrained_model_path)
        num_channels = None
        super().__init__(backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(settings, backbone_pretrained=True, frozen_backbone_layers=()):
    position_embedding = build_position_encoding(settings)
    backbone = Backbone(output_layers=['layer3'], pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_swin_transformer_backbone(settings, name, load_pretrained=False, output_layers=(3,), frozen_stages=-1,
                                    overwrite_embed_dim=None):
    import copy
    from .swin_transformer import _cfg

    pretrained_model_path = None
    if load_pretrained and 'url' in _cfg[name]:
        pretrained_model_path = _cfg[name]['url']

    max_output_index = max(output_layers)
    params = copy.deepcopy(_cfg[name]['params'])
    if max_output_index < 3:
        params['depths'] = params['depths'][0: max_output_index + 1]
        params['num_heads'] = params['num_heads'][0: max_output_index + 1]

    if overwrite_embed_dim is not None:
        params['embed_dim'] = overwrite_embed_dim
        pretrained_model_path = None

    position_embedding = build_position_encoding(settings)
    backbone = SwintranstBackbone(output_layers, frozen_stages, pretrained_model_path, **params)
    backbone.num_channels = params['embed_dim'] * 2 ** (len(params['depths'])-1)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

def build_swin_transformer_CBAM_backbone(settings, name, load_pretrained=False, output_layers=(3,), frozen_stages=-1,
                                    overwrite_embed_dim=None):
    import copy
    from .swin_transformer import _cfg

    pretrained_model_path = None
    if load_pretrained and 'url' in _cfg[name]:
        pretrained_model_path = _cfg[name]['url']

    max_output_index = max(output_layers)
    params = copy.deepcopy(_cfg[name]['params'])
    if max_output_index < 3:
        params['depths'] = params['depths'][0: max_output_index + 1]
        params['num_heads'] = params['num_heads'][0: max_output_index + 1]

    if overwrite_embed_dim is not None:
        params['embed_dim'] = overwrite_embed_dim
        pretrained_model_path = None

    position_embedding = build_position_encoding(settings)
    backbone = SwintranstCBAM_Backbone(output_layers, frozen_stages, pretrained_model_path, **params)
    backbone.num_channels = params['embed_dim'] * 2 ** (len(params['depths'])-1)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model


def build_swin_transformer_diffcnn_backbone(settings, name, load_pretrained=False, output_layers=(3,), frozen_stages=-1,
                                    overwrite_embed_dim=None):
    import copy
    from .swin_transformer import _cfg

    pretrained_model_path = None
    if load_pretrained and 'url' in _cfg[name]:
        pretrained_model_path = _cfg[name]['url']

    max_output_index = max(output_layers)
    params = copy.deepcopy(_cfg[name]['params'])
    if max_output_index < 3:
        params['depths'] = params['depths'][0: max_output_index + 1]
        params['num_heads'] = params['num_heads'][0: max_output_index + 1]

    if overwrite_embed_dim is not None:
        params['embed_dim'] = overwrite_embed_dim
        pretrained_model_path = None

    position_embedding = build_position_encoding(settings)
    backbone = SwintranstDiffcnn_Backbone(output_layers, frozen_stages, pretrained_model_path, **params)
    backbone.num_channels = params['embed_dim'] * 2 ** (len(params['depths'])-1)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model


def build_swin_transformer_bigkernel_backbone(settings, name, load_pretrained=False, output_layers=(3,), frozen_stages=-1,
                                    overwrite_embed_dim=None):
    import copy
    from .swin_transformer import _cfg

    pretrained_model_path = None
    if load_pretrained and 'url' in _cfg[name]:
        pretrained_model_path = _cfg[name]['url']

    max_output_index = max(output_layers)
    params = copy.deepcopy(_cfg[name]['params'])
    if max_output_index < 3:
        params['depths'] = params['depths'][0: max_output_index + 1]
        params['num_heads'] = params['num_heads'][0: max_output_index + 1]

    if overwrite_embed_dim is not None:
        params['embed_dim'] = overwrite_embed_dim
        pretrained_model_path = None

    position_embedding = build_position_encoding(settings)
    backbone = SwintranstBigkernel_Backbone(output_layers, frozen_stages, pretrained_model_path, **params)
    backbone.num_channels = params['embed_dim'] * 2 ** (len(params['depths'])-1)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

def build_swin_transformer_cvt_backbone(settings, name, load_pretrained=False, output_layers=(3,), frozen_stages=-1,
                                    overwrite_embed_dim=None):
    import copy
    from .swin_transformer_qkvcnn import _cfg

    pretrained_model_path = None
    if load_pretrained and 'url' in _cfg[name]:
        pretrained_model_path = _cfg[name]['url']

    max_output_index = max(output_layers)
    params = copy.deepcopy(_cfg[name]['params'])
    if max_output_index < 3:
        params['depths'] = params['depths'][0: max_output_index + 1]
        params['num_heads'] = params['num_heads'][0: max_output_index + 1]

    if overwrite_embed_dim is not None:
        params['embed_dim'] = overwrite_embed_dim
        pretrained_model_path = None

    position_embedding = build_position_encoding(settings)
    backbone = SwintranstCvt_Backbone(output_layers, frozen_stages, pretrained_model_path, **params)
    backbone.num_channels = params['embed_dim'] * 2 ** (max(output_layers))
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model, max(output_layers)


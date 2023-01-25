import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor, interpolate, 
                       nested_tensor_from_tensor_2, nested_tensor_from_tensor_list,
                       accuracy)

from ltr.models.backbone.swin_transt_backbone import build_swin_transformer_cvt_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.neck.encoder_featurefusion_network import build_encoder_featurefusion_network


class SwinTransTiouh(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, SwinTranT, freeze_transt=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = SwinTranT.featurefusion_network
        self.class_embed = SwinTranT.class_embed
        self.bbox_embed = SwinTranT.bbox_embed
        self.input_proj = SwinTranT.input_proj
        self.backbone = SwinTranT.backbone

        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim = self.featurefusion_network.d_model
        self.iou_embed = MLP(hidden_dim + 4, hidden_dim, 1, 3)

    def forward(self, search, templates):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        bs, n_t, c, h, w = templates.shape
        templates = templates.reshape(bs * n_t, c, h, w)
        if not isinstance(templates, NestedTensor):
            templates = nested_tensor_from_tensor(templates)
        feature_templates, pos_templates = self.backbone(templates)
        src_templates, mask_templates = feature_templates[-1].decompose()
        src_templates = self.input_proj(src_templates)
        _, c_src, h_src, w_src = src_templates.shape
        pos_templates = pos_templates[-1].reshape(bs, n_t, c_src, h_src, w_src)
        src_templates = src_templates.reshape(bs, n_t, c_src, h_src, w_src)
        mask_templates = mask_templates.reshape(bs, n_t, h_src, w_src)

        _, hs = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search[-1])
        hs = hs.flatten(2).permute(0, 2, 1).unsqueeze(0)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}
        return out

    def track(self, search, templates: list):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        feature_search, pos_search = self.backbone(search)
        src_search, mask_search= feature_search[-1].decompose()
        src_search = self.input_proj(src_search)

        for i in range(len(templates)):
            if i == 0:
                src_templates = templates[i]['src']
                mask_templates = templates[i]['mask']
                pos_templates = templates[i]['pos']
            else:
                src_templates = torch.cat((src_templates, templates[i]['src']), 1)
                mask_templates = torch.cat((mask_templates, templates[i]['mask']), 1)
                pos_templates = torch.cat((pos_templates, templates[i]['pos']), 1)

        _, hs = self.featurefusion_network(src_templates, mask_templates, pos_templates,
                                              src_search, mask_search, pos_search[-1])
        hs = hs.flatten(2).permute(0, 2, 1).unsqueeze(0)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((hs, outputs_coord), 3)).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        feature_template, pos_template = self.backbone(z)
        src_template, mask_template = feature_template[-1].decompose()
        template_out = {
            'pos': pos_template[-1].unsqueeze(1),
            'src': self.input_proj(src_template).unsqueeze(1),
            'mask': mask_template.unsqueeze(1)
        }
        return template_out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

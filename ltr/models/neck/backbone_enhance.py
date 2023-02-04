import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ltr.models.backbone.convolutional_block import build_DwConv_Block

class TransformerDWconvEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False, need_dwconv=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.need_conv = need_dwconv

        if need_dwconv:
            self.dwconv_block = build_DwConv_Block(in_planes=d_model, out_planes=d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        B, C, H, W = src.shape
        if self.need_conv:
            src = self.dwconv_block(src)

        src = src.flatten(2).permute(2, 0, 1)
        # pos = pos.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(1)

        q = k = self.with_pos_embed(src, pos)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(B, C, H, W)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_intermediate=False):
        if return_intermediate:
            output_list = []
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is None:
                    output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.norm is not None:
                output = self.norm(output)

            return output


class Transformer_DWconvEnhance(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False, need_dwconv=True):
        super().__init__()
        self.dim = d_model

        # self.dwconv_block = build_DwConv_Block(in_planes=self.dim, out_planes=self.dim)

        encoder_layer = TransformerDWconvEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm, need_dwconv=need_dwconv)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos, mode="encoder"):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        
        # B, C, H, W = src.shape
        # src = self.dwconv_block(src)

        # src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        assert mode in ["all", "encoder"]
        if self.encoder is None:
            memory = src
        else:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)

        # memory = memory.permute(1, 2, 0).reshape(B, C, H, W)
        return memory

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_backbone_enhance_network(params, settings, output_layer, need_dwconv=True):
    return Transformer_DWconvEnhance(
        d_model=params['embed_dim'] * 2 ** (output_layer),
        dropout=0.1,
        nhead=params['num_heads'][-1], 
        dim_feedforward=(params['embed_dim'] * 2 ** (output_layer))*4,
        num_encoder_layers=params["depths"][-1],
        need_dwconv=need_dwconv
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from paddle.nn.layer.transformer with modifications:
    * positional encodings are passes in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import numpy as np
from typing import Optional, List
import paddle.nn as nn
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg
from paddle.nn.layer.common import Linear, Dropout
from paddle.nn.layer.norm import LayerNorm
import paddle.nn.functional as F
from paddle.fluid.param_attr import ParamAttr
from paddle.nn.layer.transformer import (
      MultiHeadAttention, 
#     TransformerEncoderLayer, 
#     TransformerDecoderLayer,
#     TransformerEncoder,
#     TransformerDecoder,
)


class Transformer(dg.Layer):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = dg.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = dg.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        self.d_model = d_model
        self.nhead = nhead
    

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = L.reshape(src, (bs, c, -1)) # [bs, c, h * w]
        src = L.transpose(src, (0, 2, 1)) # [bs, h * w, c]

        pos_embed = L.reshape(pos_embed, (bs, pos_embed.shape[1], -1)) # [bs, c, h * w]
        pos_embed = L.transpose(pos_embed, (0, 2, 1)) # [bs, h * w, c]

        query_embed = L.unsqueeze(query_embed, [0])     # [1, num_queries, c_q]
        query_embed = L.expand(query_embed, (bs, 1, 1)) # [bs, num_queries, c_q]

        mask = L.reshape(mask, (bs, -1)) # [bs, h * w]

        tgt = L.zeros_like(query_embed) # [bs, num_queries, c_q]

        memory = self.encoder(src, src_mask=mask, pos=pos_embed) # [bs, h * w, c]
        hs = self.decoder(tgt, memory, memory_mask=mask, pos=pos_embed, query_pos=query_embed)
        # hs: [num_inter, bs, num_queries, c_q]

        memory = L.transpose(memory, (0, 2, 1)) # [bs, c, h * w]
        memory = L.reshape(memory, (bs, c, h, w)) # [bs, c, h, w]
        return hs, memory


def get_attention_mask(mask, nhead):
    # mask: [bs, L] -> attn_mask: [bs, nhead, L, L]
    bs, l = mask.shape
    row_mask = L.expand(L.unsqueeze(mask, [2]), (1, 1, l)) # [bs, L, L]
    col_mask = L.expand(L.unsqueeze(mask, [1]), (1, l, 1)) # [bs, L, L]
    mask = L.logical_or(row_mask, col_mask)
    attn_mask = L.zeros([bs, l, l], dtype="float32")
    attn_mask = attn_mask.numpy()
    mask = mask.numpy()
    attn_mask[mask] = -1e8
    attn_mask = dg.to_variable(attn_mask)
    attn_mask = L.expand(L.unsqueeze(attn_mask, [1]), (1, nhead, 1, 1)) # [bs, nhead, L1, L2]
    return attn_mask


class TransformerEncoder(dg.Layer):
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = dg.LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = encoder_layer.nhead
    

    def forward(self, src, src_mask=None, pos=None):
        output = src
        
        if src_mask is not None:
            src_mask = get_attention_mask(src_mask, self.nhead)

        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(dg.Layer):
    
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = dg.LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(**decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.nhead = decoder_layer.nhead
    

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                pos=None, query_pos=None):
        output = tgt

        intermediate = []

        assert tgt_mask is None, "Not implement compute tgt_mask's attn_mask."

        if memory_mask is not None:
            bs, tgt_length = tgt.shape[:2]
            memory_length = memory.shape[1]
            attn_mask = L.zeros([bs, tgt_length, memory_length], dtype="float32")
            memory_mask = L.expand(L.unsqueeze(memory_mask, [1]), (1, tgt_length, 1)) # [bs, tgt_length, memory_length]
            attn_mask = attn_mask.numpy()
            memory_mask = memory_mask.numpy()
            attn_mask[memory_mask] = -1e8
            attn_mask = dg.to_variable(attn_mask)
            attn_mask = L.expand(L.unsqueeze(attn_mask, [1]), (1, self.nhead, 1, 1)) # [bs, nhead, tgt_length, memory_length]
            memory_mask = attn_mask

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           pos=pos, query_pos=query_pos)
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return L.stack(intermediate)
        
        return L.unsqueeze(output, [0])


class TransformerEncoderLayer(dg.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 activation="relu", normalize_before=False, 
                 weight_attr=None, bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.nhead = nhead
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout,
                                            weight_attr=weight_attrs[0], bias_attr=bias_attrs[0])
        self.linear1 = Linear(d_model, dim_feedforward, 
                              weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(dropout, mode="upscale_in_train")
        self.linear2 = Linear(dim_feedforward, d_model, 
                              weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
    

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


    def forward_post(self, src, src_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, src, src_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


    def forward_pre(self, src, src_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src, src_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


    def forward(self, src, src_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, pos)
        return self.forward_post(src, src_mask, pos)


class TransformerDecoderLayer(dg.Layer):

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 activation="relu", normalize_before=False,
                 weight_attr=None, bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.nhead = nhead
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout,
                                            weight_attr=weight_attrs[0], bias_attr=bias_attrs[0])
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout,
                                             weight_attr=weight_attrs[1], bias_attr=bias_attrs[1])
        self.linear1 = Linear(d_model, dim_feedforward, 
                              weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = Dropout(dropout, mode="upscale_in_train")
        self.linear2 = Linear(dim_feedforward, d_model, 
                              weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
    

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None, pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(q, k, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(q, k, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, pos, query_pos)


def _convert_param_attr_to_list(param_attr, n):
    """
    If `param_attr` is a list or tuple, convert every element in it to a
    ParamAttr instance. Otherwise, repeat `param_attr` `n` times to
    construct a list, and rename every one by appending a increasing index
    suffix to avoid having same names when `param_attr` contains a name.
    Parameters:
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`.
        n (int): The times to repeat to construct a list when `param_attr`
            is not a list or tuple.
    Returns:
        list: A list composed of each including cell's `param_attr`.
    """
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
        # param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )



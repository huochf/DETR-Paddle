# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Various positional encodings for the transformer.
"""
import math
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from util.misc import NestedTensor


class PositionEmbeddingSine(dg.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passes")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        bs, h, w = mask.shape

        mask = mask.numpy()
        not_mask = ~mask
        not_mask = dg.to_variable(not_mask).astype('float32')
        y_embed = L.cumsum(not_mask, axis=1) # [batch_size, h, w]
        x_embed = L.cumsum(not_mask, axis=2) # [batch_size, h, w]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = (np.arange(0, self.num_pos_feats, 1, dtype="float32")) # [num_pos_feats]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # [num_pos_feats]
        dim_t = dg.to_variable(dim_t)

        x_embed = L.unsqueeze(x_embed, 3) # [batch_size, h, w, 1]
        y_embed = L.unsqueeze(y_embed, 3) # [batch_size, h, w, 1]
        pos_x = x_embed / dim_t           # [batch_size, h, w, num_pos_feats]
        pos_y = y_embed / dim_t           # [batch_size, h, w, num_pos_feats]
        pos_x_1 = L.sin(pos_x[:, :, :, 0::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_x_2 = L.cos(pos_x[:, :, :, 1::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_y_1 = L.sin(pos_y[:, :, :, 0::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_y_2 = L.cos(pos_y[:, :, :, 1::2])  # [batch_size, h, w, num_pos_feats / 2]
        pos_x = L.reshape(L.stack([pos_x_1, pos_x_2], axis=4), (bs, h, w, -1)) # [batch_size, h, w, num_pos_feats]
        pos_y = L.reshape(L.stack([pos_y_1, pos_y_2], axis=4), (bs, h, w, -1)) # [batch_size, h, w, num_pos_feats]

        pos = L.concat((pos_y, pos_x), axis=3)    # [batch_size, h, w, num_pos_feats * 2]
        pos = L.transpose(pos, perm=(0, 3, 1, 2)) # [batch_size, num_pos_feats * 2, h, w]
        return pos


class PositionEmbeddingLearned(dg.Layer):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = dg.Embedding(size=(50, num_pos_feats))
        self.col_embed = dg.Embedding(size=(50, num_pos_feats))
        self.reset_parameters()
    

    def reset_parameters(self):
        self.row_embed.param_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer())
        self.col_embed.param_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer())


    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = L.arange(0, w)
        j = L.arange(0, h)
        x_emb = self.col_embed(i) # [w, num_pos_feats]
        y_emb = self.row_embed(j) # [h, num_pos_feats]
        x_emb = L.expand(L.unsqueeze(x_emb, 0), (h, 1, 1)) # [h, w, num_pos_feats]
        y_emb = L.expand(L.unsqueeze(y_emb, 1), (1, w, 1)) # [h, w, num_pos_feats]
        pos = L.concat([x_emb, y_emb], -1)         # [h, w, num_pos_feats * 2]
        pos = L.transpose(pos, perm=(2, 0, 1))     # [num_pos_feats * 2, h, w]
        pos = L.unsqueeze(pos, 0)                  # [1, num_pos_feats * 2, h, w]
        pos = L.expand(pos, (x.shape[0], 1, 1, 1)) # [batch_size, num_pos_feats * 2, h, w]
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    
    return position_embedding

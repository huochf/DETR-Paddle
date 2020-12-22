
import numpy as np
import argparse
import torch

import paddle.fluid as F

from models.detr import DETR, PostProcess
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine


def convert_param_dict(model_dict):
    renamed_state_dict = {}
    for k, v in model_dict["model"].items():
        name_list = k.split('.')
        if len(name_list) > 2 and name_list[-2][:2] == 'bn':
            if name_list[-1] == "weight":
                ender = "weight"
            elif name_list[-1] == "bias":
                ender = "bias"
            elif name_list[-1] == "running_mean":
                ender = "_mean"
            elif name_list[-1] == "running_var":
                ender = "_variance"
            new_k = name_list[:-1] + ["module", ender]
            renamed_state_dict['.'.join(new_k)] = v.numpy()
        
        elif len(name_list) > 2 and name_list[-2][:6] == 'linear' and name_list[-1] == "weight":
            renamed_state_dict['.'.join(name_list)] = v.numpy().transpose((1, 0))
        
        elif len(name_list) >= 2 and name_list[0][-5:] == "embed" and name_list[0][0] != 'q' \
            and name_list[-1] == "weight":
            renamed_state_dict['.'.join(name_list)] = v.numpy().transpose((1, 0))
            
        elif len(name_list) > 2 and (name_list[-2] == 'self_attn' or name_list[-2] == 'multihead_attn'):
            if name_list[-1][-4:] == "bias":
                q_v, k_v, v_v = np.split(v.numpy(), 3)
                q_k = name_list[:-1] + ["q_proj", "bias"]
                k_k = name_list[:-1] + ["k_proj", "bias"]
                v_k = name_list[:-1] + ["v_proj", "bias"]
                renamed_state_dict['.'.join(q_k)] = q_v
                renamed_state_dict['.'.join(k_k)] = k_v
                renamed_state_dict['.'.join(v_k)] = v_v
            else:
                q_v, k_v, v_v = np.split(v.numpy().transpose((1, 0)), 3, axis = 1)
                # q_v, k_v, v_v = np.split(v.numpy(), 3, axis = 0)
                q_k = name_list[:-1] + ["q_proj", "weight"]
                k_k = name_list[:-1] + ["k_proj", "weight"]
                v_k = name_list[:-1] + ["v_proj", "weight"]
                renamed_state_dict['.'.join(q_k)] = q_v
                renamed_state_dict['.'.join(k_k)] = k_v
                renamed_state_dict['.'.join(v_k)] = v_v
        
        elif len(name_list) > 2 and (name_list[-3] == 'self_attn' or name_list[-3] == 'multihead_attn'):
            if name_list[-1][-4:] == "bias":
                renamed_state_dict['.'.join(name_list)] = v.numpy()
            else:
                renamed_state_dict['.'.join(name_list)] = v.numpy().transpose((1, 0))
        
        elif len(name_list) > 3 and name_list[-3] == 'downsample' and name_list[-2] == '1':
            if name_list[-1] == "weight":
                ender = "weight"
            elif name_list[-1] == "bias":
                ender = "bias"
            elif name_list[-1] == "running_mean":
                ender = "_mean"
            elif name_list[-1] == "running_var":
                ender = "_variance"
            new_k = name_list[:-1] + ["module", ender]
            renamed_state_dict['.'.join(new_k)] = v.numpy()
            
        else:
            renamed_state_dict[k] = v.numpy()

    return renamed_state_dict



def build_paddle_model():
    # * backbone
    backbone = 'resnet50'
    dilation = False
    position_embedding = 'sine'

    # * Transformer
    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 2048
    hidden_dim = 256
    dropout = 0
    nheads = 8
    num_queries = 100
    pre_norm = False
    num_classes = 91

    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone = Backbone(backbone, False, True, dilation)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048
    transformer = Transformer(
        d_model=hidden_dim, dropout=dropout, nhead=nheads, dim_feedforward=dim_feedforward, 
        num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, normalize_before=pre_norm, 
        return_intermediate_dec=True)
    model = DETR(backbone, transformer, num_classes=num_classes, num_queries=num_queries, aux_loss=True)
    
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pytorch model converter', add_help=False)
    parser.add_argument('--model_path', default='../detr-r50-e632da11.pth', type=str,
        help="path to the pytorch pretrained model")
    args = parser.parse_args()
    model_dict = torch.load(args.model_path)

    model = build_paddle_model()
    renamed_state_dict = convert_param_dict(model_dict)
    model.load_dict(renamed_state_dict)
    save_path = args.model_path[:-4]
    F.save_dygraph(model.state_dict(), save_path)
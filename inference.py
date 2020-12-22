import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import PIL
from PIL import Image
import math
import numpy as np
import cv2

import numpy as np
import paddle.fluid as F
import paddle.fluid.dygraph as dg

import datasets.transforms as T 
from models.detr import DETR, PostProcess
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from util.misc import nested_tensor_from_tensor_list


def get_model(args):
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
    
    state_dict, _ = F.load_dygraph( args.pretrained_model)
    model.load_dict(state_dict)

    return model


def get_image(args):
    test_image_raw = Image.open(args.demo_image).convert('RGB')
    transform = T.Compose([
        # T.RandomResize([400], max_size=1333),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    
    ])
    test_image = transform(test_image_raw, target=None)
    test_image = [test_image[0]]
    nested_test_image = nested_tensor_from_tensor_list(test_image)

    return nested_test_image



def plot_results(image, result):
    image = image.numpy()

    image = np.transpose(image, (1, 2, 0))
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    image = (image * std + mean) * 255
    image = image.astype(np.uint8)[:, :, ::-1] # RGB -> BGR
    
    image = image.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128),]
    colors = colors * math.ceil(100 / 12)
    
    label_to_text = json.load(open("/home/aistudio/detr/datasets/coco_category_list.json"))

    for i, item in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        s, l, box = item
        if l == 91 or s < 0.5:
            continue
        # print(l)
        color = colors[i]
        left_top, bottom_down = (box[0], box[1]), (box[2], box[3])
        cv2.rectangle(image, left_top, bottom_down, color, 2)
        label = label_to_text.get(str(l), str(l))
        cv2.putText(image, label + " [" + str(s)[:4] + "]", 
            left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # print(str(l) + " [" + str(s)[:4] + "]")
    
    return image

def postprocess(outputs, nested_test_image, args):
    image_shape = nested_test_image.tensors.shape[2:]
    postprocessor = PostProcess()
    results = postprocessor(outputs, dg.to_variable([image_shape]))

    image = plot_results(nested_test_image.tensors[0], results[0])

    output_path = args.output_dir + '/' + args.demo_image.split('/')[-1]
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR inference', add_help=False)
    parser.add_argument('--pretrained_model', default='./pretrained_models/detr-r50-e632dall.pdparams', type=str,
        help="path to the paddle pretrained model")
    parser.add_argument('--demo_image', default='./demo/COCO_train2014_000000000510.jpg', type=str,
    	help="path to the image.")
    parser.add_argument('--output_dir', default='./outputs/demo', type=str, help="output directory")
    args = parser.parse_args()
    
    with dg.guard():
        model = get_model(args)
        model.eval()
        nested_test_image = get_image(args)
        outputs = model(nested_test_image)
    postprocess(outputs, nested_test_image, args)

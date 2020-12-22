import sys
sys.path.append('..')
import argparse
import numpy as np

import paddle.fluid.dygraph as dg

from models.resnet import resnet50
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.matcher import HungarianMatcher
from models.detr import build
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.argument import get_args_parser

if __name__ == '__main__':
    model = resnet50()
    for k, v in model.state_dict().items():
        print(k + ": " + str(v.shape))
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # print(backbone.state_dict())
    # for k, v in backbone.state_dict().items():
    #     print(k + ": " + str(v.shape))
    # for name, _ in backbone.named_sublayers():
    #     print(name)
    
    # print(backbone.backbone.body)

    with dg.guard():
        backbone = build_backbone(args)
        fake_image = dg.to_variable(np.zeros([4, 3, 512, 512], dtype=np.float32))
        mask = dg.to_variable(np.zeros([4, 512, 512], dtype=np.bool))
        fake_data = NestedTensor(fake_image, mask)

        for k, v in backbone.state_dict().items():
            print(k + ': ' + str(v.shape))

        out, pos = backbone(fake_data)

        for feature_map in out:
            print(feature_map.tensors.shape) # [4, 2048, 16, 16]
            print(feature_map.mask.shape) # [4, 16, 16]

        for pos_tensor in pos:
            print(pos_tensor.shape) # [4, 256, 16, 16]
        
        transformer = build_transformer(args)
        features = dg.to_variable(np.zeros([4, 256, 16, 16], dtype="float32"))
        mask = dg.to_variable(np.zeros([4, 16, 16], dtype="bool"))
        query_embed = dg.to_variable(np.zeros([100, 256], dtype="float32"))
        pos_embed = dg.to_variable(np.zeros([4, 256, 16, 16], dtype="float32"))

        hs, memory = transformer(features, mask, query_embed, pos_embed)
        print(hs.shape) # [6, 4, 100, 256]
        print(memory.shape) # [4, 256, 16, 16]
        
        detr, criterion, postprocessors = build(args)
        out = detr(fake_data)
        for name, tensor in out.items():
            if isinstance(tensor, list):
                print (name)
                print()
                for aux_loss in tensor:
                    for name, tensor in aux_loss.items():
                        print(name)
                        print(tensor.shape)
            else:
                print(name)
                print(tensor.shape)
        # {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4],
        #  "aux_outputs": [
        #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
        #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
        #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
        #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
        #       {"pred_logits": [4, 100, 101], "pred_boxes": [4, 100, 4]},
        # ]}

        # for k, v in detr.state_dict().items():
        #     print(k + ": " + str(v.shape))

        target = [
            {"labels": dg.to_variable(np.zeros([6,], dtype="int64")), 
             "boxes": dg.to_variable(np.zeros([6, 4], dtype="float32"))},
            {"labels": dg.to_variable(np.zeros([3,], dtype="int64")), 
             "boxes": dg.to_variable(np.zeros([3, 4], dtype="float32"))},
            {"labels": dg.to_variable(np.zeros([17,], dtype="int64")), 
             "boxes": dg.to_variable(np.zeros([17, 4], dtype="float32"))},
            {"labels": dg.to_variable(np.zeros([5,], dtype="int64")), 
             "boxes": dg.to_variable(np.zeros([5, 4], dtype="float32"))},
        ]
        
        matcher = HungarianMatcher(1, 1, 1)
        indices = matcher(out, target)
        for ind in indices:
            i_ind, j_ind = ind
            print(i_ind.shape, j_ind.shape)
            # [6] [6]
            # [3] [3]
            # [17] [17]
            # [5] [5]
        
        loss = criterion(out, target)
        for name, loss in loss.items():
            print(name)
            print(loss)








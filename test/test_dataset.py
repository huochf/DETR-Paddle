import sys
sys.path.append("..")
import argparse

from datasets.vrd import build as build_vrd_dataset
from datasets.coco import build as build_coco_dataset
from datasets.vg import build as build_vg_dataset
from util.argument import get_args_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    # dataset = build_coco_dataset("train", args)
    dataset = build_vg_dataset("train", args)

    # print(len(dataset))
    # for i in range(len(dataset)):
    #     image, target = dataset.get_items(i)
    #     print(image.shape)
    #     print(target["boxes"].shape)

    reader = dataset.create_reader()
    for i, item in enumerate(reader()):
        image, target = item
        print(image.shape)
        print(target["boxes"].shape)
    
    batch_reader = dataset.batch_reader(batch_size=4)
    for i, item in enumerate(batch_reader()):
        image, target = item
        print(image[0].shape)
        for t in target:
            print(t["boxes"].shape)
            print(t["labels"].shape)
            assert t["boxes"].shape[-2] == t["labels"].shape[-1]
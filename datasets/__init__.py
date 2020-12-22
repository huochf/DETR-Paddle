from .vrd import build as build_vrd_dataset
from .vg import build as build_vg_dataset
from .coco import build as build_coco_dataset

def build(image_set, args):
    if args.dataset_file == 'vrd':
        return build_vrd_dataset(image_set, args)
    elif args.dataset_file == 'coco':
        return build_coco_dataset(image_set, args)
    elif args.dataset_file == 'vg':
        return build_vg_dataset(image_set, args)
    else:
        raise NotImplementedError
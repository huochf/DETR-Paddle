import os
import json
import numpy as np
from PIL import Image

import paddle
import paddle.fluid.layers as L

import datasets.transforms as T


class COCODetection():

    def __init__(self, img_folder, ann_file, transforms, 
        is_train=True, num_threads=4, buf_size=256):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.is_train = is_train
        self.num_threads = num_threads
        self.buf_size = buf_size

        self._load_image_list()
        self._check_legal()
    

    def create_reader(self):
        def reader():
            for count in range(len(self)):
                yield count

        return paddle.reader.xmap_readers(self.get_items, reader, self.num_threads, self.buf_size)
    

    def batch_reader(self, batch_size):
        reader = self.create_reader()

        def _batch_reader():
            batch_out = []
            for data_list in reader():
                if data_list is None:
                    continue
                
                batch_out.append(data_list)
                if len(batch_out) == batch_size:
                    batch_target = []
                    batch_image = []
                    for item in batch_out:
                        image, target = item
                        batch_image.append(image)
                        batch_target.append(target)

                    yield batch_image, batch_target
                    batch_out = []
        return _batch_reader


    def get_items(self, index):
        file_name, id = self.images_list[index]
        anno_list = self.annotations[id]

        img = Image.open(os.path.join(self.img_folder, file_name)).convert("RGB")
        w, h = img.size
        area = w * h

        labels = []
        boxes = []
        for anno in anno_list:
            bbox, l = anno
            labels.append(l)
            bbox = np.array(self._xywh_to_xyxy(bbox))
            bbox[0::2] = np.clip(bbox[0::2], a_min=0, a_max=w)
            bbox[1::2] = np.clip(bbox[1::2], a_min=0, a_max=h)
            boxes.append(bbox)
        target = {}
        target['labels'] = np.array(labels).astype("int64")
        target['boxes'] = np.array(boxes).astype("float32")
        target["area"] = np.array([area])
        target["size"] = np.array([h, w])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    

    def __len__(self, ):
        return len(self.images_list)
    

    def _load_image_list(self, ):
        with open(self.ann_file) as f:
            anno_dict = json.load(f)

        self.images_list = []
        for image_info in anno_dict['images']:
            self.images_list.append((image_info['file_name'], image_info['id']))
        
        self.annotations = {}
        for anno in anno_dict['annotations']:
            image_id = anno['image_id']
            label = anno['category_id']
            bbox = anno['bbox']
            if image_id in self.annotations:
                self.annotations[image_id].append((bbox, label))
            else:
                self.annotations[image_id] = [(bbox, label)]
        
        self.object_names = json.load(open("/home/aistudio/detr/datasets/coco_category_list.json"))
    

    def _check_legal(self, ):
        bad_image = []
        for name, id in self.images_list:
            if id not in self.annotations:
                bad_image.append((name, id))
        
        for item in bad_image:
            self.images_list.remove(item)
    

    def _xywh_to_xyxy(self, box):
        return [box[0], box[1], box[2] + box[0], box[3] + box[1]]


def make_coco_transforms(image_set):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    scales = [480, 512, 544, 576, 608, 648, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales,  max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize
        ])
    
    if image_set == "val":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    
    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = args.coco_path
    assert os.path.exists(root), f"provided VRD path {root} does not exist"
    PATH = {
        "train": (os.path.join(root, "train2017"),
                  os.path.join(root, "annotations/instances_train2017.json")),
        "val": (os.path.join(root, "val2017"),
                os.path.join(root, "annotations/instances_val2017.json"))
    }
    img_folder, ann_path = PATH[image_set]
    dataset = COCODetection(img_folder, ann_path,
        transforms=make_coco_transforms(image_set),
        is_train=True if image_set == "train" else False)
    return dataset


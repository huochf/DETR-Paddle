import os
import json
import numpy as np
from PIL import Image

import paddle
import paddle.fluid.layers as L

import datasets.transforms as T


class VRDDetection():

    def __init__(self, img_folder, ann_path, transforms, is_train=True, num_threads=4, buf_size=256):
        self.img_folder = img_folder
        self.is_train = is_train
        self.num_threads = num_threads
        self.buf_size = buf_size
        self._load_annotations(ann_path)
        self._transforms = transforms
        self._check_legal()
    

    def create_reader(self, ):
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
        image_name, target = self._get_annotation(index)
        img = Image.open(os.path.join(self.img_folder, image_name)).convert("RGB")
        w, h = img.size
        area = w * h
        target["area"] = np.array([area])
        target["size"] = np.array([h, w])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    

    def __len__(self, ):
        return len(self.image_list)
    

    def _load_annotations(self, ann_path):
        ann_file = os.path.join(ann_path, "annotations_" + ("train.json" if self.is_train else "test.json"))
        with open(ann_file) as f:
            self.annotations_dict = json.load(f)
        self.image_list = []
        for name in self.annotations_dict.keys():
            self.image_list.append(name)
        self.image_list = self.image_list
        
        self.object_names = json.load(open(os.path.join(ann_path, "objects.json")))
        self.predicate_names = json.load(open(os.path.join(ann_path, "predicates.json")))

    
    def _get_annotation(self, idx):
        image_name = self.image_list[idx]
        triple_list = self.annotations_dict[image_name]
        labels = []
        boxes = []
        for item in triple_list:
            if self._yyxx_to_xyxy(item["subject"]["bbox"]) not in boxes:
                boxes.append(self._yyxx_to_xyxy(item["subject"]["bbox"]))
                labels.append(int(item["subject"]["category"]))
            if self._yyxx_to_xyxy(item["object"]["bbox"]) not in boxes:
                boxes.append(self._yyxx_to_xyxy(item["object"]["bbox"]))
                labels.append(int(item["object"]["category"]))
        target = {"labels": np.array(labels).astype("int64"),
                  "boxes": np.array(boxes).astype("float32")}
        return image_name, target
    

    def _check_legal(self, ):
        bad_data = []
        for i, image_name in enumerate(self.image_list):
            _, target = self._get_annotation(i)
            if len(target["boxes"]) == 0:
                bad_data.append(image_name)
        
        for bad_image in bad_data:
            self.image_list.remove(bad_image)
    

    def _yyxx_to_xyxy(self, box):
        return [box[2], box[0], box[3], box[1]]


def make_vrd_transforms(image_set):
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
    root = args.vrd_path
    assert os.path.exists(root), f"provided VRD path {root} does not exist"
    PATH = {
        "train": (os.path.join(root, "sg_dataset/sg_train_images"),
                  os.path.join(root, )),
        "val": (os.path.join(root, "sg_dataset/sg_test_images"),
                os.path.join(root,))
    }
    img_folder, ann_path = PATH[image_set]
    dataset = VRDDetection(img_folder, ann_path,
        transforms=make_vrd_transforms(image_set),
        is_train=True if image_set == "train" else False)
    return dataset

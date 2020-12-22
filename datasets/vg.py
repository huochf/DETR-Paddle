import os
import json
import numpy as np
from PIL import Image

import paddle
import paddle.fluid.layers as L

import datasets.transforms as T


class VisualGenomeDetection():

    def __init__(self, img_folder, image_list_file, ann_file, object_list, transforms, 
        is_train=True, num_threads=4, buf_size=256):
        self.img_folder = img_folder
        self.image_list_file = image_list_file
        self.ann_file = ann_file
        self.object_list = object_list
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
        id, file_name = self.images_list[index]
        anno_list = self.annotations[id]

        img = Image.open(os.path.join(self.img_folder, file_name)).convert("RGB")
        w, h = img.size
        area = w * h

        bbox, labels = anno_list
        boxes = []
        for b in bbox:
            b = np.array(self._xywh_to_xyxy(b))
            b[0::2] = np.clip(b[0::2], a_min=0, a_max=w)
            b[1::2] = np.clip(b[1::2], a_min=0, a_max=h)
            boxes.append(b)
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
        with open(self.object_list) as f:
            object_list = json.load(f)
        
        object_name_to_label = {}
        for l, n in enumerate(object_list):
            object_name_to_label[n] = l

        with open(self.image_list_file) as f:
            image_list = json.load(f)
        
        with open(self.ann_file) as f:
            ann_list = json.load(f)

        self.images_list = []
        idx = 0
        for image_info in image_list:
            image_url = image_info['url']
            image_path = '/'.join(image_url.split('/')[-2:])
            self.images_list.append((idx, image_path))
            idx += 1
        
        idx = 0
        self.annotations = {}
        self.name_to_id = {}
        for annos in ann_list:
            labels = []
            bbox = []
            for anno in annos['objects']:
                if anno['synsets'] == []:
                    continue
                
                name = anno['synsets'][0].split('.')[0]

                if name not in object_list:
                    continue

                label = object_name_to_label[name]
                labels.append(label)

                bbox.append([anno['x'], anno['y'], anno['w'], anno['h']])
            self.annotations[idx] = (bbox, labels)
            idx += 1
        
        self.object_names = object_list
                

    def _check_legal(self, ):
        bad_image = []
        for idx, image_name in self.images_list:
            if not os.path.exists(os.path.join(self.img_folder, image_name)):
                bad_image.append((idx, image_name))
            
            elif len(self.annotations[idx][0]) < 2:
                bad_image.append((idx, image_name))
                
        
        for item in bad_image:
            # print("remove " + item[1] + " from image lists")
            self.images_list.remove(item)
    

    def _xywh_to_xyxy(self, box):
        return [box[0], box[1], box[2] + box[0], box[3] + box[1]]


def make_VG_transforms(image_set):
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
    root = args.vg_path
    assert os.path.exists(root), f"provided Visual Genome path {root} does not exist"
    PATH = {
        "train": (os.path.join(root, "full_images"),
                  os.path.join(root, "v1.0/image_data.json"),
                  os.path.join(root, "v1.4/objects.json"),
                  os.path.join(root, 'vg_object_list.json')),
        "val": (os.path.join(root, "full_images"),
                  os.path.join(root, "v1.0/image_data.json"),
                  os.path.join(root, "v1.4/objects.json"),
                  os.path.join(root, 'vg_object_list.json'))
    }
    img_folder, image_list, ann_path, object_list = PATH[image_set]
    dataset = VisualGenomeDetection(img_folder, image_list, ann_path, object_list,
        transforms=make_VG_transforms(image_set),
        is_train=True if image_set == "train" else False)
    return dataset


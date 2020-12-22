# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Transforms and data augmentation for both image + bbox.
"""
import numpy as np
import PIL
from PIL import Image

import paddle.fluid.dygraph as dg

from util.box_ops import box_xyxy_to_cxcywh

def crop(image, target, region):
    target = target.copy()
    i, j, h, w = region

    cropped_image = image.crop((j, i, j + w, i + h))

    # should we do somthing wrt the original size?
    target["size"] = np.array([h, w])

    fields = ["labels", "area"]
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = np.array([w, h])
        cropped_boxes = boxes - np.array([j, i, j, i])
        cropped_boxes = cropped_boxes.reshape((-1, 2, 2))
        cropped_boxes[cropped_boxes[:, :, 0] > w] = w
        cropped_boxes[cropped_boxes[:, :, 1] > h] = h
        cropped_boxes = cropped_boxes.clip(min=0, max=1e8)
        wh = cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]
        area = wh[:, 0] * wh[:, 1]
        target["boxes"] = cropped_boxes.reshape((-1, 4))
        target["area"] = area
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        cropped_boxes = target["boxes"].reshape((-1, 2, 2))
        keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)

        for field in fields:
            target[field] = target[field][keep]
        
        if len(target["boxes"]) == 0:
            target["boxes"] = np.zeros([1, 4]).astype("float32")
            target["area"] = np.zeros([1]).astype("float32")
            target["labels"] = np.zeros([1]).astype("int64")

    return cropped_image, target


def hflip(image, target):
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1])
        boxes = boxes + np.array([w, 0, w, 0])
        target["boxes"] = boxes
    
    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuplle

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)
        
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        
        return (ow, oh)
    
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    size = get_size(image.size, size, max_size)
    rescaled_image = image.resize(size) # [w, h]

    if target is None:
        return rescaled_image, None
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    
    w, h = size
    target["size"] = np.array([h, w])

    return rescaled_image, target


def pad(image, target, padding):
    # assume that we only pad on the bottom right corners
    w, h = image.size
    padded_image = Image.new('RGB', (w + padding[0], h + padding[1]), (0, 0, 0))
    padded_image.paste(image, (0, 0, w, h))

    if target is None:
        return padded_image, None
    
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = np.random(padded_image.size)
    return padded_image, target


class RandomCrop(object):

    def __init__(self, size):
        self.size = size
    

    def __call__(self, img, target):
        w, h = img.size
        th, tw = self.size

        if th == h and tw == w:
            region = (0, 0, h, w)
        else:
            i = np.random.randint(0, h - th)
            j = np.random.randint(0, w - tw)
            region = (i, j, th, tw)
        return crop(img, target, region)


class RandomSizeCrop(object):

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
    

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = np.random.randint(self.min_size, min(img.width, self.max_size))
        h = np.random.randint(self.min_size, min(img.height, self.max_size))

        w_orig, h_orig = img.size
        if w == w_orig and h == h_orig:
            region = (0, 0, h, w)
        else:
            i = np.random.randint(0, h_orig - h)
            j = np.random.randint(0, w_orig - w)
            region = (i, j, h, w)
            
        return crop(img, target, region)


class CenterCrop(object):

    def __init__(self, size):
        self.size = size
    

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
    
    
    def __call__(self, img, target):
        if np.random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):

    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
    

    def __call__(self, img, target=None):
        size = np.random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):

    def __init__(self, max_pad):
        self.max_pad = max_pad
    

    def __call__(self, img, target):
        pad_x = np.random.randint(0, self.max_pad)
        pad_y = np.random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    

    def __call__(self, img, target):
        if np.random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):

    def __call__(self, img, target):
        for k in target.keys():
            target[k] = dg.to_variable(target[k])
        img = np.array(img).astype("float32")
        img = img.tranapose((2, 0, 1))
        img = dg.to_variable(img)
        return img, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, image, target=None):
        image = np.array(image).astype("float32")
        image = image / 255.
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))
        image = dg.to_variable(image)
        if target is None:
            return image, None
        
        for k in target.keys():
            target[k] = dg.to_variable(target[k])
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes.numpy() / np.array([w, h, w, h]).astype("float32")
            boxes = dg.to_variable(boxes)
            target["boxes"] = boxes
        
        return image, target


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string = "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Utilities for bounding box manipulation and GIoU.
"""
import numpy as np
import paddle.tensor as T
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = L.split(x, 4, -1) # [num_boxes, 4] -> [num_boxes, 1] ...
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return L.squeeze(L.stack(b, axis=-1), [1])


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = L.split(x, 4, -1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return L.squeeze(L.stack(b, axis=-1), [1])


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    # from (https://pytorch.org/docs/stable/_modules/torchvision/ops/boxes.html#box_area)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-4


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1) # [N]
    area2 = box_area(boxes2) # [M]
    N, M = boxes1.shape[0], boxes2.shape[0]
    boxes1 = L.unsqueeze(boxes1, axes=[1]) # [N, 1, 4]
    boxes1 = L.expand(boxes1, [1, M, 1]) # [N, M, 4]
    boxes2 = L.unsqueeze(boxes2, axes=[0]) # [1, M, 4]
    boxes2 = L.expand(boxes2, [N, 1, 1]) # [N, M, 4] 
    lt = L.elementwise_max(boxes1[:, :, :2], boxes2[:, :, :2]) # [N, M, 2]
    rb = L.elementwise_min(boxes1[:, :, 2:], boxes2[:, :, 2:]) # [N, M, 2]

    wh = L.clip(rb - lt, min=0, max=1e8) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    area1 = L.expand(L.unsqueeze(area1, [1]), [1, M]) # [N, M]
    area2 = L.expand(L.unsqueeze(area2, [0]), [N, 1]) # [N, M]
    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalied_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert L.reduce_all(boxes1[:, 2:] >= boxes1[:, :2])
    assert L.reduce_all(boxes2[:, 2:] >= boxes2[:, :2])
    iou, union = box_iou(boxes1, boxes2)

    N, M = boxes1.shape[0], boxes2.shape[0]
    boxes1 = L.unsqueeze(boxes1, axes=[1]) # [N, 1, 4]
    boxes1 = L.expand(boxes1, [1, M, 1]) # [N, M, 4]
    boxes2 = L.unsqueeze(boxes2, axes=[0]) # [1, M, 4]
    boxes2 = L.expand(boxes2, [N, 1, 1]) # [N, M, 4] 
    lt = L.elementwise_min(boxes1[:, :, :2], boxes2[:, :, :2]) # [N, M, 2]
    rb = L.elementwise_max(boxes1[:, :, 2:], boxes2[:, :, 2:]) # [N, M, 2]

    wh = L.clip(rb - lt, min=0, max=1e8) # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1] + 1e-4 # prevent devided by zero

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """
    Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number
    of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if np.sum(masks.shape) == 0:
        return dg.to_variable(np.zeros((0, 4)))
    
    h, w = masks.shape[-2:]
    y = dg.to_variable(np.arange(0, h, 1, dtype="float32"))
    x = dg.to_variable(np.arange(0, w, 1, dtype="float32"))
    y, x = T.meshgrid([y, x]) # [h, w]

    x_mask = (masks * L.unsqueeze(x, [0])) # [N, H, W]
    x_max = L.reduce_max(L.flatten(x_mask, axis=1), dim=-1)
    non_mask = dg.to_variable(~masks.numpy())
    x_mask[non_mask] = 1e8
    x_min = L.reduce_min(L.flatten(x_mask, axis=1), dim=-1)

    y_mask = (masks * L.unsqueeze(y, [0])) # [N, H, W]
    y_max = L.reduce_max(L.flatten(y_mask, axis=1), dim=-1)
    y_mask[non_mask] = 1e8
    y_min = L.reduce_min(L.flatten(y_mask, axis=1), dim=-1)

    return L.stack([x_min, y_min, x_max, y_max], 1)

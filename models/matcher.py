# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from scipy.optimize import linear_sum_assignment

import paddle.fluid.layers as L 
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalied_box_iou


class HungarianMatcher(dg.Layer):
    """
    This class computes an assigment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 erro of the bounding box coordinates in the matching
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
    

    def forward(self, outputs, targets):
        """
        Performs the matching

        Params:
            outputs: This is a dict contains at least these entries:
                "pred_logits": Tensor of dim[batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicated box coordinates
            
            targets: This is a list of targets (len(targets) == batch_size), where each target is a dict containing:
                "labels": Tensor of dim[num_target_boxes] (where num_target_boxes is the number of ground-truth)
                          objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordiantes
        
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with dg.no_grad():
            bs, num_queries, num_classes = outputs["pred_logits"].shape

            # We flatten to compute the cost matrices in a batch
            out_prob = L.reshape(outputs["pred_logits"], [-1, num_classes]) # [batch_size * num_queries, num_classes]
            out_prob = L.softmax(out_prob, axis=-1) # [batch_size * num_queries, num_classes]
            out_bbox = L.reshape(outputs["pred_boxes"], [-1, 4]) # [batch_size * num_queries, 4]

            # Alse concat the target labels and boxes 
            tgt_ids = L.concat([v["labels"] for v in targets]).astype("int64") # [batch_size * num_target_boxes_i]
            tgt_bbox = L.concat([v["boxes"] for v in targets]).astype("float32") # [batch_size * num_target_boxes_i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that donesn't change the matching, it can be ommitted.
            cost_class = -out_prob.numpy()[:, tgt_ids.numpy()] # [batch_size * num_queries, num_all_target_boxes]
            cost_class = dg.to_variable(cost_class)

            # Compute the L1 cost between boxes
            num_all_target_boxes = tgt_bbox.shape[0]
            expanded_out_bbox = L.expand(L.unsqueeze(out_bbox, [1]), [1, num_all_target_boxes, 1]) # [batch_size * num_queries, num_all_target_boxes, 4]
            expanded_tgt_bbox = L.expand(L.unsqueeze(tgt_bbox, [0]), [bs * num_queries, 1, 1])     # [batch_size * num_queries, num_all_target_boxes, 4]
            cost_bbox = F.loss.l1_loss(expanded_out_bbox, expanded_tgt_bbox, reduction='none') # [batch_size * num_queries, num_all_target_boxes, 4]
            cost_bbox = L.reduce_mean(cost_bbox, -1) # [batch_size * num_queries, num_all_target_boxes]

            # Compute the giou cost between boxes
            cost_giou = - generalied_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = L.reshape(C, [bs, num_queries, -1]) # [batch_size, num_queries, num_all_target_boxes]

            sizes = [len(v["boxes"]) for v in targets]
           
            indices = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(L.split(C, sizes, dim=-1))]

            return [(dg.to_variable(i.astype("int64")), dg.to_variable(j.astype("int64")))
                    for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR model and criterion classes.
"""
import numpy as np
import paddle.tensor as T
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F

from .backbone import build_backbone
from .visualize_transformer import build_transformer
from .matcher import build_matcher
from util import box_ops
from util.misc import NestedTensor, nested_tensor_from_tensor_list


class DETR(dg.Layer):
    """ This is the DETR module that performs object detection """
    
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """
        Initializes the model.

        Parameters:
            backbone: See backbone.py
            transformer: See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie the detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = dg.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = dg.Embedding((num_queries, hidden_dim))
        self.input_proj = dg.Conv2D(backbone.num_channels, hidden_dim, filter_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
    

    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        
        It returns a dict with following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                             Shape = [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as 
                            (center_x, center_y, height, width). There values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrive the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                             dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, fluid.Variable)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        hs, memory, encoder_attn_weights, decoder_attn_weights = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = L.sigmoid(self.bbox_embed(hs))
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out, encoder_attn_weights, decoder_attn_weights, src.shape[2:]
    

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(dg.Layer):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special on-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.eos_coef = eos_coef
        empty_weight = L.ones([self.num_classes + 1], dtype="float32")
        empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight
        # self.add_parameter("empty_weight", empty_weight)
    

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL)
        targets dict must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        num_classes_plus_1 = outputs["pred_logits"].shape[-1]
        src_logits = outputs["pred_logits"] # [bs, num_queries, num_classes]
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = [t["labels"].numpy()[J.numpy()] for t, (_, J) in zip(targets, indices)]
        target_classes_o = [dg.to_variable(t) for t in target_classes_o]
        target_classes_o = L.concat(target_classes_o) # [bs * num_object]
        target_classes = T.creation.full(src_logits.shape[:2], self.num_classes).astype("int64") # [bs, num_queries]
        
        idx = np.array([idx[0].numpy(), idx[1].numpy()])
        target_classes = target_classes.numpy()
        target_classes[idx[0], idx[1]] = target_classes_o.numpy()
        target_classes = dg.to_variable(target_classes)

        target_classes = L.unsqueeze(target_classes, axes=[2])
        loss_ce = L.softmax_with_cross_entropy(src_logits, target_classes) # (bs, num_queries, 1)
        loss_weight = np.ones(loss_ce.shape).astype("float32")
        loss_weight[(target_classes == self.num_classes).numpy()] = self.eos_coef
        loss_ce = loss_ce * dg.to_variable(loss_weight)
        loss_ce = L.reduce_mean(loss_ce)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            out_logits = src_logits.numpy()[idx[0], idx[1], :]
            out_logits = dg.to_variable(out_logits) # [num_objects, num_classes_plus_1]
            target_labels = L.reshape(target_classes_o, (-1, 1))
            losses['class_error'] = 100 - 100 * L.accuracy(out_logits, target_labels)
        return losses
    

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        with dg.no_grad():
            pred_logits = outputs["pred_logits"] # [bs, num_queries, num_classes]
            tgt_lengths = dg.to_variable([len(v["labels"]) for v in targets]).astype("float32")
            # Count the number of predictions that are NOT "no-object" (which is the last class)
            card_pred = L.reduce_sum((L.argmax(pred_logits, -1) != pred_logits.shape[-1] - 1).astype("float32"))
            card_err = F.loss.l1_loss(card_pred, tgt_lengths)
            losses = {"cardinality_error": card_err}
            return losses
    

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"].numpy()[idx[0].numpy(), idx[1].numpy(), :] # [num_objects, 4]
        src_boxes = dg.to_variable(src_boxes)

        target_boxes = [t["boxes"].numpy()[i.numpy()] for t, (_, i) in zip(targets, indices)]
        target_boxes = [dg.to_variable(t) for t in target_boxes]
        target_boxes = L.concat(target_boxes, 0).astype("float32") # [num_objects, 4]
        loss_bbox = F.loss.l1_loss(src_boxes, target_boxes, reduction="sum") 

        losses = {}
        losses["loss_bbox"] = loss_bbox / num_boxes

        num_boxes = src_boxes.shape[0]
        mask = T.creation.diag(dg.to_variable(np.ones(num_boxes))) # mask out non-diag element
        loss_giou = (1 - box_ops.generalied_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        )) * mask
        losses["loss_giou"] = L.reduce_sum(loss_giou) / num_boxes
        return losses
    

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # assert "pred_masks" in outputs

        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks = outputs["pred_masks"]
        # src_masks = src_masks[src_idx] # []
        pass

    
    def _get_src_permutation_idx(self, indices):
        # permute prediction following indices
        batch_idx = L.concat([T.creation.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = L.concat([src for (src, _) in indices]) # [num_target]
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute target following indices
        batch_idx = L.concat([T.creation.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        src_idx = L.concat([tgt for (_, tgt) in indices]) # [num_target]
        return batch_idx, src_idx
    

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks
        }
        assert "masks" != loss, "not implement for mask loss"
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss'doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = dg.to_variable([num_boxes])

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue 
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


class PostProcess(dg.Layer):
    """
    This module converts the model's output into the format expected by the coco api
    """
    def forward(self, outputs, target_sizes):
        """
        Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each image
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = L.softmax(out_logits, -1)            # [bs, num_queries, num_classes + 1]
        labels = L.argmax(prob[:, :, :], axis=-1) # [bs, num_queries]
        scores = L.reduce_max(prob, dim=-1)         # [bs, num_queries]

        # convert to [x0, y0, x1, y1] format
        bs, num_queries, _ = out_bbox.shape
        out_bbox = L.reshape(out_bbox, (-1, 4))
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = L.reshape(boxes, (bs, num_queries, 4))
        # and fromm relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale_fct = L.stack([img_w, img_h, img_w, img_h], 1) # [bs, 4]
        scale_fct = L.expand(L.unsqueeze(scale_fct, [1]), (1, num_queries, 1))
        boxes = boxes * scale_fct

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores.numpy(), labels.numpy(), boxes.numpy())]

        return results      


class MLP(dg.Layer):
    """ very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = dg.LayerList(dg.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = L.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id` + 1, where
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, se we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1). 
    # For more details on this, check the following discussion
    # http://github.com/facebookresearch/detr/issues/108#issuecomment-650269233
    if args.dataset_file == "vrd":
        num_classes = 101
    elif args.dataset_file == 'coco':
        num_classes = 91
    elif args.dataset_file == "coco_panoptic":
        # For panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(backbone, transformer, num_classes=num_classes,
                 num_queries=args.num_queries, aux_loss=args.aux_loss)
    
    if args.masks:
        raise NotImplementedError()
    
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=args.eos_coef, losses=losses)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

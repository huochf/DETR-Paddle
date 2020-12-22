# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval function used in main.py
"""
import math
import os
import sys
from typing import Iterable
import paddle.fluid.dygraph as dg
import paddle.fluid as F

import util.misc as utils


def train_one_epoch(model, criterion, dataset, optimizer, visualizer, epoch, max_norm, args):
    model.train()
    
    metric_logger = utils.MetricLogger(args, delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    visualize_freq = 100 * print_freq
    count = 0

    for samples, targets in metric_logger.log_every(dataset, print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        losses = losses / args.batch_size

        if not math.isfinite(losses.numpy()):
            print("Loss is {}, stopping training".format(losses.numpy()))
            print(loss_dict)
            sys.exit(1)

        losses.backward()

        optimizer.minimize(losses)
        optimizer.clear_gradients()

        metric_logger.update(loss=losses.numpy(), **loss_dict)
        metric_logger.update(class_error=loss_dict["class_error"])
        metric_logger.update(lr=optimizer.current_step_lr())

        count += 1
        if visualize_freq % count == 0:
            visualizer.plot_results(samples, outputs, targets)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        
def evaluate(model, criterion, dataset, visualizer, output_dir, args):
    with dg.no_grad():
        model.eval()

        metric_logger = utils.MetricLogger(args, delimiter=" ")
        metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
        header = "Test"
        print_freq = 10
        visualize_freq = 100 * print_freq
        count = 0

        for samples, targets in metric_logger.log_every(dataset, print_freq, header):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            losses = losses / args.batch_size

            metric_logger.update(loss=losses.numpy(), **loss_dict)
            metric_logger.update(class_error=loss_dict["class_error"])

            count += 1
            if visualize_freq % count == 0:
                visualizer.plot_results(samples, outputs, targets)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import paddle.fluid as F
import paddle.fluid.dygraph as dg

from datasets import build as build_dataset
from models.detr import build as build_model
from util.argument import get_args_parser
from util.visualize import Visualizer
from engine import train_one_epoch, evaluate
import util.misc as utils


def main(args):
    print(args)

    with dg.guard():
        model, criterion, postprocessors = build_model(args)
        print("building dataset")
        dataset_train = build_dataset(image_set="train", args=args)
        dataset_val = build_dataset(image_set="val", args=args)

        clip = F.clip.GradientClipByValue(max=args.clip_max_norm)
        optimizer = F.optimizer.AdamOptimizer(parameter_list=model.parameters(), 
            learning_rate=args.lr, grad_clip=clip)
        
        dataset_train_reader = dataset_train.batch_reader(args.batch_size)
        dataset_val_reader = dataset_val.batch_reader(args.batch_size)

        output_dir = os.path.join(args.output_dir, args.dataset_file)

        if args.resume:
            print("Loading pretrained model from path: " + args.resume)
            state_dict, _ = F.load_dygraph(args.resume)
            model.load_dict(state_dict)
        
        visualizer = Visualizer(postprocessors["bbox"], output_dir, dataset_train.object_names, args)

        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(model, criterion, dataset_train, 
                optimizer, visualizer, epoch, args.clip_max_norm, args)
            
            if args.output_dir:
                checkpoint_paths = [os.path.join(output_dir, 'checkpoint')]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(os.path.join(output_dir, f"checkpoint{epoch:04}"))
                for checkpoint_path in checkpoint_paths:
                    F.save_dygraph(model.state_dict(), checkpoint_path)
            
            test_stats = evaluate(model, criterion, dataset_val, visualizer, args.output_dir, args)
            
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                         **{f"test_{k}": v for k, v in test_stats.items()},
                         "epoch": epoch,
                        }
                
            if args.output_dir:
                with open(os.path.join(output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time{}".format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


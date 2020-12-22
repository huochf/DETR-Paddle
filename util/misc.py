# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import time
import datetime
import numpy as np
from collections import defaultdict, deque
from typing import Optional, List
import paddle.fluid as F
import paddle.fluid.dygraph as dg

class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.3f}({global_avg:.3f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n


    @property
    def median(self):
        d = np.array(list(self.deque)).astype("float32")
        return np.median(d)
    

    @property
    def avg(self):
        d = np.array(list(self.deque)).astype("float32")
        return d.mean()
    

    @property
    def global_avg(self,):
        return self.total / self.count
    

    @property
    def max(self):
        return max(self.deque)
    

    @property
    def value(self):
        return self.deque[-1]
    

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):

    def __init__(self, args, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.args = args
    

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, F.Variable):
                v = float(v.numpy()[0])
            if isinstance(v, np.ndarray):
                v = float(v[0])
            v = float(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr
        ))
    

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    

    def get_print_string(self, keys):
        loss_str = []
        for name, meter in self.meters.items():
            if name in keys:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)
    

    def add_meter(self, name, meter):
        self.meters[name] = meter
    

    def log_every(self, dataset, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':4d'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
        ])
        data_reader = dataset.batch_reader(self.args.batch_size)
        for i, obj in enumerate(data_reader()):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(dataset) - 1:
                eta_seconds = iter_time.global_avg * (len(dataset) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(log_msg.format(
                    i, len(dataset) // self.args.batch_size, eta=eta_string,
                    meters=self.get_print_string(["class_error", "loss", "loss_ce", "loss_bbox", "loss_giou"]),
                    time=str(iter_time), data=str(data_time)
                ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(dataset)
        ))


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    
    def __init__(self, tensors, mask: Optional[F.Variable]):
        self.tensors = tensors
        self.mask = mask
    

    def decompose(self):
        return self.tensors, self.mask
    

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[F.Variable]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        tensor = dg.to_variable(np.zeros(batch_shape, dtype="float32"))
        mask = dg.to_variable(np.ones((b, h, w), dtype="int64"))
        for i in range(b):
            img = tensor_list[i]
            tensor[i,: img.shape[0], : img.shape[1], : img.shape[2]] = img
            mask[i, : img.shape[1], : img.shape[2]] = 0

        mask = mask.astype("bool")
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


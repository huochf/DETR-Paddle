# DETR: An End-to-end Object DEtector using TRansformer

### About the code
Paddle training code and pretrained models for **DETR** (**DE**tection **TR**ansformer). This code is modified from original project to make it compatible with **Paddle**.

**Original project: [DETR(pytorch)](https://github.com/facebookresearch/detr)**

**Paper: [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)**

**[Aistudio Project](https://aistudio.baidu.com/aistudio/projectdetail/1327221?channelType=0&channel=0)**

If this work is useful to you, please cite:

@INPROCEEDINGS{DETR,\
title={End-to-End Object Detection with Transformers}, \
author={Nicolas Carion and Francisco Massa and Gabriel Synnaeve and Nicolas Usunier and Alexander Kirillov and Sergey Zagoruyko},\
year={2020},\
booktitle={ECCV}\
}

### Contributions:

- This project provided DETR training code in Paddle.
- This project provided pretrained models converted from Pytorch version.
- This project analysis and visualize the model outputs.

### Model Zoo

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
      <th>paddle model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
      <td><a href="https://aistudio.baidu.com/aistudio/datasetdetail/65656">detr.pdparams</a><td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
      <td><a href="https://aistudio.baidu.com/aistudio/datasetdetail/65656">detr.pdparams</a><td>
    </tr>
  </tbody>
</table>


## Train - Object detection

<font color=red>Warning: this code need to optimize, model is very memory-comsuming!!!</font>

#### Train **[COCO 2017](https://aistudio.baidu.com/aistudio/datasetdetail/7122)**

1. unzip dataset, run `cd /home/aistudio/data/data7122/ && unzip train2017.zip && unzip val2017.zip && unzip annotations_trainval2017.zip`.

2. run ` cd ../../detr && python ./main.py --dataset_file coco`

#### Train **[Visual Genome](https://aistudio.baidu.com/aistudio/datasetdetail/57396)**

1. unzip dataset, ` cd /home/aistudio/data/data57396 && zip -s 0 full_images.zip --out image.zip && unzip image.zip && unzip v1.0.zip`
2. run ` cd ../../detr && python ./main.py --dataset_file vg`

#### Train **[Visual Relationship Detection](https://aistudio.baidu.com/aistudio/datasetdetail/57355)**

1. 
```
! cd /home/aistudio/data/data57355/ && unzip json_dataset.zip
! clear
! cd /home/aistudio/data/data57355/ && unzip sg_dataset.zip
! clear
! cd /home/aistudio/data/data57355/sg_dataset/sg_test_images/ && mv ./4392556686_44d71ff5a0_o.* ./4392556686_44d71ff5a0_o.jpg
```
2. ` cd ../../detr && python ./main.py --dataset_file vrd`

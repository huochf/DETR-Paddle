import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

import paddle.fluid.layers as L

from util.box_ops import box_cxcywh_to_xyxy


class Visualizer:

    def __init__(self, postprocessor, output_dir, label_to_text, args):
        self.postprocessor = postprocessor
        self.output_dir = output_dir
        self.label_to_text = label_to_text
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.pool_size = 30
        self.index = 0

        if args.dataset_file == 'coco':
            self.background = 91
        elif args.dataset_file == 'vrd':
            self.background = 101
        elif args.dataset_file == 'vg':
            self.background = 101
    

    def plot_results(self, samples, outputs, targets):
        # samples: [batch_size, 3, H, W]
        samples = [sample.numpy() for sample in samples]

        target_sizes = L.stack([t["size"] for t in targets], 0)

        results = self.postprocessor(outputs, target_sizes)

        for i, item in enumerate(zip(samples, results, targets)):
            image, result, target = item
            image = np.transpose(image, (1, 2, 0))
            std = np.array([0.229, 0.224, 0.225])
            mean = np.array([0.485, 0.456, 0.406])
            image = (image * std + mean) * 255
            image = image.astype(np.uint8)[:, :, ::-1] # RGB -> BGR
            targ_image = image.copy()
            pred_img = image.copy()

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                      (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128),]
            rect_num = len(target["boxes"])
            colors = colors * math.ceil(rect_num / 12)

            h, w = target["size"].numpy()
            for i, item in enumerate(zip(target["labels"], target["boxes"])):
                l, box = item
                color = colors[i]
                box = L.unsqueeze(box, [0])
                box = box_cxcywh_to_xyxy(box) # [1, 4]
                box = L.squeeze(box, [0]) # [4]
                box = (box.numpy() * np.array([w, h, w, h])).astype(np.int)
                left_top, bottom_down = (box[0], box[1]), (box[2], box[3])
                cv2.rectangle(targ_image, left_top, bottom_down, color, 2)
                l = l.numpy()[0]
                if isinstance(self.label_to_text, dict):
                    label_name = self.label_to_text.get(str(l), str(l))
                else:
                    if l < len(self.label_to_text):
                        label_name = self.label_to_text[l]
                    else:
                        label_name = str(l)

                cv2.putText(targ_image, label_name, left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            rect_num = len(result["labels"])
            colors = colors * math.ceil(rect_num / 12)
            for i, item in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
                s, l, box = item

                if l == self.background:
                    continue

                color = colors[i]
                left_top, bottom_down = (box[0], box[1]), (box[2], box[3])
                cv2.rectangle(pred_img, left_top, bottom_down, color, 2)
                
                if isinstance(self.label_to_text, dict):
                    label_name = self.label_to_text.get(str(l), str(l))
                else:
                    if l < len(self.label_to_text):
                        label_name = self.label_to_text[l]
                    else:
                        label_name = str(l)

                cv2.putText(pred_img, label_name + " [" + str(s)[:4] + "]", 
                    left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            show_image = np.concatenate((targ_image, pred_img), 1)
            cv2.imwrite(os.path.join(self.output_dir, str(self.index) + ".jpg"), show_image)
            self.index = (self.index + 1) % self.pool_size



def show_heatmap_image(reference_point, encoder_attn_weights, heatmap_size, test_image_raw, save_path=None):
    x, y = reference_point
    index = y * heatmap_size[1] + x

    h, w = test_image_raw.size
    image = np.array(test_image_raw.copy())[:, :, ::-1]
    alpha = 0.5

    heatmap = encoder_attn_weights[5][0][7][index].numpy().reshape(heatmap_size)
    heatmap = heatmap / heatmap.max() * 255
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (h, w))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heat_image = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    # heat_image = heatmap * 0.4 + image
    cv2.circle(heat_image, (x * 32, y * 32), 4, [0, 255, 255], 8)
    plt.rcParams['figure.dpi'] = 150
    plt.axis("off")
    plt.imshow(heat_image[:, :, ::-1])
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()


def show_encoder_heapmap(encoder_attn_weights, fig_size=(8, 6), save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    count = 1
    for l in range(6):
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        for h in range(8):
            plt.subplot(6, 8, count)
            plt.axis("off")
            plt.imshow(encoder_attn_weights[l][0][h].numpy())
            count += 1
            plt.title('layer: %d, multihead: %d' % (l + 1, h + 1), fontsize=4, pad=3)
    plt.suptitle('Encoder Attention Heapmap')
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()


def visualize_encoder_heapmap_by_reference_points_multihead(encoder_attn_weights, reference_points, layer, 
    fig_size, heatmap_size, save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    points_list = reference_points
    points_num = len(points_list)
    count = 1
    for i in range(points_num):
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        x, y = points_list[i]
        index = y * heatmap_size[1] + x
        for j in range(8):
            plt.subplot(points_num, 8, count)
            plt.axis("off")
            plt.scatter(x, y, color="red")
            plt.imshow(encoder_attn_weights[layer][0][j][index].numpy().reshape(heatmap_size))
            count += 1
            plt.title('reference point: (%d, %d), multihead: %d' % (x, y, j + 1), fontsize=3.5, pad=3)
    plt.suptitle('Encoder Attention Heapmap in Layer %d' % (layer + 1))
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()


def visualize_encoder_heapmap_by_reference_points_layer(encoder_attn_weights, reference_points, multihead, 
    fig_size, heatmap_size, save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    points_list = reference_points
    points_num = len(points_list)
    count = 1
    for i in range(points_num):
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        x, y = points_list[i]
        index = y * heatmap_size[1] + x
        for j in range(6):
            plt.subplot(points_num, 6, count)
            plt.axis("off")
            plt.scatter(x, y, color="red")
            plt.imshow(encoder_attn_weights[j][0][multihead][index].numpy().reshape(heatmap_size))
            count += 1
            plt.title('reference point: (%d, %d), layer: %d' % (x, y, j + 1), fontsize=3.5, pad=3)
    plt.suptitle('Encoder Attention Heapmap in Multi-Header %d' % (multihead + 1))
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches=0)
    plt.show()


def show_decoder_self_attn_heapmap(decoder_attn_weights, fig_size=(8, 6), save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    count = 1
    for l in range(6):
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        for h in range(8):
            plt.subplot(6, 8, count)
            plt.axis("off")
            plt.imshow(decoder_attn_weights[l][0][h].numpy())
            count += 1
            plt.title('layer: %d, multihead: %d' % (l + 1, h + 1), fontsize=4, pad=3)
    plt.suptitle('Decoder Self-Attention Heapmap')
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()


def show_decoder_cross_attn_heapmap(decoder_attn_weights, fig_size=(8, 6), save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    count = 1
    for l in range(6):
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        for h in range(8):
            plt.subplot(6, 8, count)
            plt.axis("off")
            plt.imshow(decoder_attn_weights[l][0][h].numpy())
            count += 1
            plt.title('layer: %d, multihead: %d' % (l + 1, h + 1), fontsize=4, pad=3)
    plt.suptitle('Decoder Cross-Attention Heapmap')
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()


def visualize_decoder_heapmap_by_different_slots_layer(decoder_attn_weights, slots_list, multihead, 
    fig_size, heatmap_size, save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    slot_num = len(slots_list)
    count = 1
    for slot_idx in slots_list:
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        for l in range(6):
            plt.subplot(slot_num, 6, count)
            plt.axis("off")
            plt.imshow(decoder_attn_weights[l][1][0][multihead][slot_idx].numpy().reshape(heatmap_size))
            count += 1
            plt.title('slot: %d, layer: %d' % (slot_idx + 1, l + 1), fontsize=4, pad=3)
    plt.suptitle('Decoder Cross-Attention Heapmap in Multi-Header %d' % (multihead + 1))
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()


def visualize_decoder_heapmap_by_different_slots_multihead(decoder_attn_weights, slots_list, layer, 
    fig_size, heatmap_size, save_path=None):
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = 150
    slot_num = len(slots_list)
    count = 1
    for slot_idx in slots_list:
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        for h in range(8):
            plt.subplot(slot_num, 8, count)
            plt.axis("off")
            plt.imshow(decoder_attn_weights[layer][1][0][h][slot_idx].numpy().reshape(heatmap_size))
            count += 1
            plt.title('slot: %d, multihead: %d' % (slot_idx + 1, h + 1), fontsize=4, pad=3)
    plt.suptitle('Decoder Cross-Attention Heapmap in Layer %d' % (layer + 1))
    if save_path is not None:
        plt.savefig(save_path, format='png', transparent=True, dpi=300, pad_inches =0)
    plt.show()





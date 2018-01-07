# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2
import os

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer


CAFFE_ROOT = './caffe-fast-rcnn'
CLASSES = ('__background__', 'person')


def check_file_path(file_path):
    if not os.path.isfile(file_path):
        raise IOError(('{:s} not found.').format(file_path))


def load_model(caffemodel, prototxt, gpu_mode=True, gpu_id=0):

    # 设置CPU/GPU计算
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
        caffe.set_device(gpu_id)

    # 加载模型，设置TEST阶段
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('\nLoaded network {:s}'.format(caffemodel))

    return net


def vis_detect_result(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(plt.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0],
            bbox[3] - bbox[1], 
            fill=False,
            edgecolor='red', 
            linewidth=3.5))

        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with p({} | box) >= {:.1f}').format(
        class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def detect_image(net, im_file, conf_thresh, vis_result=True):    

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    image = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, image)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals').format(
        timer.total_time, boxes.shape[0])
   
    if vis_result:
        # Visualize detections for each class
        CONF_THRESH = conf_thresh
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1    # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            vis_detect_result(image, cls, dets, thresh=CONF_THRESH)
  
    return net


# helper function
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square_helper(data, padsize=1, padval=0):
    # 归一化
    data -= data.min()
    data /= data.max()
    
    # 根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # 先将padding后的data分成n*n张图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data



def vis_weights(net, layer, num, padsize=1, padval=0):
    """
    - net: caffe net
    - layer: which layer to be visualized
    - num: number of filters
    """
    filters = net.params[layer][0].data
    N, C, W, H = filters.shape
    print("{} layers filters, size: {:d}, channels: {:d}, width: {:d}, height: {:d}".format(
        layer, N, C, W, H)) 
    if layer == 'conv1' or C == 3:
        data = vis_square_helper(filters.transpose(0, 2, 3, 1), padsize, padval)   # N, W, H, C
    else:
        data = vis_square_helper(filters[:num].reshape(C * num, W, H), padsize, padval)

    plt.imshow(data)
    plt.title("{} layer {:d} filters weight visualization.".format(layer, num))
    plt.show()


def vis_features(net, layer, num, padsize=1, padval=0):
    feats = net.blobs[layer].data[0, :num]   
    data = vis_square_helper(feats, padsize, padval)
    plt.imshow(data)
    plt.title("{} layer {:d} feature maps visualization.".format(layer, num))
    plt.show()




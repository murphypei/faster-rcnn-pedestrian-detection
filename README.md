# Pedestrian-RCNN

This code extends [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by adding:
  * ResNet implementation.
  * Online Hard Example Mining.
  * Caltech Dataset train and test interface. 
  * Add RPN tools.

## Base

* The faster rcnn code is based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).
* The ohem code is based on [ohem](https://github.com/abhi2610/ohem).
* To reduce the memory usage, we use batchnorm layer in [Microsoft's caffe](https://github.com/Microsoft/caffe)
* The caltech dataset interface based on [GBJim-py-faster-rcnn](https://github.com/GBJim/py-faster-rcnn)


## Installation

The installation and useage are same as Faster R-CNN

1. clone the repository and caffe submodule 
2. build lib and caffe
3. train your model command like: `./experiments/scripts/faster_rcnn_end2end.sh 0 ResNet50 caltech_reasonable`

## Notices

For different datasets, the interval frame rate is different, so it need to be changed.

In `./lib/datasets/caltech.py`, change the `start_frame` and `frame_rate` of `insert_frame` function.
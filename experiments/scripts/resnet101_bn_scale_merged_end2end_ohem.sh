#!/bin/bash

set -x

TRAIN_IMDB="voc_2007_trainval"
TEST_IMDB="voc_2007_test"
PT_DIR="pascal_voc"
MODEL="ResNet101_BN_SCALE_Merged"
TRAIN_NET="faster_rcnn_end2end"

# save log to file
LOG="experiments/logs/${TRAIN_NET}_${MODEL}_OHEM.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

./tools/train_net.py --gpu 0 \
  --solver models/${PT_DIR}/${MODEL}_OHEM/${TRAIN_NET}_ohem/solver.prototxt \
  --weights data/imagenet_models/${MODEL}.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters 70000 \
  --cfg experiments/cfgs/${TRAIN_NET}_ohem.yml 

./tools/test_net.py --gpu 0 \
  --def models/${PT_DIR}/${MODEL}_OHEM/${TRAIN_NET}_ohem/test.prototxt \
  --net output/${TRAIN_NET}/${TRAIN_IMDB}/resnet101_faster_rcnn_bn_scale_merged_end2end_ohem_iter_70000.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${TRAIN_NET}_ohem.yml 

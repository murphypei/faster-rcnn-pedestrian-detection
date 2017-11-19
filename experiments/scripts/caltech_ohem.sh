#!/bin/bash

set -x    # print exec command to screen.
set -e    # exit if exec command doesn't return 0.

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )                    # get all args from command as a array
len=${#array[@]}                # args array length
EXTRA_ARGS=${array[@]:3:$len}   # get another args after 3 args
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
    all)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="caltech_all_trainval"
    TEST_IMDB="caltech_all_test"
    PT_DIR="caltech"
    ITERS=120000
    ;;
    reasonable)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="caltech_reasonable_trainval"
    TEST_IMDB="caltech_reasonable_test"
    PT_DIR="caltech"
    ITERS=120000
    ;;
    person_class_only)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="caltech_person_class_trainval"
    TEST_IMDB="caltech_person_class_test"
    PT_DIR="caltech"
    ITERS=490000
    ;;
    *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_ohem_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end_ohem/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end_ohem.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x


time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end_ohem/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end_ohem.yml \
  ${EXTRA_ARGS}
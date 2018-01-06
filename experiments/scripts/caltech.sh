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


# add ohem args
if [[ ! -n $EXTRA_ARGS_SLUG ]]; then
    TRAIN_METHOD="faster_rcnn_end2end"
elif [[ $EXTRA_ARGS_SLUG = "ohem" ]]; then
    TRAIN_METHOD="faster_rcnn_end2end_ohem"
else
    echo "Invalid input train method:"
    echo ${EXTRA_ARGS_SLUG}
    exit
fi

case $DATASET in
    caltech_all)
    TRAIN_IMDB="caltech_all_trainval"
    TEST_IMDB="caltech_all_test"
    PT_DIR="caltech"
    ITERS=150000
    ;;
    caltech_reasonable)
    TRAIN_IMDB="caltech_reasonable_trainval"
    TEST_IMDB="caltech_reasonable_test"
    PT_DIR="caltech"
    ITERS=120000
    ;;
    caltech_person)
    TRAIN_IMDB="caltech_person_trainval"
    TEST_IMDB="caltech_person_test"
    PT_DIR="caltech"
    ITERS=120000
    ;;
    inria_all)
    TRAIN_IMDB="inria_all_trainval"
    TEST_IMDB="inria_all_test"
    PT_DIR="caltech"
    ITERS=70000
    ;;
    inria_reasonable)
    TRAIN_IMDB="inria_reasonable_trainval"
    TEST_IMDB="inria_reasonable_test"
    PT_DIR="caltech"
    ITERS=70000
    ;;
    inria_person)
    TRAIN_IMDB="inria_person_trainval"
    TEST_IMDB="inria_person_test"
    PT_DIR="caltech"
    ITERS=70000
    ;;
    *)
    echo "No dataset given"
    exit
    ;;
esac

echo ${GPU_ID}
echo ${NET}
echo ${DATASET}
echo ${TRAIN_METHOD}
echo ${TRAIN_IMDB}
echo ${TEST_IMDB}
echo ${PT_DIR}
echo ${ITERS}


LOG="experiments/logs/${TRAIN_METHOD}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/${TRAIN_METHOD}/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${TRAIN_METHOD}.yml \


set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/${TRAIN_METHOD}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${TRAIN_METHOD}.yml \
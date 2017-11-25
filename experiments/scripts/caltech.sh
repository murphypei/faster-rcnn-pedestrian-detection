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
if [[ "${EXTRA_ARGS_SLUG}" == "ohem" ]]; then
   TRAIN_METHOD="faster_rcnn_end2end_ohem"
else
    TRAIN_METHOD="faster_rcnn_end2end"
fi

PT_DIR="caltech"


case $DATASET in
    all)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="caltech_all_trainval"
    TEST_IMDB="caltech_all_test"
    ITERS=120000
    ;;
    reasonable)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="caltech_reasonable_trainval"
    TEST_IMDB="caltech_reasonable_test"
    ITERS=110000
    ;;
    person)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="caltech_person_trainval"
    TEST_IMDB="caltech_person_test"
    ITERS=70000
    ;;
    *)
    echo "No dataset given"
    exit
    ;;
esac

# echo ${GPU_ID}
# echo ${NET}
# echo ${DATASET}
# echo ${TRAIN_METHOD}
# echo ${TRAIN_IMDB}
# echo ${TEST_IMDB}
# echo ${PT_DIR}
# echo ${ITERS}
# exit

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
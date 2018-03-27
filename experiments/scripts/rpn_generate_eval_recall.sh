#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ;;
  coco)
    echo "Not implemented: use experiments/scripts/faster_rcnn_end2end.sh for coco"
    exit
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/rpn_test_eval_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/rpn_generate.py --gpu ${GPU_ID} \
  --def ./models/caltech/${NET}/faster_rcnn_alt_opt/rpn_test.pt \
  --cfg ./experiments/cfgs/faster_rcnn_rpn_only.yml \
  --imdb ${TEST_IMDB} \
  ${EXTRA_ARGS}


set +x
RPN_FILE=`grep "Wrote RPN proposals" ${LOG} | awk '{print $5}'`
set -x

METHOD="rpn"

time ./tools/eval_recall.py --imdb ${TEST_IMDB} --method ${METHOD} --rpn-file ${RPN_FILE}
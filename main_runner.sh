#!/bin/bash

# todo: activate tf environment

PREPROC_NAME=NEW_DS_BASE_1
LEARN_NAME=P-NET2_7_p
MODEL_NAME=${PREPROC_NAME}_${LEARN_NAME}

MODEL_IN_DIR=notebook_dumps/models_in/$MODEL_NAME

mkdir -p notebook_dumps/logs
LOG_FILE=notebook_dumps/logs/`date '+%Y-%m-%d_%H%M%S'`.log


# run code
python3 main_cluster.py --dataset-input $MODEL_IN_DIR/$MODEL_NAME.dataset.params.json --preproc-input $MODEL_IN_DIR/$PREPROC_NAME.preproc.params.json --model-input $MODEL_IN_DIR/$LEARN_NAME.learn.params.json --train-input $MODEL_IN_DIR/$MODEL_NAME.train.params.json --load-model-name $MODEL_IN_DIR/$MODEL_NAME.h5 -epochs 2 -batch 4 --reduce-train-set 8 --reduce-val-set 4 --preproc-prefix local --model-prefix local 2>&1 | tee $LOG_FILE

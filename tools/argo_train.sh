#!/usr/bin/env bash

CONFIG=projects/configs/$1.py
GPUS=$2
PORT=${PORT:-28230}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --work-dir work_dirs/$1 \
    --launcher pytorch ${@:3} \
    --cfg-options evaluation.pklfile_prefix=work_dirs/$1/argo/results 
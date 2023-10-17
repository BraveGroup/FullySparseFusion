#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-30600}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
    --eval bbox 
#--cfg-options evaluation.jsonfile_prefix=

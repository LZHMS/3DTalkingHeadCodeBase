#!/bin/bash

# Configuration
NUM_GPUS=1  # Number of GPUs to use
MASTER_PORT=29500  # Port for distributed communication

TRAINER=ToyTrainer
DATASET=MNIST
TAG=GPU0

DIR=output/${DATASET}/${TRAINER}/${TAG}
if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}. Skip this job"
  rm -rf ${DIR}
else
  mkdir -p ${DIR}
  python -m torch.distributed.run --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py --mode train \
    --config-file config/toy_trainer_config.yaml \
    ENV.OUTPUT_DIR ${DIR} \
    DATASET.NAME ${DATASET} \
    TRAINER.NAME ${TRAINER} \
    2>&1 | tee ${DIR}/output.log
fi
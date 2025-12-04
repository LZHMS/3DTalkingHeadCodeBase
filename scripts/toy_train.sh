#!/bin/bash

# Configuration
NUM_GPUS=2  # Number of GPUs to use
MASTER_PORT=29500  # Port for distributed communication

TRAINER=ToyTrainer
DATASET=MNIST
TAG=GPU01

DIR=output/${DATASET}/${TRAINER}/${TAG}

mkdir -p ${DIR}

/zhli/miniconda3/envs/diffposetalk/bin/python -m torch.distributed.run --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  train.py --mode train \
  --config-file config/toy_trainer_config.yaml \
  ENV.OUTPUT_DIR ${DIR} \
  DATASET.NAME ${DATASET} \
  TRAINER.NAME ${TRAINER} \
  2>&1 | tee ${DIR}/output.log
#!/bin/bash

# custom config
DATA=./data
TRAINER=FlowMatchingTrainer  # DiffPoseTalkTrainer or FlowMatchingTrainer 

DATASET=HDTF_TFHP
TAG=Train

DIR=output/${DATASET}/${TRAINER}/${TAG}
STYLEDIR=output/${DATASET}/StyleEncoderTrainer/${TAG}/model/model.pth.tar-iter-100000
mkdir -p ${DIR}
python -m main.train \
  --config-file config/flowmatching_trainer_config.yaml \
  ENV.OUTPUT_DIR ${DIR} \
  DATASET.ROOT ${DATA} \
  DATASET.NAME ${DATASET} \
  TRAINER.NAME ${TRAINER} \
  ENV.EXTRA.STYLE_ENC_CKPT ${STYLEDIR} \
  2>&1 | tee ${DIR}/output.log
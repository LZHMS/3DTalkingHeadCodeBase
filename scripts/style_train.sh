#!/bin/bash

# custom config
TRAINER=StyleEncoderTrainer

DATASET=HDTF_TFHP
TAG=Train

DIR=output/${DATASET}/${TRAINER}/${TAG}
if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}. Skip this job"
  rm -rf ${DIR}
else
  mkdir -p ${DIR}
  /zhli/miniconda3/envs/diffposetalk/bin/python -m main.train \
    --config-file config/style_trainer_config.yaml \
    ENV.OUTPUT_DIR ${DIR} \
    DATASET.NAME ${DATASET} \
    TRAINER.NAME ${TRAINER} \
    2>&1 | tee ${DIR}/output.log
fi
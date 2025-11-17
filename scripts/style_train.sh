#!/bin/bash

# custom config
DATA=./data
TRAINER=StyleEncoderTrainer

DATASET=HDTF_TFHP
TAG=Train

DIR=output/${DATASET}/${TRAINER}/${TAG}
if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}. Skip this job"
  rm -rf ${DIR}
else
  mkdir -p ${DIR}
  /zhli/miniconda3/envs/diffposetalk/bin/python -m main.train --config-file config/style_trainer_config.yaml \
    --wandb-name "${TRAINER}_${DATASET}_${TAG}" \
    --wandb-notes "Training Style Encoder on ${DATASET}" \
    --wandb-tags "StyleEncoder,${DATASET},${TAG}" \
    --use-wandb \
    --wandb-mode "offline" \
    ENV.OUTPUT_DIR ${DIR} \
    DATASET.ROOT ${DATA} \
    DATASET.NAME ${DATASET} \
    TRAINER.NAME ${TRAINER} \
    2>&1 | tee ${DIR}/output.log
fi
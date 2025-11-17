#!/bin/bash

# custom config
DATA=./data
TRAINER=DiffPoseTalkTrainer

DATASET=HDTF_TFHP
TAG=Train

DIR=output/${DATASET}/${TRAINER}/${TAG}
STYLEDIR=output/${DATASET}/StyleEncoderTrainer/${TAG}/model/model.pth.tar-iter-100000
if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}. Skip this job"
  rm -rf ${DIR}
else
  mkdir -p ${DIR}
  /zhli/miniconda3/envs/diffposetalk/bin/python -m main.train --config-file config/difftalk_trainer_config.yaml \
    --wandb-name "${TRAINER}_${DATASET}_${TAG}" \
    --wandb-notes "Training diffusion model of DiffPoseTalker on ${DATASET}" \
    --wandb-tags "DenoisingModel,DiffPoseTalk,${DATASET},${TAG}" \
    --use-wandb \
    --wandb-mode "offline" \
    ENV.OUTPUT_DIR ${DIR} \
    DATASET.ROOT ${DATA} \
    DATASET.NAME ${DATASET} \
    TRAINER.NAME ${TRAINER} \
    ADD.STYLE_ENC_CKPT ${STYLEDIR} \
    2>&1 | tee ${DIR}/output.log
fi
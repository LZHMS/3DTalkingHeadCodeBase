#!/bin/bash

export PYOPENGL_PLATFORM=osmesa

# custom config
TRAINER=DiffPoseTalkTrainer  # DiffPoseTalkTrainer or FlowMatchingTrainer
DATASET=HDTF_TFHP
TAG=Train

DIR=output/${DATASET}/${TRAINER}/${TAG}
STYLEDIR=output/${DATASET}/StyleEncoderTrainer/${TAG}/model/model.pth.tar-iter-100000
if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}. Skip this job"
  rm -rf ${DIR}
else
  mkdir -p ${DIR}
  python train.py --mode train\
    --config-file config/difftalk_trainer_config.yaml \
    ENV.OUTPUT_DIR ${DIR} \
    DATASET.NAME ${DATASET} \
    TRAINER.NAME ${TRAINER} \
    ENV.EXTRA.STYLE_ENC_CKPT ${STYLEDIR} \
    2>&1 | tee ${DIR}/output.log
fi
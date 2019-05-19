#!/bin/zsh

TOTAL_RUNS=5
ARCH_NAME=resnet18_tinyimagenet

for RUN in $(seq $TOTAL_RUNS)
do
    SAVE_DIR="save_${ARCH_NAME}_run_${RUN}"
    LOG_NAME="run.log"
    COMMAND="python -u ../../main.py --arch ${ARCH_NAME} |& tee ${LOG_NAME}"
    mkdir -p ${SAVE_DIR}
    cd ${SAVE_DIR}
    echo ${COMMAND}
    eval ${COMMAND}
    cd ..
done

#!/bin/zsh

TOTAL_RUNS=15
DATASET=cifar64_ordered
LIST_OF_ARCH_NAMES=(resnet32_cifar64_fixed_eye_no_bias)
OPTIMIZER=resnet_cifar_sgd
EPOCHS=200

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`
ROOT_PATH=`realpath ${SCRIPT_PATH}/../../`
MAIN_PY_SCRIPT=`realpath ${ROOT_PATH}/model-eval/main.py`


for ARCH_NAME in ${LIST_OF_ARCH_NAMES}
do
    for RUN in $(seq 1 $TOTAL_RUNS)
    do
        SAVE_DIR="save_${ARCH_NAME}_run_${RUN}"
        LOG_NAME="run.log"
        COMMAND="python -u ${MAIN_PY_SCRIPT} -a ${ARCH_NAME} -d ${DATASET} --optimizer ${OPTIMIZER} --epochs ${EPOCHS}|& tee ${LOG_NAME}"
        mkdir -p ${SAVE_DIR}
        cd ${SAVE_DIR}
        echo ${COMMAND}
        eval ${COMMAND}
        cd ..
    done
done

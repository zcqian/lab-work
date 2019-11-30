#!/bin/zsh

TOTAL_RUNS=15
LIST_OF_DATASETS=(cifar10_imbalance_10 cifar10_imbalance_50 cifar10_imbalance_200)
LIST_OF_ARCH_NAMES=(resnet32_cifar10 resnet32_cifar10_fixed_eye)
OPTIMIZER=resnet_cifar_sgd
EPOCHS=200

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`
ROOT_PATH=`realpath ${SCRIPT_PATH}/../../`
MAIN_PY_SCRIPT=`realpath ${ROOT_PATH}/model-eval/main.py`

for DATASET in ${LIST_OF_DATASETS}
do
    mkdir ${DATASET}
    cd ${DATASET}
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
    cd ..
done

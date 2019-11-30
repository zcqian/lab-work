#!/bin/zsh

START_RUN=1
TOTAL_RUNS=10
LIST_OF_DATASETS=(cifar10)
LIST_OF_ARCH_NAMES=(rn32_cf10 rn32_10_fc_sq_ex)
OPTIMIZER=resnet_cifar_sgd_wd5
EPOCHS=200

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`
ROOT_PATH=`realpath ${SCRIPT_PATH}`
while [[ ! -f ${ROOT_PATH}/model-eval/main.py ]]; do
    ROOT_PATH=`realpath ${ROOT_PATH}/..`
done
MAIN_PY_SCRIPT=`realpath ${ROOT_PATH}/model-eval/main.py`
BASE_WORK_DIR=`pwd`

for RUN in $(seq $START_RUN $TOTAL_RUNS); do
    for DATASET in ${LIST_OF_DATASETS}; do
        for ARCH_NAME in ${LIST_OF_ARCH_NAMES}; do            
            WORK_DIR="${BASE_WORK_DIR}/${DATASET}/${ARCH_NAME}/save_run_${RUN}"
            LOG_NAME="run.log"
            COMMAND="python -u \"${MAIN_PY_SCRIPT}\" -a ${ARCH_NAME} -d ${DATASET} --optimizer ${OPTIMIZER} --epochs ${EPOCHS} > ${LOG_NAME}"
            # sure it ain't thread/multi-process safe
            if ! [[ -e ${WORK_DIR}/colab-run-complete ]]; then
                rm -rf ${WORK_DIR}
                mkdir -p ${WORK_DIR}
                pushd ${WORK_DIR}
                echo ${COMMAND}
                eval ${COMMAND}
                touch ${WORK_DIR}/colab-run-complete
                popd
            else
                echo "Ignoring ${WORK_DIR} because it is already complete." 
            fi
        done
    done
done

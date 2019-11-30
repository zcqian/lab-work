#!/bin/zsh

START_RUN=1
TOTAL_RUNS=10
LIST_OF_DATASETS=(cifar100 cifar100_imbalance_10)

LIST_OF_ARCH_NAMES=(rn32_cf100)
OPTIONS=(a b c d e f g null)
for OPTION in ${OPTIONS}; do
    LIST_OF_ARCH_NAMES+=(rn32_cf100_ex${OPTION} rn32_cf100_ex${OPTION}_fixed_eye 
                         rn32_cf100_ex${OPTION}_no_bias rn32_cf100_ex${OPTION}_fixed_eye_no_bias)
done
LIST_OF_ARCH_NAMES+=(rn32_cf100_exg_groups_2_fixed_eye rn32_cf100_exg_groups_4_fixed_eye
                     rn32_cf100_exg_d_sep_fixed_eye)

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
            COMMAND="python -u ${MAIN_PY_SCRIPT} -a ${ARCH_NAME} -d ${DATASET} --optimizer ${OPTIMIZER} --epochs ${EPOCHS}|& tee ${LOG_NAME}"
            if ! [[ -e ${WORK_DIR}/model_best.pth.tar ]]; then
                mkdir -p ${WORK_DIR}
                pushd ${WORK_DIR}
                echo ${COMMAND}
                eval ${COMMAND}
                popd
            else
                echo "Ignoring ${WORK_DIR} because model_best.pth.tar already exists there." 
            fi
        done
    done
done

#!/bin/zsh

START_RUN=1
TOTAL_RUNS=10
LIST_OF_DATASETS=(cifar100_imbalance_10 cifar100_imbalance_20 cifar100_imbalance_50 cifar100_imbalance_100 cifar100_imbalance_200)
LIST_OF_ARCH_NAMES=(resnet32_cifar100 resnet32_cifar100_conv100 resnet32_cifar100_conv100_no_last_relu_fixed_eye_no_bias
                    resnet32_cifar100_conv100_fixed_eye resnet32_cifar100_conv100_fixed_eye_no_bias
                    resnet32_cifar100_conv100_no_last_relu_fixed_eye resnet32_cifar100_conv100_no_last_relu_fixed_eye_no_bias
                    resnet32_cifar100_conv100_no_bias resnet32_cifar100_no_bias
)
OPTIMIZER=resnet_cifar_sgd
EPOCHS=200

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`
ROOT_PATH=`realpath ${SCRIPT_PATH}/../../`
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

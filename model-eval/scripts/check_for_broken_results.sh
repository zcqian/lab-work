#!/bin/zsh

for TARGET_DIR in $@; do
    for f in $(find "${TARGET_DIR}" -name run.log); do
        DIR=$(dirname $f)
        if [[ ! -f ${DIR}/model_best.pth.tar ]]; then;
            echo "${DIR} is bad"
        fi
    done
done
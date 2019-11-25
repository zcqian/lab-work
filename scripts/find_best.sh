#!/bin/bash

INPUT_FILE=$1

top1=$(grep '*' ${INPUT_FILE} | cut -d ' ' -f 4 | sort -nr | head -n1)

grep '*' ${INPUT_FILE} | nl | grep ${top1}

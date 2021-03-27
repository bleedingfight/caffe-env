#!/bin/bash
export LD_LIBRARY_PATH='/opt/OpenBLAS/lib':'/home/liushuai/miniconda3/lib':'/usr/local/lib':${LD_LIBRARY_PATH}
CAFFE_ROOT=${HOME}/caffe
NETWORK_ROOT=/tmp/pet
Solver_PATH=${NETWORK_ROOT}/solver.prototxt
TRAIN_VAL=${NETWORK_ROOT}/train_val.pbtxt
python train_pet.py
${CAFFE_ROOT}/.build_debug/tools/caffe train --solver=${Solver_PATH} --gpu=0 >train.log 2>&1
# ${CAFFE_ROOT}/.build_debug/tools/caffe test -model ${CAFFE_ROOT}/models/resnet50/train_val.prototxt -weights resnet-50-cervix_iter_900.caffemodel

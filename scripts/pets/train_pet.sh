#!/bin/bash
CAFFE_ROOT=${HOME}/caffe
SLOVER_ROOT=${CAFFE_ROOT}/models/resnet50
${CAFFE_ROOT}/.build_debug/tools/caffe train --solver=$SLOVER_ROOT/solver.prototxt --gpu=0
${CAFFE_ROOT}/.build_debug/tools/caffe test -model ${CAFFE_ROOT}/models/resnet50/train_val.prototxt -weights resnet-50-cervix_iter_440.caffemodel

#!/bin/bash
CAFFE_ROOT=${HOME}/caffe-env
SLOVER_ROOT=${CAFFE_ROOT}/models/resnet50
${CAFFE_ROOT}/.build_debug/tools/caffe train --solver=$SLOVER_ROOT/solver.prototxt --gpu=0

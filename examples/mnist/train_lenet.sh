#!/usr/bin/env sh
set -e
CAFFE_HOME=${HOME}/caffe
MODEL_PATH=${CAFFE_HOME}/examples/mnist
export LD_LIBRARY_PATH="/opt/OpenBLAS/lib":${LD_LIBRARY_PATH}
# ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
/home/liushuai/caffe/.build_release/tools/caffe train --solver=${MODEL_PATH}/lenet_solver.prototxt $@

#!/usr/bin/env sh
set -e
CAFFE_HOME=${HOME}/caffe-env 
if [ ! -d ${CAFFE_HOME} ];then
	CAFFE_HOME=${HOME}/caffe
fi
backup=lenet_network
if [ ! -d ${backup} ];then
	mkdir ${backup}
fi
MODEL_PATH=${CAFFE_HOME}/examples/fashion_mnist
export LD_LIBRARY_PATH="${HOME}/miniconda3/lib":"/opt/OpenBLAS/lib":"/usr/local/lib":${LD_LIBRARY_PATH}
# ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
caffe_bin=`find ${CAFFE_HOME} -type f -perm -111 -name caffe.bin`
${caffe_bin} train --solver=${MODEL_PATH}/lenet_solver_gru.prototxt $@

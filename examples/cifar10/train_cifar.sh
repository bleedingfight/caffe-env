#!/bin/bash
CAFFE_HOME=${HOME}/caffe
printf "\033[32mCurrent Caffe Home is:%s\n\033[0m" ${CAFFE_HOME}
printf "\033[31mCreate cifar dataset\033[0m\n"
cd ${CAFFE_HOME}
bash create_cifar10.sh
caffe_bin=`find ${CAFFE_HOME} -name caffe.bin`
printf "===>Caffe bin ${caffe_bin}"
${caffe_bin} train \
  --solver=${CAFFE_HOME}/examples/cifar10/cifar10_quick_solver.prototxt $@

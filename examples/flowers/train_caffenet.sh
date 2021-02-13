#!/usr/bin/env sh
set -e
printf "\033[31mCreate Dataset.....\n\033[0m"
bash create_flower.sh 227 227
CAFFE_HOME=${HOME}/caffe
if [ ! -d ${CAFFE_HOME} ];then
	printf "You Must Set Caffe HOME,Default Caffe Home is:\033[31m%s\033[0m\n" ${CAFFE_HOME}
fi
CAFFE_BIN=`find ${CAFFE_HOME} -name "caffe" -perm 0777`
${CAFFE_BIN} train \
    --solver=${CAFFE_HOME}/models/flower_caffenet/solver.prototxt --gpu=0

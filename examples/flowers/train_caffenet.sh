#!/usr/bin/env sh
set -e
DATASET_SOURCE=${HOME}/Datasets/flower_photos
OUTPUT_DATASOURCE=/tmp/flower_photos
printf "\033[31mCreate Dataset.....\n\033[0m"
bash create_flower.sh 227 227
CAFFE_HOME=${HOME}/caffe
if [ ! -d ${CAFFE_HOME} ];then
	CAFFE_HOME=${HOME}/caffe-env 
fi 

if [ ! -d ${CAFFE_HOME} ];then
	printf "You Must Set Caffe HOME,Default Caffe Home is:\033[31m%s\033[0m\n" ${CAFFE_HOME}
	exit
fi
CAFFE_BIN=`find ${CAFFE_HOME} -name "caffe" -perm 0777`
MEAN_TOOLS=`find ${CAFFE_HOME} -name compute_image_mean`
${MEAN_TOOLS} ${OUTPUT_DATASOURCE}/export/flower_val_lmdb ${OUTPUT_DATASOURCE}/export/mean_val.binaryproto
${MEAN_TOOLS} ${OUTPUT_DATASOURCE}/export/flower_train_lmdb ${OUTPUT_DATASOURCE}/export/mean_train.binaryproto
cd ${CAFFE_HOME}
${CAFFE_BIN} train \
    --solver=${CAFFE_HOME}/models/bvlc_reference_caffenet/solver.prototxt --gpu=0

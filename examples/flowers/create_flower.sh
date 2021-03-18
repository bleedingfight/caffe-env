#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e
CAFFE_HOME=${HOME}/caffe
if [ ! -d ${CAFFE_HOME} ];then 
	CAFFE_HOME=${HOME}/caffe-env 
fi 
if [ ! -d ${CAFFE_HOME} ];then
	exit
fi
DATASET_PATH=${HOME}/Datasets/flower_photos
OUTPUT_DATA_PATH=/tmp/flower_photos/export

RESIZE_W=$1
RESIZE_H=$2
PROCESSED_PATH=$3

printf "\033[31m weight: [%s] height: [%s]\n\033[0m\n" ${RESIZE_W} ${RESIZE_H}
# 处理后的数据的存放路径
if [ -z ${PROCESSED_PATH} ];then
	printf "Current dataset export path:\033[31m %s \033[0m exists,it will be delete!\n" ${PROCESSED_PATH}
	PROCESSED_PATH=${OUTPUT_DATA_PATH}
fi
printf "\033[33mDataset path:[%s] Output data path:[%s]\033[0m\n" ${DATASET_PATH} ${PROCESSED_PATH}

# python data_process.py -d ${DATASET_PATH} -o ${PROCESSED_PATH} -t train_val.pbtxt -s solver.pbtxt

python data_process.py -d ${DATASET_PATH} -o ${PROCESSED_PATH} -t ${CAFFE_HOME}/models/bvlc_reference_caffenet/train_val.prototxt -s ${CAFFE_HOME}/models/bvlc_reference_caffenet/solver.prototxt -de True
# 数据数据为TRAIN_DATA_ROOT+train.txt文件中写好的路径
TRAIN_DATA_ROOT=/
TEST_DATA_ROOT=/
# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE_HEIGHT=${RESIZE_H}
RESIZE_WIDTH=${RESIZE_W}
if [ ! -d ${PROCESSED_PATH} ];then
	PROCESSED_PATH=${DATASET_PATH}
fi 

printf "Resized Image to [\033[31m%s\033[0m x \033[31m%s\033[0m]\n" ${RESIZE_HEIGHT} ${RESIZE_WIDTH}

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_flower.sh to the path" \
       "where the processed flower training data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_flower.sh to the path" \
       "where the processed flower validation data is stored."
  exit 1
fi

printf "\033[31mCreating train lmdb...\033[0m\n"

tools=`find ${CAFFE_HOME} -name convert_imageset`
if [ ! -f ${tools} ];then
	exit
fi 
train_lmdb=${PROCESSED_PATH}/flower_train_lmdb
val_lmdb=${PROCESSED_PATH}/flower_val_lmdb
if [ -d ${train_lmdb} ];then
	rm -rf ${train_lmdb}
fi
if [ -d ${val_lmdb} ];then
	rm -rf ${val_lmdb}
fi 
GLOG_logtostderr=1 ${tools} \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    ${PROCESSED_PATH}/train.txt \
    ${train_lmdb}

printf "\033[31mCreating val lmdb...\n\033[0m"

GLOG_logtostderr=1 ${tools} \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    ${PROCESSED_PATH}/test.txt \
    ${val_lmdb}

mean_tool=`find ${CAFFE_HOME} -name compute_image_mean`
${mean_tool} ${PROCESSED_PATH}/flower_train_lmdb ${PROCESSED_PATH}/flower_train.binaryproto
${mean_tool} ${PROCESSED_PATH}/flower_val_lmdb ${PROCESSED_PATH}/flower_val.binaryproto
printf "\033[310Create binaryproto file finished\033[0m"

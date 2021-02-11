#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e
CAFFE_HOME=${HOME}/caffe

DATA=/tmp/flower_photos/export
EXAMPLE=${DATA}
TOOLS=${CAFFE_HOME}/.build_release/tools
# 处理后的数据的存放路径
PROCESSED_PATH=/tmp/flower_photos/export
if [ -d ${PROCESSED_PATH} ];then
	printf "Current dataset export path:\033[31m %s \033[0m exists,it will be delete!\n" ${PROCESSED_PATH}
	rm -rf ${PROCESSED_PATH}
fi
python data_process.py
# 数据数据为TRAIN_DATA_ROOT+train.txt文件中写好的路径
TRAIN_DATA_ROOT=/
TEST_DATA_ROOT=/
# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
  printf "Resized Image to [\033[31m%s\033[0m x \033[31m%s\033[0m]\n" ${RESIZE_HEIGHT} ${RESIZE_WIDTH}
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

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

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/flower_train_lmdb

printf "\033[31mCreating val lmdb...\n\033[0m"

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/flower_test_lmdb
printf "\033[31mComputer flower means:%s\033[0m\n" ${CAFFE_HOME}/examples/flowers/mean.binaryproto
${TOOLS}/compute_image_mean /tmp/flower_photos/export/flower_train_lmdb ${CAFFE_HOME}/examples/flowers/flower_mean.binaryproto

echo "Done."

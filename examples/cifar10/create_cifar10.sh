#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e
CAFFE_HOME=${HOME}/caffe
cd ${CAFFE_HOME}
EXAMPLE=${CAFFE_HOME}/examples/cifar10
DATA=${CAFFE_HOME}/data/cifar10
DBTYPE=lmdb
function delete_output_path(){
	output_data_path=$1
    if [ -d ${output_data_path} ];then
	    rm -rf ${output_data_path}
    fi
}
echo "Creating $DBTYPE..."
output_train_path=$EXAMPLE/cifar10_train_$DBTYPE 
output_test_path=$EXAMPLE/cifar10_test_$DBTYPE
delete_output_path ${output_train_path}
delete_output_path ${output_test_path}
convert_tool=`find ${CAFFE_HOME} -name convert_cifar_data.bin -type f`

${convert_tool} $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."
compute_image_mean=`find ${CAFFE_HOME} -name compute_image_mean`
${compute_image_mean} -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."

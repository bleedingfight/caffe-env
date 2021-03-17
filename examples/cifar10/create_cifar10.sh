#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e
CAFFE_HOME=${HOME}/caffe-env 
EXAMPLE=${CAFFE_HOME}/examples/cifar10
DATA=${CAFFE_HOME}/data/cifar10
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."

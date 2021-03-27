#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e
CAFFE_HOME=${HOME}/caffe-env
export LD_LIBRARY_PATH="/usr/local/lib":${LD_LIBRARY_PATH}
if [ ! -d ${CAFFE_HOME} ];then
	CAFFE_HOME=${HOME}/caffe
fi
printf "Caffe Home:\033[31m%s \033[0m\n" ${CAFFE_HOME}
EXAMPLE=${CAFFE_HOME}/examples/fashion_mnist
DATA=${CAFFE_HOME}/data/fashion-mnist
if [ ! -d ${EXAMPLE} ];then
	mkdir -p ${EXAMPLE}
else
	rm -rf ${EXAMPLE}
	mkdir -p ${EXAMPLE}
fi
solver_path=`find ${CAFFE_HOME} -type f -name lenet_solver.prototxt`
network=`find ${CAFFE_HOME} -type f -name lenet_train_test.prototxt`
cp ${solver_path} ${EXAMPLE}/lenet_solver.prototxt
cp ${network} ${EXAMPLE}

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/mnist_train_${BACKEND}
rm -rf $EXAMPLE/mnist_test_${BACKEND}
convert_tools=`find ${CAFFE_HOME} -name convert_mnist_data.bin`
${convert_tools} $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_${BACKEND} --backend=${BACKEND}
${convert_tools} $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."

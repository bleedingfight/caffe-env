#!/bin/bash
export PYTHONPATH="/home/liushuai/caffe-env/python":${PYTHONPATH}
export LD_LIBRARY_PATH="/home/liushuai/caffe-env/.build_debug/lib":${LD_LIBRARY_PATH}
CAFFE_HOME=${HOME}/caffe-env 
cd ${CAFFE_HOME}
python language.py

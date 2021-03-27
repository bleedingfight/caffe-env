import caffe
from os.path import join
import sys
import os
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
# this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = os.environ['HOME']+"/caffe"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
print("Caffe Home:{}".format(caffe_root))
# this file should be run from {caffe_root}/examples (otherwise change this line)
model = "/home/liushuai/caffe/examples/fashion_mnist/lenet_network"
deploy_model = join(model, 'deploy.prototxt')
model_weight = join(model, 'model.caffemodel')
caffe.set_mode_gpu()
net = caffe.Net(deploy_model,      # defines the structure of the model
                model_weight,  # contains the trained weights
                caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
(train_x, train_y), (test_x, test_y) = mnist.load_data()
data = test_x[0].reshape((1, 1, 28, 28))
net.blobs['data'].data[...] = data
output = net.forward()
print("predict:{}".format(np.argmax(output['prob'])))

import tensorflow as tf
import caffe
import numpy as np
lstm_deploy = './lstm_caffe_model/deploy.prototxt'
caffe_model = 'lstm_mnist.caffemodel'
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
test_data = test_x[0, :, :]
test_label = test_y[0]
model = caffe.Net(lstm_deploy, caffe_model, caffe.TEST)
model.blobs['clip'] = np.random.randint(0, 1, (1, 1))
model.blobs['data'] = test_data
output = model.forward()
print("Caffe model predict res:{} thuth:{}".format(
    np.argmax(output['prob'], axis=1), test_label))

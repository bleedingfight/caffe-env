import caffe
import sys
import os
import tensorflow as tf
# this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = os.environ['HOME']+"/caffe-env"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
print("Caffe Home:{}".format(caffe_root))


def inference_with_caffe(solver, caffe_model):
    caffe.set_mode_cpu()
    net = caffe.Net(solver, caffe_model, caffe.TEST)
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
    transformed_image = test_x[0][:][:]
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    prob = tf.argmax(output['prob'], axis=1)
    print("Predict value:{} ground thruth:{}".format(prob, test_y[0]))


if __name__ == "__main__":
    model_path = '/home/liushuai/caffe-env/examples/fashion_mnist/lstm_caffe_model'
    solver = os.path.join(model_path, 'deploy.prototxt')
    caffe_model = os.path.join(model_path, "model.caffemodel")
    inference_with_caffe(solver, caffe_model)

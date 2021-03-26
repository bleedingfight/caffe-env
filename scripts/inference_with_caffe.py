import caffe
import sys
import os
# this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = os.environ['HOME']+"/caffe-env"
sys.path.insert(0, os.path.join(caffe_root, 'python'))
print("Caffe Home:{}".format(caffe_root))


def inference_with_caffe(solver, caffe_model):
    caffe.set_mode_cpu()
    net = caffe.Net(solver, caffe_model, caffe.TEST)
    print(net)


if __name__ == "__main__":
    model_path = '/home/liushuai/caffe_model/inception-bn-res-blstm'
    solver = os.path.join(model_path, 'deploy.prototxt')
    caffe_model = os.path.join(model_path, "model.caffemodel")
    inference_with_caffe(solver, caffe_model)

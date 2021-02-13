import caffe
import cv2
import sys
import numpy as np
sys.path.insert(0, '../../python')


def parse_means(means_ninary, np_array_file='mean.npy'):
    blob = caffe.proto.caffe_pb2.BlobProto()
    with open(means_ninary, 'rb') as f:
        data = f.read()
    blob.ParseFromString(data)
    array_data = caffe.io.blobproto_to_array(blob)  # [1,3,w,h]
    np.save(np_array_file, array_data[0])
    return array_data


def inference(config, weights):
    pass


if __name__ == "__main__":
    binary_file = 'flower_mean.binaryproto'
    data = parse_means(binary_file)
    print(data[0])

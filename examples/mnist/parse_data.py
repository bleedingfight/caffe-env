import cv2 as cv
import os
import lmdb
from caffe.proto import caffe_pb2
import caffe
import sys


def convert_lmdb_to_image(lmdb_data, image_path):
    env = lmdb.open(lmdb_data, readonly=True)  # 打开数据文件
    txn = env.begin()  # 生成处理句柄
    cur = txn.cursor()  # 生成迭代器指针
    datum = caffe_pb2.Datum()  # caffe 定义的数据类型

    for key, value in cur:
        datum.ParseFromString(value)  # 反序列化成datum对象

        label = datum.label
        data = caffe.io.datum_to_array(datum)
        image = data[0]
        image = data.transpose(1, 2, 0)
        filename = os.path.join(image_path, '{}.jpg'.format(str(label)))
        cv.imwrite(filename, image)
    env.close()


if __name__ == "__main__":
    lmdb_path = 'mnist_train_lmdb'
    image_path = '/tmp/mnist_image'
    convert_lmdb_to_image(lmdb_path, image_path)

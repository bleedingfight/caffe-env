import caffe
import numpy as np

MEAN_PROTO_PATH = '/tmp/pet/mean_val.binaryproto'        # 待转换的pb格式图像均值文件路径
MEAN_NPY_PATH = 'mean.npy'             # 转换后的numpy格式图像均值文件路径

blob = caffe.proto.caffe_pb2.BlobProto()      # 创建protobuf blob
data = open(MEAN_PROTO_PATH, 'rb').read()     # 读入mean.binaryproto文件内容
blob.ParseFromString(data)             # 解析文件内容到blob

array = np.array(caffe.io.blobproto_to_array(blob))[0,:,:,:]
np.save('mean.npy',array)

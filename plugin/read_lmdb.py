import lmdb
import numpy as np
import cv2
data = '/home/liushuai/caffe-env/examples/fashion_mnist/mnist_test_lmdb'
env = lmdb.open(data, readonly=True)
txn = env.begin()
cur = txn.cursor()
key, value = next(iter(cur))
im = np.frombuffer(image, dtype=np.uint8)
print(image)
# while True:
#     cv2.imshow('image', image)
#     k == cv2.waitKey(1)
#     if k == 27:
#         break
# cv2.destroyAllWindows()

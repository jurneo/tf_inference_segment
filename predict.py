import os
import sys
import argparse
import numpy as np
import cv2

import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import models
# from torchvision import transforms

save_model_path = "saved_model.pb"

graph_def = tf.GraphDef()

image = cv2.imread('104.bmp', -1)
image = cv2.resize(image, (256, 256), 0, 0, interpolation=cv2.INTER_LINEAR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
np_image = np.asarray(image).astype(np.float)
# print(np_image)
np_image = np_image / 255.0

temp = []
temp.append(np_image)

try:
    with tf.device('/device:GPU:5'):   
        with tf.gfile.Open(save_model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            sess = tf.Session(graph=graph)

            input_tensor = graph.get_tensor_by_name('input_1:0')
            output_tensor = graph.get_tensor_by_name('conv2d_16/Sigmoid:0')

            scores = sess.run(output_tensor, feed_dict={input_tensor: temp})
            print(scores.shape)
            # print(scores)
except RuntimeError as e:
    print(e)

# new_image = np.zeros(shape=[256, 256, 1], dtype=np.uint8)
# for i in range(scores.shape[1]):
#     for j in range(scores.shape[2]):
#         for c in range (scores.shape[3]):
#             new_image[i][j][c] = int (scores[0][i][j][c] * 255.0)

# cv2.imwrite("result.png", new_image)



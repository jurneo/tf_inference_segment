import os
import sys
import argparse
import numpy as np
import cv2
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf
import time

image = cv2.imread('104.bmp', -1)
image = cv2.resize(image, (256, 256), 0, 0, interpolation=cv2.INTER_LINEAR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
np_image = np.asarray(image).astype(np.float)
np_image = np_image / 255.0
temp = []
temp.append(np_image)

save_model_path = "saved_model.pb"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

graphs = []
sessions = []
input_tensors = []
output_tensors = []

for g in range(0,2):
    graphs.append(tf.Graph())
    with graphs[g].as_default() as graph:
        with open(save_model_path, 'rb') as f:
            graph_text = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_text)
            # print(type(graph_def.node),len(graph_def.node))
        # for i in range(0, len(graph_def.node)):
        #     graph_def.node[i].device = '/gpu:' + str(g)
            #print(n.name, n.op, n.device)
        with tf.device('/gpu:' + str(g)):
            tf.import_graph_def(graph_def, name='')
            sess = tf.Session(graph=graph, config=config)
            sessions.append(sess)
            print('gpu:',g)
            ip = graph.get_tensor_by_name('input_1:0')
            op = graph.get_tensor_by_name('conv2d_16/Sigmoid:0')
            input_tensors.append(ip)
            output_tensors.append(op)

# exit(0)

def run_test(idx):
    count = 5000
    for i in range(0,count):
        scores = sessions[idx].run(output_tensors[idx], feed_dict={input_tensors[idx]: temp})
        if i%100==0:
            print('iteration: ',i,' from device: /device:GPU:'+str(idx))
                #i, idx, scores.shape)

indices = [0,1]

# make the Pool of workers
pool = ThreadPool(2) 
results = pool.map(run_test, indices)

# close the pool and wait for the work to finish 
start = time.time()
pool.close() 
pool.join() 
end = time.time()
print(end - start)


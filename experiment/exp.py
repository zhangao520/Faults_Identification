import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys
import numpy as np
import glob
import os
import time
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfe.enable_eager_execution(config=config)

sys.path.append('..')
from model.model import FaultSegCNN

name_list = [
            '_0',
            '_dip_crossline',
            '_dip_curvature_shape_index',
            '_dip_inline',
            '_energy_gradient_crossline',
            '_energy_gradient_inline',
            '_inst_freq',
            '_sobel_filter_similarity']
dx=36
dy=31
dz=70
BATCH_SIZE = 32
ckpt_dir = '../ckpt/'
optimizer = tf.train.AdamOptimizer()
device = 'gpu:1'
model = FaultSegCNN(device=device,checkpoint_directory=ckpt_dir)
print(model.variables)
filenames = []
with open('../data/train.txt','r') as f:
    file = f.readlines()
    for name in file:
        filenames.append("{:0>4d}".format(int(name))+'.npy')
filenames = tf.constant(filenames)
    
def _parse_function(filename):
    cube = np.zeros((0,dx,dy,dz))
    for name in name_list:
        data = np.load(os.path.join('../fault_data/npdata/'+name[1:],filename.decode()))
        cube = np.concatenate((cube,[data]))
    cube = tf.convert_to_tensor(cube.transpose((1,2,3,0)),dtype=tf.float32)
    annotation = np.load(os.path.join('../fault_data/npdata/annotation/',filename.decode()))
    annotation = tf.convert_to_tensor(annotation,dtype=tf.int32)
    return cube,annotation
    
train_dataset = tf.data.Dataset.from_tensor_slices((filenames))
train_dataset = train_dataset.map(lambda filename :
                      tuple(tf.py_func(_parse_function,[filename],[tf.float32,tf.int32])))
train_dataset = train_dataset.shuffle(buffer_size=500)
train_dataset = train_dataset.batch(BATCH_SIZE)

filenames = []
with open('../data/test.txt','r') as f:
    file = f.readlines()
    for name in file:
        filenames.append("{:0>4d}".format(int(name))+'.npy')
filenames = tf.constant(filenames)
    
def _parse_function(filename):
    cube = np.zeros((0,dx,dy,dz))
    for name in name_list:
        data = np.load(os.path.join('../fault_data/npdata/'+name[1:],filename.decode()))
        cube = np.concatenate((cube,[data]))
    cube = tf.convert_to_tensor(cube.transpose((1,2,3,0)),dtype=tf.float32)
    annotation = np.load(os.path.join('../fault_data/npdata/annotation/',filename.decode()))
    annotation = tf.convert_to_tensor(annotation,dtype=tf.int32)
    return cube,annotation
    
eval_dataset = tf.data.Dataset.from_tensor_slices((filenames))
eval_dataset = eval_dataset.map(lambda filename :
                      tuple(tf.py_func(_parse_function,[filename],[tf.float32,tf.int32])))
eval_dataset = eval_dataset.batch(BATCH_SIZE)

# acc = model.compute_accuracy(train_dataset,load_model=True)
# print('Train accuracy:', acc.result().numpy())

# acc = model.compute_accuracy(eval_dataset,load_model=True)
# print('Eval accuracy:', acc.result().numpy())

# for cube,annotation in tfe.Iterator(train_dataset):
#     logits = model.predict(cube,training=False)
#     print(logits.shape)
#     break

# for cube,annotation in tfe.Iterator(eval_dataset):
#     #logits = model.predict(cube,training=False)
#     print(cube.shape,annotation.shape)
#     break   
model.fit(train_dataset,eval_dataset,optimizer,num_epochs=1)
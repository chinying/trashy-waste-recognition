# coding: utf-8
import caffe
import os
import sys
import numpy as np

caffe.set_mode_cpu()
caffe_root = "./cdata/"
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + 'Inception21k.prototxt'
model_weights = caffe_root + 'Inception21k.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
mu = np.load(caffe_root + 'ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# image = caffe.io.load_image(caffe_root + 'test-images/empty-pet-bottle-500x500.jpg')
image = caffe.io.load_image(caffe_root + 'test-images/banana-peel.jpg')
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

prob = output['softmax'][0]
top_10 = prob.argsort()[-10:][::-1]  # take top 10 results, this is in reverse order

def process_synset(s):
    tokens = s.split(" ")
    return (tokens[0], ' '.join(tokens[1:]))

# read synset
synsets_file = open(caffe_root + 'synset.txt')
synsets_lines = synsets_file.read().split("\n")
synsets_file.close()
synsets = list(map(lambda s: process_synset(s), synsets_lines))

# synset_ids = list(zip(range(100000), synsets))

for p in top_10:
    print(p, synsets[p])

# coding: utf-8
import caffe
import os
import sys
import numpy as np

caffe.set_mode_cpu()

class ImageClassifier(object):
    def __init__(self):
        caffe_root = "../cdata/"
        self.caffe_root = caffe_root
        sys.path.insert(0, caffe_root + 'python')

        model_def = caffe_root + 'Inception21k.prototxt'
        model_weights = caffe_root + 'Inception21k.caffemodel'

        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        mu = np.load(caffe_root + 'ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        # read synset
        synsets_file = open(caffe_root + 'synset.txt')
        synsets_lines = synsets_file.read().split("\n")
        synsets_file.close()
        self.synsets = list(map(lambda s: self.process_synset(s), synsets_lines))

    def process_synset(self, s):
        tokens = s.split(" ")
        return (tokens[0], ' '.join(tokens[1:]))


    def classify_image(self, filename):
        image = caffe.io.load_image(self.caffe_root + filename)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        output = self.net.forward()
        prob = output['softmax'][0]
        top_10 = prob.argsort()[-10:][::-1]  # take top 10 results, this is in reverse order

        for p in top_10:
            print(p, self.synsets[p])


if __name__ == "__main__":
    c = ImageClassifier()
    c.classify_image("test-images/nalgene.png")


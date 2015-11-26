"""
Convert AlexNet caffe model to theano model
"""

import os
import sys
import cPickle as pickle
import subprocess
import collections
import numpy as np
import theano
import theano.tensor as T
import caffe
import layers


def run():
    model_name = 'alexnet'
    directory_caffe = './caffemodel'
    directory_theano = './theanomodel'
    url_prototxt = 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt'
    url_caffemodel = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
    filename_prototxt = '%s/%s.prototxt' % (directory_caffe, model_name)
    filename_caffemodel = '%s/%s.caffemodel' % (directory_caffe, model_name)
    filename_theanomodel = '%s/%s.model' % (directory_theano, model_name)

    # download caffemodel
    print 'downloading caffemodel'
    if not os.path.exists(directory_caffe):
        os.mkdir(directory_caffe)
    if not os.path.exists(filename_prototxt):
        p = subprocess.Popen(('wget', url_prototxt, '-O', filename_prototxt))
        p.wait()
    if not os.path.exists(filename_caffemodel):
        p = subprocess.Popen((
            'wget',
            url_caffemodel,
            '-O',
            filename_caffemodel,
        ))
        p.wait()

    # load caffe model
    print 'loading caffe model'
    model_caffe = caffe.Net(filename_prototxt, filename_caffemodel, True)
    conv1_W = theano.shared(model_caffe.params['conv1'][0].data[:, :, ::-1, ::-1])
    conv2_W = theano.shared(model_caffe.params['conv2'][0].data[:, :, ::-1, ::-1])
    conv3_W = theano.shared(model_caffe.params['conv3'][0].data[:, :, ::-1, ::-1])
    conv4_W = theano.shared(model_caffe.params['conv4'][0].data[:, :, ::-1, ::-1])
    conv5_W = theano.shared(model_caffe.params['conv5'][0].data[:, :, ::-1, ::-1])
    conv1_b = theano.shared(model_caffe.params['conv1'][1].data.squeeze())
    conv2_b = theano.shared(model_caffe.params['conv2'][1].data.squeeze())
    conv3_b = theano.shared(model_caffe.params['conv3'][1].data.squeeze())
    conv4_b = theano.shared(model_caffe.params['conv4'][1].data.squeeze())
    conv5_b = theano.shared(model_caffe.params['conv5'][1].data.squeeze())
    fc6_W = theano.shared(model_caffe.params['fc6'][0].data.squeeze())
    fc7_W = theano.shared(model_caffe.params['fc7'][0].data.squeeze())
    fc8_W = theano.shared(model_caffe.params['fc8'][0].data.squeeze())
    fc6_b = theano.shared(model_caffe.params['fc6'][1].data.squeeze())
    fc7_b = theano.shared(model_caffe.params['fc7'][1].data.squeeze())
    fc8_b = theano.shared(model_caffe.params['fc8'][1].data.squeeze())

    # make theano model
    print 'building theano model'
    model_theano = collections.OrderedDict()
    model_theano['data'] = T.tensor4()
    model_theano['conv1'] = layers.convolution_layer(model_theano['data'], conv1_W, conv1_b, subsample=(4, 4))
    model_theano['relu1'] = layers.relu_layer(model_theano['conv1'])
    model_theano['norm1'] = layers.lrn_layer(model_theano['relu1'])
    model_theano['pool1'] = layers.pooling_layer(model_theano['norm1'])
    model_theano['conv2'] = layers.convolution_layer(model_theano['pool1'], conv2_W, conv2_b, border='same', group=2)
    model_theano['relu2'] = layers.relu_layer(model_theano['conv2'])
    model_theano['norm2'] = layers.lrn_layer(model_theano['relu2'])
    model_theano['pool2'] = layers.pooling_layer(model_theano['norm2'])
    model_theano['conv3'] = layers.convolution_layer(model_theano['pool2'], conv3_W, conv3_b, border='same')
    model_theano['relu3'] = layers.relu_layer(model_theano['conv3'])
    model_theano['conv4'] = layers.convolution_layer(model_theano['relu3'], conv4_W, conv4_b, border='same', group=2)
    model_theano['relu4'] = layers.relu_layer(model_theano['conv4'])
    model_theano['conv5'] = layers.convolution_layer(model_theano['relu4'], conv5_W, conv5_b, border='same', group=2)
    model_theano['relu5'] = layers.relu_layer(model_theano['conv5'])
    model_theano['pool5'] = layers.pooling_layer(model_theano['relu5'])
    model_theano['fc6'] = layers.inner_product_layer(model_theano['pool5'], fc6_W, fc6_b)
    model_theano['relu6'] = layers.relu_layer(model_theano['fc6'])
    model_theano['fc7'] = layers.inner_product_layer(model_theano['relu6'], fc7_W, fc7_b)
    model_theano['relu7'] = layers.relu_layer(model_theano['fc7'])
    model_theano['fc8'] = layers.inner_product_layer(model_theano['relu7'], fc8_W, fc8_b)
    model_theano['prob'] = layers.softmax_layer(model_theano['fc8'])

    # check
    print 'checking model'
    data = np.random.randn(*model_caffe.blobs['data'].data.shape)
    data = data.astype(np.float32) * 10
    model_caffe.blobs['data'].data[:] = data
    model_caffe.forward()
    theano_output = theano.function(
        [model_theano['data']],
        model_theano['prob'],
    )(data)
    error = (
        theano_output.squeeze() -
        model_caffe.blobs['prob'].data.squeeze()
    ).max()
    assert error < 1e-6

    # save
    print 'saving'
    if not os.path.exists(directory_theano):
        os.mkdir(directory_theano)
    sys.setrecursionlimit(100000)
    pickle.dump(
        model_theano, 
        open(filename_theanomodel, 'wb'),        
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    print 'done'

if __name__ == '__main__':
    run()

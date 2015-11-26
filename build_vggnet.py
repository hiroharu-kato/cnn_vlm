"""
Convert VGG-19 Caffe model to Theano model
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
    model_name = 'vggnet'
    directory_caffe = './caffemodel'
    directory_theano = './theanomodel'
    url_prototxt = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt'
    url_caffemodel = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
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
        p = subprocess.Popen(('wget', url_caffemodel, '-O', filename_caffemodel))
        p.wait()

    # load caffe model
    model_caffe = caffe.Net(filename_prototxt, filename_caffemodel, True)
    conv1_1_W = theano.shared(model_caffe.params['conv1_1'][0].data[:, :, ::-1, ::-1])
    conv1_2_W = theano.shared(model_caffe.params['conv1_2'][0].data[:, :, ::-1, ::-1])
    conv2_1_W = theano.shared(model_caffe.params['conv2_1'][0].data[:, :, ::-1, ::-1])
    conv2_2_W = theano.shared(model_caffe.params['conv2_2'][0].data[:, :, ::-1, ::-1])
    conv3_1_W = theano.shared(model_caffe.params['conv3_1'][0].data[:, :, ::-1, ::-1])
    conv3_2_W = theano.shared(model_caffe.params['conv3_2'][0].data[:, :, ::-1, ::-1])
    conv3_3_W = theano.shared(model_caffe.params['conv3_3'][0].data[:, :, ::-1, ::-1])
    conv3_4_W = theano.shared(model_caffe.params['conv3_4'][0].data[:, :, ::-1, ::-1])
    conv4_1_W = theano.shared(model_caffe.params['conv4_1'][0].data[:, :, ::-1, ::-1])
    conv4_2_W = theano.shared(model_caffe.params['conv4_2'][0].data[:, :, ::-1, ::-1])
    conv4_3_W = theano.shared(model_caffe.params['conv4_3'][0].data[:, :, ::-1, ::-1])
    conv4_4_W = theano.shared(model_caffe.params['conv4_4'][0].data[:, :, ::-1, ::-1])
    conv5_1_W = theano.shared(model_caffe.params['conv5_1'][0].data[:, :, ::-1, ::-1])
    conv5_2_W = theano.shared(model_caffe.params['conv5_2'][0].data[:, :, ::-1, ::-1])
    conv5_3_W = theano.shared(model_caffe.params['conv5_3'][0].data[:, :, ::-1, ::-1])
    conv5_4_W = theano.shared(model_caffe.params['conv5_4'][0].data[:, :, ::-1, ::-1])
    conv1_1_b = theano.shared(model_caffe.params['conv1_1'][1].data.squeeze())
    conv1_2_b = theano.shared(model_caffe.params['conv1_2'][1].data.squeeze())
    conv2_1_b = theano.shared(model_caffe.params['conv2_1'][1].data.squeeze())
    conv2_2_b = theano.shared(model_caffe.params['conv2_2'][1].data.squeeze())
    conv3_1_b = theano.shared(model_caffe.params['conv3_1'][1].data.squeeze())
    conv3_2_b = theano.shared(model_caffe.params['conv3_2'][1].data.squeeze())
    conv3_3_b = theano.shared(model_caffe.params['conv3_3'][1].data.squeeze())
    conv3_4_b = theano.shared(model_caffe.params['conv3_4'][1].data.squeeze())
    conv4_1_b = theano.shared(model_caffe.params['conv4_1'][1].data.squeeze())
    conv4_2_b = theano.shared(model_caffe.params['conv4_2'][1].data.squeeze())
    conv4_3_b = theano.shared(model_caffe.params['conv4_3'][1].data.squeeze())
    conv4_4_b = theano.shared(model_caffe.params['conv4_4'][1].data.squeeze())
    conv5_1_b = theano.shared(model_caffe.params['conv5_1'][1].data.squeeze())
    conv5_2_b = theano.shared(model_caffe.params['conv5_2'][1].data.squeeze())
    conv5_3_b = theano.shared(model_caffe.params['conv5_3'][1].data.squeeze())
    conv5_4_b = theano.shared(model_caffe.params['conv5_4'][1].data.squeeze())
    fc6_W = theano.shared(model_caffe.params['fc6'][0].data.squeeze())
    fc7_W = theano.shared(model_caffe.params['fc7'][0].data.squeeze())
    fc8_W = theano.shared(model_caffe.params['fc8'][0].data.squeeze())
    fc6_b = theano.shared(model_caffe.params['fc6'][1].data.squeeze())
    fc7_b = theano.shared(model_caffe.params['fc7'][1].data.squeeze())
    fc8_b = theano.shared(model_caffe.params['fc8'][1].data.squeeze())

    # make theano model
    model_theano = collections.OrderedDict()
    model_theano['data'] = T.tensor4()

    model_theano['conv1_1'] = layers.convolution_layer(model_theano['data'], conv1_1_W, conv1_1_b, border='same')
    model_theano['relu1_1'] = layers.relu_layer(model_theano['conv1_1'])
    model_theano['conv1_2'] = layers.convolution_layer(model_theano['relu1_1'], conv1_2_W, conv1_2_b, border='same')
    model_theano['relu1_2'] = layers.relu_layer(model_theano['conv1_2'])
    model_theano['pool1'] = layers.pooling_layer(model_theano['relu1_2'], size=(2, 2), stride=(2, 2))

    model_theano['conv2_1'] = layers.convolution_layer(model_theano['pool1'], conv2_1_W, conv2_1_b, border='same')
    model_theano['relu2_1'] = layers.relu_layer(model_theano['conv2_1'])
    model_theano['conv2_2'] = layers.convolution_layer(model_theano['relu2_1'], conv2_2_W, conv2_2_b, border='same')
    model_theano['relu2_2'] = layers.relu_layer(model_theano['conv2_2'])
    model_theano['pool2'] = layers.pooling_layer(model_theano['relu2_2'], size=(2, 2), stride=(2, 2))

    model_theano['conv3_1'] = layers.convolution_layer(model_theano['pool2'], conv3_1_W, conv3_1_b, border='same')
    model_theano['relu3_1'] = layers.relu_layer(model_theano['conv3_1'])
    model_theano['conv3_2'] = layers.convolution_layer(model_theano['relu3_1'], conv3_2_W, conv3_2_b, border='same')
    model_theano['relu3_2'] = layers.relu_layer(model_theano['conv3_2'])
    model_theano['conv3_3'] = layers.convolution_layer(model_theano['relu3_2'], conv3_3_W, conv3_3_b, border='same')
    model_theano['relu3_3'] = layers.relu_layer(model_theano['conv3_3'])
    model_theano['conv3_4'] = layers.convolution_layer(model_theano['relu3_3'], conv3_4_W, conv3_4_b, border='same')
    model_theano['relu3_4'] = layers.relu_layer(model_theano['conv3_4'])
    model_theano['pool3'] = layers.pooling_layer(model_theano['relu3_4'], size=(2, 2), stride=(2, 2))

    model_theano['conv4_1'] = layers.convolution_layer(model_theano['pool3'], conv4_1_W, conv4_1_b, border='same')
    model_theano['relu4_1'] = layers.relu_layer(model_theano['conv4_1'])
    model_theano['conv4_2'] = layers.convolution_layer(model_theano['relu4_1'], conv4_2_W, conv4_2_b, border='same')
    model_theano['relu4_2'] = layers.relu_layer(model_theano['conv4_2'])
    model_theano['conv4_3'] = layers.convolution_layer(model_theano['relu4_2'], conv4_3_W, conv4_3_b, border='same')
    model_theano['relu4_3'] = layers.relu_layer(model_theano['conv4_3'])
    model_theano['conv4_4'] = layers.convolution_layer(model_theano['relu4_3'], conv4_4_W, conv4_4_b, border='same')
    model_theano['relu4_4'] = layers.relu_layer(model_theano['conv4_4'])
    model_theano['pool4'] = layers.pooling_layer(model_theano['relu4_4'], size=(2, 2), stride=(2, 2))

    model_theano['conv5_1'] = layers.convolution_layer(model_theano['pool4'], conv5_1_W, conv5_1_b, border='same')
    model_theano['relu5_1'] = layers.relu_layer(model_theano['conv5_1'])
    model_theano['conv5_2'] = layers.convolution_layer(model_theano['relu5_1'], conv5_2_W, conv5_2_b, border='same')
    model_theano['relu5_2'] = layers.relu_layer(model_theano['conv5_2'])
    model_theano['conv5_3'] = layers.convolution_layer(model_theano['relu5_2'], conv5_3_W, conv5_3_b, border='same')
    model_theano['relu5_3'] = layers.relu_layer(model_theano['conv5_3'])
    model_theano['conv5_4'] = layers.convolution_layer(model_theano['relu5_3'], conv5_4_W, conv5_4_b, border='same')
    model_theano['relu5_4'] = layers.relu_layer(model_theano['conv5_4'])
    model_theano['pool5'] = layers.pooling_layer(model_theano['relu5_4'], size=(2, 2), stride=(2, 2))

    model_theano['fc6'] = layers.inner_product_layer(model_theano['pool5'], fc6_W, fc6_b)
    model_theano['relu6'] = layers.relu_layer(model_theano['fc6'])
    model_theano['fc7'] = layers.inner_product_layer(model_theano['relu6'], fc7_W, fc7_b)
    model_theano['relu7'] = layers.relu_layer(model_theano['fc7'])
    model_theano['fc8'] = layers.inner_product_layer(model_theano['relu7'], fc8_W, fc8_b)
    model_theano['prob'] = layers.softmax_layer(model_theano['fc8'])

    # check
    data = np.random.randn(*model_caffe.blobs['data'].data.shape).astype(np.float32) * 10
    model_caffe.blobs['data'].data[:] = data
    model_caffe.forward()
    theano_output = theano.function([model_theano['data']], model_theano['prob'])(data)
    error = (theano_output.squeeze()-model_caffe.blobs['prob'].data.squeeze()).max()
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

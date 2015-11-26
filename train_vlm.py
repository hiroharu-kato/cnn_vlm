"""
Train visual language model of all layers
"""

import sys
import glob
import cPickle as pickle
import collections
import numpy as np
import theano
import theano.tensor as T

import utils
import solvers
import layers


def two_layer_lstm(prefix, x, layer_size, hidden_size, axis, is_forward):
    tparams = collections.OrderedDict()
    tparams['%s_Wx1' % prefix] = theano.shared(0.01*np.random.randn(hidden_size*4, layer_size).astype(np.float32))
    tparams['%s_Wh1' % prefix] = theano.shared(0.01*np.random.randn(hidden_size*4, hidden_size).astype(np.float32))
    tparams['%s_b1' % prefix] = theano.shared(np.zeros(hidden_size*4).astype(np.float32))
    tparams['%s_Wx2' % prefix] = theano.shared(0.01*np.random.randn(hidden_size*4, hidden_size).astype(np.float32))
    tparams['%s_Wh2' % prefix] = theano.shared(0.01*np.random.randn(hidden_size*4, hidden_size).astype(np.float32))
    tparams['%s_b2' % prefix] = theano.shared(np.zeros(hidden_size*4).astype(np.float32))
    tparams['%s_Wx3' % prefix] = theano.shared(0.01*np.random.randn(layer_size, hidden_size).astype(np.float32))
    tparams['%s_b3' % prefix] = theano.shared(np.zeros(layer_size).astype(np.float32))

    l = x
    l = layers.recurrent_layer(
        l,
        tparams['%s_Wx1' % prefix],
        tparams['%s_Wh1' % prefix],
        tparams['%s_b1' % prefix],
        axis=axis,
        is_forward=is_forward,
    )
    l = layers.recurrent_layer(
        l,
        tparams['%s_Wx2' % prefix],
        tparams['%s_Wh2' % prefix],
        tparams['%s_b2' % prefix],
        axis=axis,
        is_forward=is_forward,
    )
    l = layers.linear_layer(
        l,
        tparams['%s_Wx3' % prefix],
        tparams['%s_b3' % prefix],
    )
    return l, tparams


def apply_pca(pca, x, num_components):
    w = pca.components_[:num_components][:, :, None, None]
    b = pca.mean_[None, :, None, None]
    x = theano.sandbox.cuda.dnn.dnn_conv(x - b, w)
    return x


def apply_standard_normalization(mean_std, x):
    return (
        (x - mean_std['mean'][None, :, None, None]) /
        mean_std['std'][None, :, None, None]
    )


def build_model(model_name, layer_names, layer_sizes):
    directory = './theanomodel'
    filename_model = '%s/%s.model' % (directory, model_name)
    filename_mean_std = '%s/%s_mean_std_%s.pkl'
    filename_pca = '%s/%s_pca_%s.pkl'

    # load model
    model = pickle.load(open(filename_model))

    # build model
    tparams = collections.OrderedDict()
    for layer_name, layer_size in zip(layer_names, layer_sizes):
        # input
        x = model[layer_name]

        # standard normalization
        mean_std = pickle.load(open(filename_mean_std % (directory, model_name, layer_name)))
        x = apply_standard_normalization(mean_std, x)

        # apply PCA
        pca = pickle.load(open(filename_pca % (directory, model_name, layer_name)))
        num_components = (pca.components_.shape[0] + 1) / 2
        x = apply_pca(pca, x, num_components)

        # append RNNs
        layer_size = num_components
        hidden_size = layer_size
        l13, t13 = two_layer_lstm('%s_2f' % layer_name, x, layer_size, hidden_size, 2, True)
        l23, t23 = two_layer_lstm('%s_2b' % layer_name, x, layer_size, hidden_size, 2, False)
        l33, t33 = two_layer_lstm('%s_3f' % layer_name, x, layer_size, hidden_size, 3, True)
        l43, t43 = two_layer_lstm('%s_3b' % layer_name, x, layer_size, hidden_size, 3, False)
        tparams.update(t13)
        tparams.update(t23)
        tparams.update(t33)
        tparams.update(t43)

        # prediction error map
        l14 = (l13[:, :, :-1, :] - x[:, :, 1:, :])**2
        l24 = (l23[:, :, 1:, :] - x[:, :, :-1, :])**2
        l34 = (l33[:, :, :, :-1] - x[:, :, :, 1:])**2
        l44 = (l43[:, :, :, 1:] - x[:, :, :, :-1])**2
        l15 = l14 * T.arange(0, 1, 1./l14.shape[2])[None, None, :l14.shape[2], None]
        l25 = l24 * T.arange(0, 1, 1./l24.shape[2])[None, None, l24.shape[2]-1::-1, None]
        l35 = l34 * T.arange(0, 1, 1./l34.shape[3])[None, None, None, :l34.shape[3]]
        l45 = l44 * T.arange(0, 1, 1./l44.shape[3])[None, None, None, l44.shape[3]-1::-1]

        model['loss_lm_%s' % layer_name] = (l15.mean() + l25.mean() + l35.mean() + l45.mean()) / 4
        model['loss_lm_%s_2f' % layer_name] = l15
        model['loss_lm_%s_2b' % layer_name] = l25
        model['loss_lm_%s_3f' % layer_name] = l35
        model['loss_lm_%s_3b' % layer_name] = l45
        model['loss_lm_%s_map' % layer_name] = (
            l15.mean(1)[:, :, :-1] +
            l25.mean(1)[:, :, :-1] +
            l35.mean(1)[:, :-1, :] +
            l45.mean(1)[:, :-1, :]
        )

    # sum up prediction error maps
    model['loss_lm'] = 0
    for layer_name in layer_names:
        model['loss_lm'] += model['loss_lm_%s' % layer_name] / len(layer_names)

    return model, tparams


def run(
    model_name='alexnet',
    layer_names=['data', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
    layer_sizes=[3, 96, 256, 384, 384, 256],
    batch_size=16,
    lr=1e+1,
    max_iter=20000,
    image_size=(227, 227),
    save_interval=1000,
):
    directory = './theanomodel'
    filename_train = './image/valid/*.JPEG'
    filename_save = '%s/%s_vlm_%s.pkl' % (directory, model_name, '_'.join(layer_names))
    lr_reduce_factor = 0.1
    lr_reduce_interval = 5000

    # load model
    img_mean = utils.load_mean_image()
    model, tparams = build_model(model_name, layer_names, layer_sizes)

    # build solver
    solver = solvers.SGD(
        model['loss_lm'],
        [model['data']],
        tparams.values(),
        lr=lr,
    )

    # get filename
    filename_train = glob.glob(filename_train)

    # train
    loss = []
    for iter_ in range(max_iter+1):
        # load images
        imgs = []
        for filename in np.random.choice(filename_train, batch_size):
            img = utils.load_image(filename, img_mean)
            img = utils.clip_image(img, image_size)
            imgs.append(img)
        imgs = np.array(imgs)

        # update
        l = solver.update(imgs)
        loss.append(l)
        print 'training VLM: %d / %d %f' % (iter_, max_iter, l)

        # reduce learning rate
        if iter_ != 0 and iter_ % lr_reduce_interval == 0:
            solver.reset()
            solver.lr.set_value(
                solver.lr.get_value() *
                np.array(lr_reduce_factor).astype(np.float32)
            )

        # save
        if iter_ % save_interval == 0:
            print 'average loss:', np.mean(loss[-save_interval:])
            sys.setrecursionlimit(100000)
            pickle.dump(
                model, 
                open(filename_save, 'wb'),                  
                protocol=pickle.HIGHEST_PROTOCOL,
            )


if __name__ == '__main__':
    run()

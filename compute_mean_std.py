"""
Compute mean and std of feature maps
"""

import glob
import cPickle as pickle
import numpy as np
import theano
import utils


def get_function_mean_var(model, layer_names):
    """
    Return a function which computes mean and mean**2 of each layer
    """
    outputs = []
    for layer_name in layer_names:
        outputs += [
            model[layer_name].mean(0).mean(1).mean(1),
            (model[layer_name]**2).mean(0).mean(1).mean(1),
        ]
    return theano.function([model['data']], outputs)


def run(
    model_name='alexnet',
    layer_names=[
        'data',
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
        'fc6', 'fc7', 'fc8'
    ],
    image_size=(227, 227),
):
    directory = './theanomodel'
    filename_save = '%s/%s_mean_std_%s.pkl'
    filename_train = './image/valid/*.JPEG'
    filename_model = '%s/%s.model' % (directory, model_name)

    # load model
    img_mean = utils.load_mean_image()
    model = pickle.load(open(filename_model))

    # get filename
    filename_train = glob.glob(filename_train)

    # get function to compute mean and mean**2
    func = get_function_mean_var(model, layer_names)

    # for all images
    means = {layer_name: [] for layer_name in layer_names}
    vars_ = {layer_name: [] for layer_name in layer_names}
    for i, fn in enumerate(filename_train):
        print 'computing mean: %d / %d' % (i, len(filename_train))
        img = utils.load_image(fn, img_mean)
        img = utils.clip_image(img, image_size)
        mvs = func(img[None, :, :, :])
        for j, layer_name in enumerate(layer_names):
            means[layer_name].append(mvs[j*2])
            vars_[layer_name].append(mvs[j*2+1])

    # save
    for layer_name in layer_names:
        mean = np.vstack(means[layer_name]).mean(0)
        std = (np.vstack(vars_[layer_name]).mean(0) - mean**2)**0.5
        data = {
            'mean': mean,
            'std': std,
        }
        filename = filename_save % (directory, model_name, layer_name)
        pickle.dump(
            data, 
            open(filename, 'wb'),                  
            protocol=pickle.HIGHEST_PROTOCOL,
        )

if __name__ == '__main__':
    run()

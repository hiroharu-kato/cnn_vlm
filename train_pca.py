"""
Train PCA of feature maps
"""

import glob
import cPickle as pickle
import numpy as np
import sklearn.decomposition
import theano
import theano.tensor as T
import utils


def get_function_sample(model, layer_names, num_samples):
    # build function to sample num_samples samples from layer_name of model
    srng = T.shared_randomstreams.RandomStreams(0)
    outputs = []
    for layer_name in layer_names:
        output = model[layer_name]
        output = output.dimshuffle((0, 2, 3, 1))
        output = output.reshape((
            output.shape[0]*output.shape[1]*output.shape[2],
            output.shape[3])
        )
        indices = srng.random_integers(
            low=0,
            high=output.shape[0]-1,
            size=num_samples,
            ndim=1,
        )
        output = output[indices]
        outputs.append(output)
    return theano.function([model['data']], outputs)


def run(
    model_name='alexnet',
    layer_names=['data', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
    image_size=(227, 227),
    num_samples=100000,
):
    directory = './theanomodel'
    filename_save = '%s/%s_pca_%s.pkl'
    filename_mean_std = '%s/%s_mean_std_%s.pkl'
    filename_train = './image/valid/*.JPEG'
    filename_model = '%s/%s.model' % (directory, model_name)

    # load model
    img_mean = utils.load_mean_image()
    model = pickle.load(open(filename_model))

    # get filename
    filename_train = glob.glob(filename_train)
    num_samples_per_image = num_samples / len(filename_train) + 1

    # get function to sample
    func = get_function_sample(model, layer_names, num_samples_per_image)

    # sample
    samples = {layer_name: [] for layer_name in layer_names}
    for i, filename in enumerate(filename_train):
        print 'training PCA: %d / %d' % (i, len(filename_train))
        img = utils.load_image(filename, img_mean)
        img = utils.clip_image(img, image_size)
        s = func(img[None, :, :, :])
        for j, layer_name in enumerate(layer_names):
            samples[layer_name].append(s[j])

    # PCA
    for layer_name in layer_names:
        filename = filename_mean_std % (directory, model_name, layer_name)
        mean_std = pickle.load(open(filename))
        samples[layer_name] = np.vstack(samples[layer_name])
        samples[layer_name] = (
            (samples[layer_name] - mean_std['mean'][None, :]) /
            mean_std['std'][None, :]
        )
        pca = sklearn.decomposition.PCA(whiten=False)
        pca.fit(samples[layer_name])
        filename = filename_save % (directory, model_name, layer_name)
        pickle.dump(
            pca,
            open(filename, 'wb'),       
            protocol=pickle.HIGHEST_PROTOCOL,
        )

if __name__ == '__main__':
    run()

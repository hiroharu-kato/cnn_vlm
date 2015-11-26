import sys
import pickle
import numpy as np
import scipy.misc
import scipy.ndimage
import theano
import utils


def run(filename_in, filename_out):
    # parameters of CNN-VLM model
    model_name = 'vggnet'
    layer_names = [
        'data',
        'conv1_2',
        'conv2_2',
        'conv3_4',
        'conv4_4',
        'conv5_4',
    ]
    directory = './theanomodel'
    filename = (
        '%s/%s_vlm_%s.pkl' %
        (directory, model_name, '_'.join(layer_names))
    )

    # parameters of saliency map
    layer_target = 'conv5_4'
    pool_size = 2**4
    pow_ = 0.5
    sigma = 0.03

    # load model
    model = pickle.load(open(filename))
    func = theano.function(
        [model['data']],
        model['loss_lm_%s_map' % layer_target]
    )
    img_mean = utils.load_mean_image()

    # load image
    img1 = utils.load_image(filename_in, img_mean)

    # compute unnaturalness map
    img2 = func(img1[None, :, :, :]).squeeze()

    # pow
    img2 = img2 ** pow_

    # resize to original size
    img2 = rescale_image(img2, pool_size)
    img2 = pad_image(img2, img1.shape[1:])

    # blur
    img2 = scipy.ndimage.filters.gaussian_filter(
        img2,
        (sigma*img2.shape[0], sigma*img2.shape[1]),
    )

    # normalize
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    img2 = (img2 * 255).astype(np.uint8)

    # save
    scipy.misc.imsave(filename_out, img2)


def rescale_image(img, scale):
    return scipy.misc.imresize(img, (scale*img.shape[0], scale*img.shape[1]))


def pad_image(img, shape):
    pad_ud = shape[0] - img.shape[0]
    pad_lr = shape[1] - img.shape[1]
    pad_u = pad_ud / 2
    pad_d = pad_ud - pad_u
    pad_l = pad_lr / 2
    pad_r = pad_lr - pad_l
    img = np.vstack((
        np.zeros((pad_u, img.shape[1])),
        img,
        np.zeros((pad_d, img.shape[1])),
    ))
    img = np.hstack((
        np.zeros((img.shape[0], pad_l)),
        img,
        np.zeros((img.shape[0], pad_r)),
    ))
    return img


if __name__ == '__main__':
    filename_in = sys.argv[1]
    filename_out = sys.argv[2]
    run(filename_in, filename_out)

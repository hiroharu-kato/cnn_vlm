import numpy as np
import scipy.misc


def load_mean_image():
    filename = './theanomodel/imagenet_mean.npy'
    return np.load(filename).squeeze().astype(np.float32)


def load_image(filename, img_mean):
    # load
    img = scipy.misc.imread(filename)

    # RGB to BGR
    if img.ndim == 3:
        img = img[:, :, :3]
        img = img[:, :, ::-1]
    else:
        img = np.tile(img[:, :, None], (1, 1, 3))
    img = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)

    # subtract mean
    if img.shape == img_mean.shape:
        img = img - img_mean
    else:
        img = img - img_mean.mean(1).mean(1)[:, None, None]

    return img


def clip_image(img, size):
    clip_h1 = (img.shape[1] - size[0]) / 2
    clip_h2 = (img.shape[1] - size[0] + 1) / 2
    clip_w1 = (img.shape[2] - size[1]) / 2
    clip_w2 = (img.shape[2] - size[1] + 1) / 2
    return img[:, clip_h1:-clip_h2, clip_w1:-clip_w2]

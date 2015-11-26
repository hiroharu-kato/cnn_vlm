import sys
import pickle
import numpy as np
import scipy.misc
import scipy.optimize
import theano
import utils


def reset_solver(solver):
    for g in solver.gshared:
        g.set_value(np.zeros(g.get_value().shape).astype(np.float32))
    for v in solver.vshared:
        v.set_value(np.zeros(v.get_value().shape).astype(np.float32))


def run(filename_in, filename_out):
    # parameters
    lambda_layer = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4]
    lambda_loss = [1, 10]
    learning_rate = 2**19
    momentum = 0.9
    max_iter = 1000

    model_name = 'alexnet'
    image_size = (227, 227)
    layer_names = [
        'data',
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
    ]
    directory = './theanomodel'
    filename_reconstructor = (
        '%s/%s_vlm_%s_reconstructor.pkl' %
        (directory, model_name, '_'.join(layer_names))
    )
    filename_cnn = (
        '%s/%s.model' % (directory, model_name)
    )
    filename_mean_std = (
        '%s/%s_mean_std_data.pkl' %
        (directory, model_name)
    )

    # compute target feature
    cnn = pickle.load(open(filename_cnn))
    img_mean = utils.load_mean_image()
    func = theano.function([cnn['data']], cnn['fc8'])
    img = utils.load_image(filename_in, img_mean)
    target = func(img[None, :, :, :]).flatten()

    # load reconstructor
    print 'loading reconstructor (which takes several tens of minutes)'
    reconstructor = pickle.load(open(filename_reconstructor))

    # set hyper-parameters
    reconstructor['target'].set_value(target.astype(np.float32))
    reconstructor['lambda_layer'].set_value(
        np.array(lambda_layer).astype(np.float32)
    )
    reconstructor['lambda_loss'].set_value(
        np.array(lambda_loss).astype(np.float32)
    )
    reconstructor['solver'].lr.set_value(learning_rate)
    reconstructor['solver'].momentum.set_value(momentum)

    # init solver
    ms = pickle.load(open(filename_mean_std))
    x_init = (
        np.random.randn(1, 3, image_size[0], image_size[1]) *
        ms['std'][None, :, None, None] +
        ms['mean'][None, :, None, None]
    ).astype(np.float32)
    reconstructor['x'].set_value(x_init)
    reset_solver(reconstructor['solver'])

    # optimize
    for i in range(max_iter):
        loss = reconstructor['solver'].update()
        print 'iter %d / %d, loss %f' % (i, max_iter, loss)

    # save
    img = reconstructor['x'].get_value().squeeze()
    img = (img+img_mean.mean(1).mean(1)[:, None, None])
    img = img.swapaxes(0, 2).swapaxes(0, 1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(filename_out, img[:, :, ::-1])


if __name__ == '__main__':
    filename_in = sys.argv[1]
    filename_out = sys.argv[2]
    run(filename_in, filename_out)

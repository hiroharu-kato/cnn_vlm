import sys
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import solvers


def run(
    model_name='alexnet',
    layer_names=['data', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
    layer_target=['conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
    image_size=(227, 227),
    target_layer='fc8',
):
    # paramters
    dim_target = 1000
    directory = './theanomodel'
    filename_model = (
        '%s/%s_vlm_%s.pkl' %
        (directory, model_name, '_'.join(layer_names))
    )
    filename_save = (
        '%s/%s_vlm_%s_reconstructor.pkl' %
        (directory, model_name, '_'.join(layer_names))
    )

    # hyper-parameters
    lambda_loss = theano.shared(np.zeros(2))
    lambda_layer = theano.shared(np.zeros(len(layer_target)))

    #
    model = pickle.load(open(filename_model))

    # input
    x_init = np.zeros((1, 3, image_size[0], image_size[1])).astype(np.float32)
    x = theano.shared(x_init, borrow=False)
    xx = T.tensor4()
    xx_shared = theano.shared(np.zeros(x_init.shape).astype(np.float32))
    T.Apply(T.add, [x + xx], [model['data']])

    # loss_target
    target_shared = theano.shared(np.zeros(dim_target).astype(np.float32))
    mean_std = pickle.load(open(
        '%s/%s_mean_std_%s.pkl' %
        (directory, model_name, target_layer)
    ))
    loss_target = ((
        (model[target_layer] - target_shared[None, :, None, None]) /
        mean_std['std'][None, :, None, None]
    )**2).mean()

    # loss_lm
    loss_lm = T.sum([
        lambda_layer[i] * model['loss_lm_%s' % layer_target[i]]
        for i in range(len(layer_target))
    ])

    # total loss
    loss = (
        lambda_loss[0] * loss_target +
        lambda_loss[1] * loss_lm
    )

    # functions
    solver = solvers.SGD(loss, [], [x], givens={xx: xx_shared})

    # save
    data = {
        'solver': solver,
        'x': x,
        'lambda_loss': lambda_loss,
        'lambda_layer': lambda_layer,
        'target': target_shared,
    }
    sys.setrecursionlimit(100000)
    pickle.dump(
        data,
        open(filename_save, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == '__main__':
    run()

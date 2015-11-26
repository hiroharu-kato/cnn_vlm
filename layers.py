"""

"""

import theano
import theano.tensor as T
import theano.sandbox.cuda.dnn as cudnn


def lrn_layer(tensor, n=5, alpha=0.0001, beta=0.75, k=1.):
    """
    from pylearn2
    """
    tensor = tensor.dimshuffle((1, 2, 3, 0))
    half = n // 2
    sq = T.sqr(tensor)
    ch, r, c, b = tensor.shape
    extra_channels = T.alloc(0., ch + 2*half, r, c, b)
    sq = T.set_subtensor(extra_channels[half:half+ch, :, :, :], sq)
    scale = k
    for i in xrange(n):
        scale += alpha / n * sq[i:i+ch, :, :, :]
    scale = scale ** beta
    return (tensor / scale).dimshuffle((3, 0, 1, 2))


def convolution_layer(tensor, W, b, subsample=(1, 1), border='valid', group=1):
    W_shape = W.get_value().shape

    if border == 'same':
        pad = (W_shape[-2] / 2, W_shape[-1] / 2)
    else:
        pad = (0, 0)

    if group == 1:
        tensor = theano.sandbox.cuda.dnn.dnn_conv(
            tensor,
            W,
            subsample=subsample,
            border_mode=pad,
        )
    else:
        s = T.repeat(tensor.shape[1]/group, group)
        outputs = []
        for i, t in enumerate(T.split(tensor, s, group, axis=1)):
            W_ = W[i*W_shape[0]/group:(i+1)*W_shape[0]/group]
            outputs.append(theano.sandbox.cuda.dnn.dnn_conv(
                t,
                W_,
                subsample=subsample,
                border_mode=pad,
            ))
        tensor = T.concatenate(outputs, axis=1)
    tensor = tensor + b[None, :, None, None]

    return tensor


def inner_product_layer(tensor, W, b):
    tensor = tensor.reshape((
        tensor.shape[0],
        tensor.shape[1]*tensor.shape[2]*tensor.shape[3]
    ))
    tensor = T.dot(tensor, W.transpose())
    tensor = tensor + b
    tensor = tensor[:, :, None, None]
    return tensor


def linear_layer(tensor, W, b):
    tensor = cudnn.dnn_conv(
        tensor,
        W[:, :, None, None],
    )
    tensor = tensor + b[None, :, None, None]
    return tensor


def softmax_layer(tensor):
    n, c, h, w = tensor.shape
    tensor = tensor.reshape((n, c*h*w))
    tensor = T.nnet.softmax(tensor)
    tensor = tensor.reshape((n, c, h, w))
    return tensor


def relu_layer(tensor):
    return T.nnet.relu(tensor)


def pooling_layer(tensor, size=(3, 3), stride=(2, 2)):
    return theano.sandbox.cuda.dnn.dnn_pool(tensor, ws=size, stride=stride)


def recurrent_layer(tensor, Wx, Wh, b, axis=2, is_forward=True):
    # function for scan
    # x_: [num, channels, width]
    # hx_: [num, dim_output*n, width]
    # h_: [num, dim_output, width]
    def _step_lstm(x_, hx_, h_, c_):
        hh_ = cudnn.dnn_conv(
            h_[:, :, :, None],
            Wh[:, :, None, None],
        ).squeeze()
        preact = hx_ + hh_

        i = T.nnet.sigmoid(preact[:, dim_output*0:dim_output*1])
        f = T.nnet.sigmoid(preact[:, dim_output*1:dim_output*2])
        o = T.nnet.sigmoid(preact[:, dim_output*2:dim_output*3])
        c = T.tanh(preact[:, dim_output*3:dim_output*4])
        c = f * c_ + i * c
        h = o * T.tanh(c)

        return h, c

    # init parameters
    dim_output = Wx.get_value().shape[0] / 4

    # hx
    hx = cudnn.dnn_conv(
        tensor,
        Wx[:, :, None, None],
    )
    hx = hx + b[None, :, None, None]

    # transform
    # input: [num, channels, height, width]
    #     -> [height, num, channels, width]
    if axis == 2:
        s = (2, 0, 1, 3)
    elif axis == 3:
        s = (3, 0, 1, 2)
    tensor = tensor.dimshuffle(s)
    hx = hx.dimshuffle(s)
    if not is_forward:
        tensor = tensor[::-1]
        hx = hx[::-1]

    # loop
    tensor = theano.scan(
        _step_lstm,
        sequences=[tensor, hx],
        outputs_info=[
            T.zeros((
                tensor.shape[1],
                dim_output,
                tensor.shape[3]
            )),
            T.zeros((
                tensor.shape[1],
                dim_output,
                tensor.shape[3]
            )),
        ],
    )[0][0]

    # invert transform
    if not is_forward:
        tensor = tensor[::-1]
    if axis == 2:
        s = (1, 2, 0, 3)
    elif axis == 3:
        s = (1, 2, 3, 0)
    tensor = tensor.dimshuffle(s)

    return tensor

import numpy as np
import theano
import theano.tensor as T
import collections


class SGD(object):
    def __init__(
        self,
        loss,
        inputs,
        tparams,
        givens=None,
        lr=0.1,
        momentum=0.9,
    ):
        dtype = theano.config.floatX
        self.lr = theano.shared(np.array(lr).astype(dtype))
        self.momentum = theano.shared(np.array(momentum).astype(dtype))
        self.tparams = tparams
        self.gshared = [
            theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
            for p in tparams
        ]
        self.vshared = [
            theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
            for p in tparams
        ]

        # Function that computes gradients
        grads = T.grad(loss, wrt=tparams)
        gsup = collections.OrderedDict()
        for gs, g in zip(self.gshared, grads):
            gsup[gs] = g
        self.f_grad_shared = theano.function(
            inputs,
            loss,
            updates=gsup,
            givens=givens,
        )

        # Function that updates the weights from the previously computed
        # gradient.
        pup = collections.OrderedDict()
        for p, g, v in zip(tparams, self.gshared, self.vshared):
            pup[p] = p + self.momentum * v - self.lr * g
            pup[v] = self.momentum * v - self.lr * g

        self.f_update = theano.function([], [], updates=pup)
        self.f_loss = theano.function(inputs, loss, givens=givens)

    def reset(self):
        for i in range(len(self.gshared)):
            self.gshared[i].set_value(0*self.gshared[i].get_value())
        for i in range(len(self.vshared)):
            self.vshared[i].set_value(0*self.vshared[i].get_value())

    def update(self, *inputs):
        loss = self.f_grad_shared(*inputs)
        self.f_update()
        return loss

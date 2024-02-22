from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """


class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """

    def __init__(self, net, lr=1e-4):
        self.net = net  # model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """

    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)

        #### MLP ####
        if (
            not hasattr(self.net, "preprocess")
            and not hasattr(self.net, "rnn")
            and not hasattr(self.net, "postprocess")
        ):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """


class SGD(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        for n, dv in layer.grads.items():
            v = self.momentum * self.velocity.get(n, 0.0) - self.lr * dv
            layer.params[n] += v
            self.velocity[n] = v
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for n, dv in layer.grads.items():
            decay_ave = self.decay * self.cache.get(n, 0.0) + (1 - self.decay) * dv**2
            f_theta = (self.lr * dv) / np.sqrt(decay_ave + self.eps)
            layer.params[n] -= f_theta
            self.cache[n] = decay_ave
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        t = self.t + 1
        for n, dv in layer.grads.items():
            mt = self.beta1 * self.mt.get(n, 0.0) + (1 - self.beta1) * dv
            vt = self.beta2 * self.vt.get(n, 0.0) + (1 - self.beta2) * dv**2
            bias_mt = mt / (1 - self.beta1**t)
            bias_vt = vt / (1 - self.beta2**t)
            layer.params[n] -= (self.lr * bias_mt) / (np.sqrt(bias_vt) + self.eps)

            self.mt[n] = mt
            self.vt[n] = vt
        self.t = t
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

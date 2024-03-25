from builtins import object
import torch
import numpy as np
from typing import List

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    [*conv* - relu - 2x2 max pool] - [*affine* - relu] - [*affine*] -> softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network.                                                                 #
        # - Weights should be initialized from a Gaussian centered at 0.0          #
        #    with standard deviation equal to weight_scale;                        #
        # - biases should be initialized to zero.                                  #
        # - All weights and biases should be stored in the dictionary self.params. #
        # - Store weights and biases for the:                                      #
        #   - convolutional layer using the keys 'W1' and 'b1'                     #
        #   - use keys 'W2' and 'b2' for the weights and biases of the             #
        #      hidden affine layer                                                 #
        #   - and keys 'W3' and 'b3' for the weights and biases of the             #
        #      output affine layer.                                                #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        C, H, W = input_dim
        F, k = num_filters, filter_size

        mu, sd = 0.0, weight_scale
        self.params["W1"] = np.random.normal(loc=mu, scale=sd, size=(F, C, k, k))
        self.params["b1"] = np.zeros((F,))

        HH, WW = H // 2, W // 2
        self.params["W2"] = np.random.normal(loc=mu, scale=sd, size=(F * HH * WW, hidden_dim))
        self.params["b2"] = np.zeros((hidden_dim,))

        self.params["W3"] = np.random.normal(loc=mu, scale=sd, size=(hidden_dim, num_classes))
        self.params["b3"] = np.zeros((num_classes,))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def to(self, device):
      """
      Moves the model's parameters to specified device.

      :param device: "cpu" or "cuda"
      """
      for key in self.params:
        tensor = torch.as_tensor(self.params[key], dtype=torch.float32)
        self.params[key] = tensor.to(device)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs566/fast_layers.py and  #
        # cs566/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        o1, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        o2, ar_cache = affine_relu_forward(o1, W2, b2)
        o3, af_cache = affine_forward(o2, W3, b3)
        scores = o3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Backprop through the affine-relu layer
        """
        Math time! We first derive the following...
         L = softmax_loss(o3)
         dL/do3 = dout

         o3 = W3 @ o2 + b3 (affine)
         dL/dW3 = dL/do3 * do3/dW3 = dout * o2
         dL/db3 = dL/do3 * do3/db3 = dout * 1
         dL/o2 = dL/do3 * do3/o2 = dout * W3
         do2 = dout * W3
        
         o2 = affine_relu(W2, b2, o1)
         dL/dW2 = dL/o2 * o2/dW2 = do2 * affine_relu_backward()
         dL/db2 = dL/o2 * o2/db2 = do2 * affine_relu_backward()
         dL/do1 = dL/o2 * o2/do1 = do2 * affine_relu_backward()
         do1 = do2 * affine_relu_backward()

         o1 = conv_relu_pool_forward(W1, b1, X)
         dL/dW1 = dL/do1 * do1/dW1 = do1 * conv_relu_pool_backward()
         dL/db1 = dL/do1 * do1/db1 = do1 * conv_relu_pool_backward()
         dL/dX = dL/do1 * do1/dX = do1 * conv_relu_pool_backward()
        
        With regards to the transposes, we know that
         W3 @ o2 = out => W3 = o2.T @ dout && o2 = dout @ W3.T
         W2 @ o1 = o2  => W2 = o1.T @ o2  && do1 = o2  @ W2.T
         W1 @ X  = a1  => W1 = X.T  @ do1  && dX  = da1  @ W1.T (but this is handled)

        Code:
         grads["W3"] = o2.T @ dout
         grads["b3"] = np.sum(dout, axis=0)
         o2 = dout @ W3.T

         N, F, HH, WW = o1.shape
         grads["W2"] = o1.reshape(N, F * HH * WW).T @ o2
         grads["b2"] = np.sum(o2, axis=0)
         do1 = (o2 @ W2.T).reshape(N, F, HH, WW)

         dx, dW1, db1 = conv_relu_pool_backward(do1, crp_cache)
         grads["W1"] = dW1
         grads["b1"] = db1
        
        Update:
         Turns out I could just call the backward functions, and this is
         exactly what I was doing above, but including regularization and
         other things to increase precision.

         It was a good learning lesson!
        """
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * 0.5 * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        do3, dW3, db3 = affine_backward(dscores, af_cache)
        do2, dW2, db2 = affine_relu_backward(do3, ar_cache)
        do1, dW1, db1 = conv_relu_pool_backward(do2, crp_cache)

        dW3 += self.reg * W3
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        
        grads = {
          "W1": dW1, "W2": dW2, "W3": dW3,
          "b1": db1, "b2": db2, "b3": db3,
        }
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.initializer import (calc_normal_std_he_forward, 
                                ConstantInitializer, NormalInitializer)

import numpy as np

def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()

def act_bn_linear(h, maps, act=None, test=False, name=""):
    h = PF.affine(h, maps, with_bias=False, name=name)
    h = PF.batch_normalization(h, batch_stat=not test, name=name)
    if act:
        h = F.relu(h)
    return h

def mlp(image, maps=256, ncls=10, test=False):
    image /= 255.
    with nn.parameter_scope("ref"):
        h = act_bn_linear(image, maps, test, name="fc0")
        h = act_bn_linear(h, maps, test, name="fc1")
        h = act_bn_linear(h, maps, test, name="fc2")
        pred = PF.affine(h, ncls, name="fc")
    return pred

def act_bn_conv(h, maps, act=None, test=False, name=""):
    h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), with_bias=False, name=name)
    h = PF.batch_normalization(h, batch_stat=not test, name=name)
    if act:
        h = F.relu(h)
    return h

def cnn(image,  maps=128, ncls=10, test=False):
    image /= 255.
    with nn.parameter_scope("ref"):
        h = act_bn_conv(image, maps, test, name="conv0")
        h = F.max_pooling(h, (2, 2))  # 28x28 -> 14x14
        h = act_bn_conv(h, maps, test, name="conv1")
        h = F.max_pooling(h, (2, 2))  # 14x14 -> 7x7
        h = act_bn_conv(h, maps, test, name="conv2")
        h = F.average_pooling(h, (7, 7))  # 7x7 -> 1x1
        pred = PF.affine(h, ncls, name="fc")
    return pred

# Decouple
def decouple(h):
    """Redundant decouple.
    x_input is actually not needed.
    """
    h_d = h
    h.need_grad = True
    h_copy = nn.Variable(h.shape, need_grad=True)
    x_input = nn.Variable(h.shape)
    g_label = nn.Variable(h.shape)

    # Memory sharing
    h_copy.data = h.data
    x_input.data = h.data
    g_label.data = h_copy.grad
    
    # Dot not overwrite the necessities with usnig clear_buffer
    h.persistent = True
    h_d.persistent = True
    h_copy.persistent = True
    x_input.persistent = True
    g_label.persistent = True

    return h_d, h_copy, x_input, g_label

def mlp_gradient_synthesizer(x, y=None, test=False):
    maps = x.shape[1]
    if y is not None:
        h = F.one_hot(y, (10, ))
        h = F.concatenate(*[x, y], axis=1)
    else:
        h = x
    with nn.parameter_scope("gs"):
        h = act_bn_linear(h, maps, test, name="fc0")
        h = act_bn_linear(h, maps, test, name="fc1")
        w_init = ConstantInitializer(0)
        b_init = ConstantInitializer(0)
        g_pred = PF.affine(h, maps, w_init=w_init, b_init=b_init, name="fc")
        g_pred.persistent = True
    return g_pred

def cnn_gradient_synthesizer(x, y=None, test=False):
    bs = x.shape[0]
    maps = x.shape[1]
    s0, s1 = x.shape[2:]
    if y is not None:
        h = F.one_hot(y, (10, ))
        h = F.reshape(h, (bs, 10, 1, 1))
        h = F.broadcast(h, (bs, 10, s0, s1))
        h = F.concatenate(*[x, h], axis=1)
    else:
        h = x
    with nn.parameter_scope("gs"):
        h = act_bn_conv(h, maps, test, name="conv0")
        w_init = ConstantInitializer(0)
        b_init = ConstantInitializer(0)
        g_pred = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), 
                                w_init=w_init, b_init=b_init, name="conv")
        g_pred.persistent = True
    return g_pred

def mlp_dni(image, y=None, maps=256, ncls=10, test=False):
    with nn.parameter_scope("ref"):
        image /= 255.
        h = act_bn_linear(image, maps, test, name="fc0")
        h = act_bn_linear(h, maps, act=None, test=test, name="fc1")

    # decoupled here
    h_d, h_copy, x_input, g_label = decouple(h)  
    g_pred = mlp_gradient_synthesizer(x_input, y, test)
    h_d.grad = g_pred.data

    h = F.relu(h)
    with nn.parameter_scope("ref"):
        h = act_bn_linear(h, maps, test, name="fc2")
        pred = PF.affine(h, ncls, name="fc")
        pred.persistent = True

    return h_d, h_copy, pred, g_pred, g_label

def cnn_dni(image, y=None, maps=128, ncls=10, test=False):
    with nn.parameter_scope("ref"):
        image /= 255.
        h = act_bn_conv(image, maps, test, name="conv0")
        h = F.max_pooling(h, (2, 2))  # 28x28 -> 14x14
        h = act_bn_conv(h, maps, act=None, test=test, name="conv1")

    # decoupled here
    h_d, h_copy, x_input, g_label = decouple(h)  
    g_pred = cnn_gradient_synthesizer(x_input, y, test)
    h_d.grad = g_pred.data

    h = F.relu(h)  # decouple after non-linearity
    h = F.max_pooling(h, (2, 2))  # 14x14 -> 7x7
    with nn.parameter_scope("ref"):
        h = act_bn_conv(h, maps, test, name="conv2")
        h = F.average_pooling(h, (7, 7))  # 7x7 -> 1x1
        pred = PF.affine(h, ncls, name="fc")
        pred.persistent = True

    return h_d, h_copy, pred, g_pred, g_label
    
def ce_loss(pred, label):
    return F.mean(F.softmax_cross_entropy(pred, label))

def se_loss(pred, label):
    return F.mean(F.squared_error(pred, label))

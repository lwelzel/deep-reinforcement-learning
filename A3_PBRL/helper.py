#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#######################
#
#  Additional general functions for computing and plotting
#  
#  IMPORTANT FUNCTIONS
#
#
#
#
#######################

import numpy as np

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp'.
        Borrowed from Helper.py from previous Assignment.
    '''
    x = x / temp  # scale by temperature
    z = x - max(x)  # substract max to prevent overflow of softmax
    return np.exp(z) / np.sum(np.exp(z))  # compute softmax


def argmax(x):
    ''' Own variant of np.argmax with random tie breaking.
        Borrowed from Helper.py from previous Assignment.
    '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)




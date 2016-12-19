#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""
from numpy.random import permutation


class MassPrior(object):
    """
    TODO
    """
    def __init__(self, prior):
        self.prior = prior

    def sample(self, size=1): #TODO: revisit size
        return permutation(self.prior)

# class Distribution(object):
#     def __init(self):
#         raise NotImplementedError()
#
#     def sample(self, n):
#         raise NotImplementedError()
#
#     def evaluate(self, x):
#         raise NotImplementedError()
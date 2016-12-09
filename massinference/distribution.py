#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""
from numpy.random import uniform


class MassPrior(object):
    """
    TODO
    """
    def sample(self, size=1):
        return uniform(10**10, 10**15, size)

# class Distribution(object):
#     def __init(self):
#         raise NotImplementedError()
#
#     def sample(self, n):
#         raise NotImplementedError()
#
#     def evaluate(self, x):
#         raise NotImplementedError()
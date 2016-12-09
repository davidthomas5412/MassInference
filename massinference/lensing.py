#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO

TODO fix spacing
"""

from numpy import isnan, log, ones, shape, arccos, arccosh


def bmo_f(x, t):
    """
    TODO

    input are pandas series
    """
    # x[[x == 1]]=1.+1e-5
    f = f_f(x)
    z = t**2/(2*(t**2+1)**2)*(
        ((t**2+1)/(x**2-1))*(1-f)
        +
        2*f
        -
        3.14159/(t**2+x**2)**.5
        +
        (t**2-1)*l_f(x,t)
        /
        (t*(t**2+x**2)**.5)
        )
    if isnan(z).any():
        raise Exception("bmo_f({}, {}) => isnan(z).any() == True".format(x, t))
    return 4*z


def bmo_g(x, t):
    # x[x==1]=1.+1e-5
    z = t**2/((t**2+1)**2)*(
        ((t**2+1)+2*(x**2-1))*f_f(x)  # possibly need -1 here!!
        +
        t*3.14159
        +
        (t**2-1)*log(t)
        +
        ((t**2+x**2)**.5)*(-3.14159+(t**2-1)*l_f(x,t)/t)
        )
    return 4*z/(x**2)


def l_f(x, t): # TODO: better names
    return log(x/(((t**2+x**2)**.5)+t))


def delta_c(c):
    return (200./3)*(c**3)/(log(1+c)-c/(1+c))


def f_f(x):
    z = ones(shape(x))
    z[x > 1] = arccos(1 / x[x > 1]) / ((x[x > 1] ** 2 - 1) ** .5)
    z[x < 1] = arccosh(1 / x[x < 1])/((1 - x[x < 1] ** 2) ** .5)
    z[x == 1] = 0.69314718  # np.log(2)

    if isnan(z).any():
        raise Exception("f_f({}) => isnan(z).any() == True".format(x))
    return z
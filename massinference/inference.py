#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""
from numpy import mean, sqrt, log, pi, sum, zeros


def log_likelihood(predictions, e1, e2, sigma_e):
    """
        Compare observed ellipticity to ray traced reduced shear.
    """
    e1_std = e1.std()
    e2_std = e2.std()
    std_obs = mean([e1_std, e2_std])
    sigma = sqrt(sigma_e ** 2 + std_obs ** 2)

    # Calculate the (log of) normalization constant
    samples = 2. * len(predictions)
    log_z = (samples / 2.) * log(2. * pi * sigma ** 2)
    log_likelihood = zeros(len(predictions))

    for i in xrange(len(predictions)):
        g = predictions[i]
        # Calculate chi2
        chi2 = sum((e1 - g.real) ** 2 / sigma ** 2) + sum((e2 - g.imag) ** 2 / sigma ** 2)
        # Calculate log-likelihood
        log_likelihood[i] = -log_z + (-0.5) * chi2
    return log_likelihood

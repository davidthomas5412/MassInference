#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""
from numpy import abs, array, zeros
from .angle import Angle
from .catalog import SourceCatalog


class Lenser(object):
    def __init__(self):
        raise NotImplementedError()


class MapLenser(Lenser):
    """
    TODO
    """
    def __init__(self, kappa_map, shear_map):
        self.kappa_map = kappa_map
        self.shear_map = shear_map

    def lens(self, source_catalog):
        '''
        Lense background galaxies by the shear and convergence in their respective Kappamaps and Shearmaps.
        '''

        # Exctract needed data from catalog galaxies
        ra = Angle.radian_to_degree(source_catalog.column_fast(SourceCatalog.RA))
        dec = Angle.radian_to_degree(source_catalog.column_fast(SourceCatalog.DEC))
        e1_int = source_catalog.column_fast(SourceCatalog.E1)
        e2_int = source_catalog.column_fast(SourceCatalog.E2)

        # Initialize new variables (note: e and g have to be initialized as complex for memory allocation)
        source_count = len(source_catalog)
        kappa = zeros(source_count)
        gamma1 = zeros(source_count)
        gamma2 = zeros(source_count)
        e1 = zeros(source_count)
        e2 = zeros(source_count)
        e = e1+1j*e2

        # Extract convergence and shear values at each galaxy location from maps
        for i in range(len(source_catalog)):
            kappa[i] = self.kappa_map.at(ra[i], dec[i], mapfile=0)
            gamma1[i] = self.shear_map.at(ra[i], dec[i], mapfile=0)
            gamma2[i] = self.shear_map.at(ra[i], dec[i], mapfile=1)

        # Calculate the reduced shear g and its conjugate g_conj
        g = (gamma1 + 1j*gamma2)/(1.0-kappa)
        g_conj = array([val.conjugate() for val in g])

        # Calculate the observed ellipticity for weak lensing events
        index = abs(g) < 1.0
        e[index] = ((e1_int[index] + 1j*e2_int[index]) + g[index]) / (1.0+g_conj[index] * (
            e1_int[index] + 1j*e2_int[index]))

        # Calculate the observed ellipticity for strong lensing events
        index = ~index
        e1_int_conj = array([val.conjugate() for val in e1_int])
        e2_int_conj = array([val.conjugate() for val in e2_int])
        e[index] = (1.0 + g[index]*(e1_int_conj[index]+1j*e2_int_conj[index])) / \
                   ((e1_int_conj[index]+1j*e2_int_conj[index]) + g_conj[index])

        # Calculate Cartesian and polar components
        return e.real, e.imag

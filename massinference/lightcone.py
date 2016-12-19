#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""

from numpy import arctan2, cos, digitize, linspace, log10, sin, sum, sqrt, zeros
from pandas import DataFrame
from .angle import Angle
from .catalog import SourceCatalog, HaloCatalog
from .distance import Distance
from .lensing import bmo_f, bmo_g, delta_c


class Lightcone(object):
    """
    TODO
    """
    def __init__(self, source_id, halo_catalog, radius, ra, dec, z, da_p):
        self.source_id = source_id
        x = Angle.radian_to_arcmin(-cos(halo_catalog.column_fast(HaloCatalog.DEC)) * (
            halo_catalog.column_fast(HaloCatalog.RA) - ra))
        y = Angle.radian_to_arcmin(halo_catalog.column_fast(HaloCatalog.DEC) - dec)
        self.phi = arctan2(y, x)
        radius_to_line_of_sight = sqrt(x*x + y*y)
        self.halo_indices = ((radius_to_line_of_sight < radius) &
                             (halo_catalog.column_fast(HaloCatalog.Z) <= z)
                             ).nonzero()[0]
        self.ra = ra
        self.dec = dec
        self.z = z
        self.cone_rphys = Angle.arcmin_to_radian(radius_to_line_of_sight[self.halo_indices] *
                                                 da_p[self.halo_indices])

    def compute_shear(self, kappa_s, x_trunc, r_s):
        cone_kappa_s = kappa_s[self.halo_indices]
        cone_x_trunc = x_trunc[self.halo_indices]
        cone_r_s = r_s[self.halo_indices]
        cone_phi = self.phi[self.halo_indices]
        cone_x = self.cone_rphys / cone_r_s
        cone_f = bmo_f(cone_x, cone_x_trunc)
        cone_g = bmo_g(cone_x, cone_x_trunc)
        gamma = 1.0 * cone_kappa_s * (cone_g - cone_f)
        # Combine shears and convergences:
        total_kappa = sum(cone_kappa_s * cone_f)
        total_gamma_1 = - sum(gamma * cos(2 * cone_phi))
        total_gamma_2 = - sum(gamma * sin(2 * cone_phi))
        g = (total_gamma_1 + 1j * total_gamma_2) / (1.0 - total_kappa) # g_halo
        return g


class LightconeManager(object):
    """
    TODO
    """
    def __init__(self, source_catalog, halo_factory, radius=4):
        self.source_catalog = source_catalog
        self.halo_factory = halo_factory
        # TODO: add back
        # halo_catalog = halo_factory.generate()
        # z = halo_catalog.column_fast(HaloCatalog.Z)[0] #constant
        halo_catalog = halo_factory.mutable_mass_halo_catalog
        halo_z = halo_catalog.column_fast(HaloCatalog.Z)
        source_z = source_catalog.dataframe[SourceCatalog.Z].max()
        grid = Grid(source_z)
        p = grid.snap(halo_z)
        da_p = grid.Da_p[p]
        self.rho_crit = grid.rho_crit[p]
        self.sigma_crit = grid.sigma_crit[p]
        self.lightcones = []
        self.results = {}
        for i in range(len(source_catalog)):
            row = source_catalog.dataframe.iloc[i]
            source_id = row[SourceCatalog.ID]
            ra = row[SourceCatalog.RA]
            dec = row[SourceCatalog.DEC]
            z = row[SourceCatalog.Z]
            self.lightcones.append(Lightcone(source_id, halo_catalog, radius, ra, dec, z, da_p))

    def run(self, steps):
        results = zeros((steps, len(self.lightcones)), dtype=complex)
        for step in xrange(steps):
            # Make a Simple Monte Carlo draw:
            #TODO: change back to self.halo_factory.generate()
            halo_catalog = self.halo_factory.mutable_mass_halo_catalog
            # Compute quantities for each halo:
            m200 = halo_catalog.column_fast(HaloCatalog.HALO_MASS)
            r200 = (3 * m200 / (800 * 3.14159 * self.rho_crit)) ** (1./3)
            logc_maccio = 1.020 - 0.109 * (log10(m200) - 12) # see paper
            c200 = 10 ** logc_maccio
            r_s = r200/c200
            rho_s = delta_c(c200) * self.rho_crit
            # Compute quantities relevant to each lensing event:
            kappa_s = rho_s * r_s / self.sigma_crit
            truncation_scale = 10
            r_trunc = truncation_scale * r200
            x_trunc = r_trunc / r_s
            for i,lightcone in enumerate(self.lightcones):
                results[step, i] = lightcone.compute_shear(kappa_s, x_trunc, r_s)
        return results


class Grid(object):
    """
    TODO
    """
    def __init__(self,zs,nplanes=100,cosmo=[0.25,0.75,0.73]):

        distance = Distance()
        self.zmax = zs*1.0
        self.zs = zs*1.0
        self.nplanes = nplanes
        self.cosmo = cosmo

        # These are the plane redshifts:
        self.redshifts, self.dz = linspace(0.0,self.zmax,self.nplanes,
                                                               endpoint=True,retstep=True)
        self.redshifts += (self.dz/2.)
        self.nz = len(self.redshifts)

        # Snap lens to grid, and compute special distances:
        self.Da_s = distance.Da(zs)
        self.plane = {}

        # Grid planes:
        self.Da_p = zeros(self.nz)
        self.rho_crit = zeros(self.nz)
        self.Da_ps = zeros(self.nz)
        self.Da_pl = zeros(self.nz)
        self.sigma_crit = zeros(self.nz)
        for i in range(self.nz):
            z = self.redshifts[i]
            self.Da_p[i] = distance.Da(0,z)
            self.rho_crit[i] = distance.rho_crit_univ(z)
            self.Da_ps[i] = distance.Da(z,zs)
            zl = 0
            self.Da_pl[i] = distance.Da(z,zl)
            self.sigma_crit[i] = (1.663*10**18)*(self.Da_s/(self.Da_p[i]*self.Da_ps[i]))  # units M_sun/Mpc^2

    def snap(self, z):
        snapped_p = digitize(z, self.redshifts - self.dz / 2.0) - 1
        snapped_p[snapped_p < 0] = 0  # catalogs have some blue-shifted objects!
        snapped_z = self.redshifts[snapped_p]
        return snapped_p

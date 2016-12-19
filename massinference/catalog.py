#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Catalog.

TODO: add plotting
"""
from pandas import DataFrame, read_csv, read_table
from numpy import full, sqrt, power, arctan2, rad2deg
from numpy.random import uniform, normal
from .data import GUO_FILE
from .seed import set_numpy_random_seed

#TODO: allow access to columns directly via []

class Catalog(object):
    """
    TODO
    """
    REQUIRED_KEYS = []

    def __init__(self, dataframe):
        self.check(dataframe)
        self.dataframe = dataframe
        self._matrix = dataframe.as_matrix()

    def __len__(self):
        return len(self.dataframe)

    def check(self, dataframe):
        """
        TODO
        """
        columns = dataframe.columns
        for key in self.REQUIRED_KEYS:
            if key not in columns:
                raise Exception("Missing {} key in catalog dataframe!".format(key))
        if len(dataframe) == 0:
            raise Exception("Empty catalog!")

    def column_fast(self, key):
        """
        TODO

        use numpy arrays to avoid pandas.Series indexing and make huge savings on vectorized math operations
        see https://penandpants.com/2014/09/05/performance-of-pandas-series-vs-numpy-arrays/
        """
        return self._matrix[:, self.dataframe.columns.get_loc(key)]


class SourceCatalog(Catalog):
    """
    TODO
    """
    ID = 'id'
    RA = 'ra'
    DEC = 'dec'
    Z = 'z'
    E1 = 'e1'
    E2 = 'e2'
    E_MAG = 'e_mag'
    E_PHI = 'e_phi'

    REQUIRED_KEYS = [ID, RA, DEC, Z, E1, E2, E_MAG, E_PHI]

    def __init__(self, dataframe):
        super(SourceCatalog, self).__init__(dataframe)

    #TODO: handle csv vs table
    @staticmethod
    def from_file(filename, keymap):
        dataframe = read_csv(filename, usecols=keymap.keys())
        dataframe.rename(columns=keymap, inplace=True)
        return SourceCatalog(dataframe)


class SourceCatalogFactory(object):
    """
    TODO
    """
    Z = 1.3857

    def __init__(self, limits, density, sigma_e=0.2, random_seed=None):
        if random_seed:
            set_numpy_random_seed(random_seed)
        self.limits = limits
        self.density = density
        self.sigma_e = sigma_e

    def generate(self):
        """
        TODO
        """
        df = DataFrame()
        area = (self.limits.xf.arcmin - self.limits.xi.arcmin) * (self.limits.yf.arcmin -
                                                           self.limits.yi.arcmin)
        count = abs(int(area * self.density)) #TODO: figure out negative area
        df[SourceCatalog.ID] = range(count)
        df[SourceCatalog.RA] = uniform(self.limits.xi.radian, self.limits.xf.radian,
                                       count)
        df[SourceCatalog.DEC] = uniform(self.limits.yi.radian, self.limits.yf.radian,
                                        count)
        df[SourceCatalog.Z] = full(count, SourceCatalogFactory.Z)
        e1 = normal(0, self.sigma_e, count)
        e2 = normal(0, self.sigma_e, count)
        # Change any |e|> 1 ellipticity components
        while abs(e1 > 1.0).any() or abs(e2 > 1.0).any():
            for i in abs(e1 > 1.0).nonzero():
                e1[i] = normal(0.0, self.sigma_e)
            for i in (abs(e2) > 1.0).nonzero():
                e2[i] = normal(0.0, self.sigma_e)
        df[SourceCatalog.E1] = e1
        df[SourceCatalog.E2] = e2
        df[SourceCatalog.E_MAG] = sqrt(power(df[SourceCatalog.E1], 2) + power(df[SourceCatalog.E2],
                                       2))
        df[SourceCatalog.E_PHI] = rad2deg(arctan2(df[SourceCatalog.E2], df[SourceCatalog.E1])) / 2.0
        return SourceCatalog(df)


class HaloCatalog(Catalog):
    """
    TODO
    """
    ID = 'id'
    HALO_MASS = 'mass_h'
    STELLAR_MASS = 'mass_s'
    RA = 'ra'
    DEC = 'dec'
    Z = 'z'

    def __init__(self, dataframe):
        super(HaloCatalog, self).__init__(dataframe)

    @staticmethod
    def from_file(filename, keymap):
        dataframe = read_table(filename, usecols=keymap.keys())
        dataframe.rename(columns=keymap, inplace=True)
        return HaloCatalog(dataframe)

    #TODO: switch this map
    @staticmethod
    def default():
        keymap = {
            'GalID': HaloCatalog.ID,
            'M_Subhalo[M_sol/h]': HaloCatalog.HALO_MASS,
            'M_Stellar[M_sol/h]': HaloCatalog.STELLAR_MASS,
            'pos_0[rad]': HaloCatalog.RA,
            'pos_1[rad]': HaloCatalog.DEC,
            'z_spec': HaloCatalog.Z,
        }
        return HaloCatalog.from_file(GUO_FILE, keymap=keymap)


class FastSampleHaloCatalogFactory(object):
    """
    TODO
    """
    def __init__(self, mutable_mass_halo_catalog, mass_distribution, random_seed=None):
        if random_seed:
            set_numpy_random_seed(random_seed)
        self.mutable_mass_halo_catalog = mutable_mass_halo_catalog
        self.mass_distribution = mass_distribution

    def generate(self):
        halo_mass = self.mass_distribution.sample(len(self.mutable_mass_halo_catalog))
        self.mutable_mass_halo_catalog.set_halo_mass(halo_mass)
        return self.mutable_mass_halo_catalog


class MutableHaloMassCatalog(HaloCatalog):
    """
    TODO
    """
    def __init__(self, dataframe):
        super(MutableHaloMassCatalog, self).__init__(dataframe)

    @staticmethod
    def from_file(filename, keymap, limits, z):
        dataframe = read_table(filename, usecols=keymap.keys())
        dataframe.rename(columns=keymap, inplace=True)
        dataframe[HaloCatalog.RA] = -dataframe[HaloCatalog.RA] # left handed coordinate system
        dataframe = dataframe[
                        (dataframe[HaloCatalog.RA] > limits.xf.radian) &
                        (dataframe[HaloCatalog.RA] < limits.xi.radian) &
                        (dataframe[HaloCatalog.DEC] > limits.yi.radian) &
                        (dataframe[HaloCatalog.DEC] < limits.yf.radian) &
                        (dataframe[HaloCatalog.HALO_MASS] > 0) &
                        (dataframe[HaloCatalog.Z] <= z)
                        ]\
            .reset_index(drop=True)
        return MutableHaloMassCatalog(dataframe)

    #TODO: best sequence of signatures for radius
    @staticmethod
    def default(limits, z):
        keymap = {
            'GalID': HaloCatalog.ID,
            'M_Subhalo[M_sol/h]': HaloCatalog.HALO_MASS,
            'M_Stellar[M_sol/h]': HaloCatalog.STELLAR_MASS,
            'pos_0[rad]': HaloCatalog.RA,
            'pos_1[rad]': HaloCatalog.DEC,
            'z_spec': HaloCatalog.Z
        }
        return MutableHaloMassCatalog.from_file(GUO_FILE, keymap, limits, z)

    def set_halo_mass(self, column):
        self.dataframe[HaloCatalog.HALO_MASS] = column

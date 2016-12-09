#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lens maps for lensing a background.
"""

import struct
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

from .data import KAPPA_FILE, GAMMA_1_FILE, GAMMA_2_FILE
from .plot import Limits, PlotConfig, set_figure_size, make_wcs_axes


class LensMap(object):
    """
    Stores lensing values in a 2D world-coordinate map.
    Superclass of shearmap and kappamap.
    """
    def __init__(self, map_files):
        """
        Args:
             map_files (str of list(str)): files to make map from. One file for kappamap,
                list of two files for shearmap.
        """
        if type(map_files) != list:
            map_files = [map_files]

        # Declare needed attributes as lists
        self.headers = []
        self.values = []
        self.NX = []
        self.PIXSCALE = []
        self.field = []
        self.wcs = []
        self.output = []
        self.map_x = []
        self.map_y = []

        # Parsing the file name(s)
        # 0 <= x,y <= 7, each (x,y) map covers 4x4 square degrees
        for i in range(len(map_files)):
            input_parse = map_files[i].split(
                '_')  # Creates list of filename elements separated by '_'
            self.map_x.append(int(input_parse[3]))  # The x location of the map grid
            self.map_y.append(int(input_parse[4]))  # The y location of the map grid

            # Initialise some necessary WCS parameters for Stefan
            # Hilbert's binary data files:
            self.field.append(4.0)  # degrees
            self.NX.append(4096)  # pixels
            self.PIXSCALE.append(self.field[i] / (1.0*self.NX[i]))  # degrees
            self.set_wcs(i)

            map_file = open(map_files[i], 'rb')
            data = map_file.read()
            map_file.close()
            fmt = str(self.NX[i] * self.NX[i])+'f'
            start = 0
            stop = struct.calcsize(fmt)
            values = struct.unpack(fmt, data[start:stop])
            self.values.append(np.array(values, dtype=np.float32).reshape(self.NX[i], self.NX[i])
                               .transpose())

            # Create a FITS header
            header = pyfits.Header()
            # Add WCS keywords to the FITS header (in apparently random order):
            for keyword in self.wcs[i].keys():
                header.set(keyword, self.wcs[i][keyword])
            self.headers.append(header)

        # Check file consistency
        assert self.PIXSCALE[1:] == self.PIXSCALE[:-1]
        assert self.field[1:] == self.field[:-1]
        assert self.NX[1:] == self.NX[:-1]

    def at(self, x, y, mapfile=0):
        """
        Interpolating the map to return a single value at a specified point.

        Args:
            x (float): the x coordinate
            y (float): the y coordinate
            mapfile (optional int): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.

        Returns:
            float: map value at provided coordinates in world coordinate system.
        """
        i, j = self.world_to_image(x, y, mapfile)
        return self.lookup(i, j, mapfile)

    def set_wcs(self, i):
        """
        WCS parameters to allow conversion between
            - image coordinates i,j (pixels)
            - physical coordinates x,y (rad)
            - sky coordinates ra,dec (deg, left-handed system)

        Args:
            i (int): mapfile index
        """
        self.wcs.append(dict())
        self.wcs[i]['CRPIX1'] = 0.0
        self.wcs[i]['CRPIX2'] = 0.0
        self.wcs[i]['CRVAL1'] = 0.5 * self.field[i] - self.map_x[i] * self.field[i]
        self.wcs[i]['CRVAL2'] = -0.5 * self.field[i] + self.map_y[i] * self.field[i]
        self.wcs[i]['CD1_1'] = -self.PIXSCALE[i]
        self.wcs[i]['CD1_2'] = 0.0
        self.wcs[i]['CD2_1'] = 0.0
        self.wcs[i]['CD2_2'] = self.PIXSCALE[i]
        self.wcs[i]['CTYPE1'] = 'RA---TAN'
        self.wcs[i]['CTYPE2'] = 'DEC--TAN'
        self.wcs[i]['LTV1'] = 0.5 * self.field[i] / self.PIXSCALE[i] - 0.5
        self.wcs[i]['LTV2'] = 0.5 * self.field[i] / self.PIXSCALE[i] - 0.5
        self.wcs[i]['LTM1_1'] = 1.0 / np.deg2rad(self.PIXSCALE[i])
        self.wcs[i]['LTM2_2'] = 1.0 / np.deg2rad(self.PIXSCALE[i])

    def world_to_image(self, a, d, mapfile=0):
        """
        Args:
            a (float): physical right-ascension-coordinate
            d (float): world declination-coordinate
            mapfile (optional int): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.

        Returns:
            float, float: image coordinates
        """
        i = (a - self.wcs[mapfile]['CRVAL1']) / self.wcs[mapfile]['CD1_1'] + self.wcs[mapfile][
            'CRPIX1']
        # if a negative pixel is returned for i, reinput a as a negative degree
        if i < 0:
            a -= 360
            i = (a - self.wcs[mapfile]['CRVAL1']) / self.wcs[mapfile]['CD1_1'] + self.wcs[
                mapfile]['CRPIX1']
        j = (d - self.wcs[mapfile]['CRVAL2']) / self.wcs[mapfile]['CD2_2'] + self.wcs[mapfile][
            'CRPIX2']
        return i, j

    def image_to_world(self, i, j, mapfile=0):
        """
        Note: only approximate WCS transformations - assumes dec=0.0 and small field

        Args:
            i (float): x-axis pixel coordinate
            j (float): y-axis pixel coordinate
            mapfile (optional int): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.

        Returns:
            float, float: approximate world coordinates
        """
        a = self.wcs[mapfile]['CRVAL1'] + self.wcs[mapfile]['CD1_1']*(i - self.wcs[mapfile]['CRPIX1'])
        # if a < 0.0: a += 360.0 : We are using WCS with negative RAs in degrees, now
        d = self.wcs[mapfile]['CRVAL2'] + self.wcs[mapfile]['CD2_2']*(j - self.wcs[mapfile]['CRPIX2'])
        return a, d

    def lookup(self, i, j, mapfile):
        """
        Lookup value in image.

        Note:
            Need to transpose x and y, since the maps are transposed...

        Args:
            i (float): image x-coordinate
            j (float): image y-coordinate
            mapfile (int): mapfile index

        Return:
             float: weighted mean of 4 neighbouring pixels, as suggested by Stefan
        """
        ix = int(j)
        iy = int(i)
        px = j - ix
        py = i - iy

        if (0 <= ix) and (ix < self.NX[mapfile]-1) and (0 <= iy) and (iy < self.NX[mapfile]-1):
            mean = self.values[mapfile][ix, iy] * (1.0 - px) * (1.0 - py) \
                  + self.values[mapfile][ix+1, iy] * px * (1.0 - py) \
                  + self.values[mapfile][ix, iy+1] * (1.0 - px) * py \
                  + self.values[mapfile][ix+1, iy+1] * px * py
        else:
            mean = None

        return mean


class KappaMap(LensMap):
    """
    Read in, store, transform and interrogate a convergence map.

    Args:
        kappa_file (str): name of file containing a convergence map
    """
    def __init__(self, kappa_file):
        super(KappaMap, self).__init__(kappa_file)

    def plot(self, plot_config=None, fig_size=10, limits=None):
        """
        Plot the convergence as a heatmap image.

        Args:
            plot_config (optional PlotConfig): plot config to use if overlay, otherwise defaults to
                None and makes new plot
            fig_size (optional float): figure size in inches, defaults to 10
            limits (optional list(float)): list of four plot limits [xi, xf, yi, yf], defaults to
                full file

        Returns:
            PlotConfig: plot configuration object which can be used to overlay plots
        """
        if plot_config is None:
            fig = plt.figure()
            if limits is None:
                limits = Limits.default(self)
            pix_limits = limits.to_pixels(self)
            set_figure_size(fig, fig_size, limits)
            ax = make_wcs_axes(fig, limits, pix_limits, self.headers[0])
            plot_config = PlotConfig(fig, ax, limits)

        # Get the colormap limits
        kmin = np.min(self.values[0])
        kmax = np.max(self.values[0])
        cax = plot_config.ax.imshow(self.values[0], vmin=kmin, vmax=kmax, cmap='inferno',
                                    origin='lower')
        plot_config.fig.colorbar(cax, ticks=[kmin, (kmin+kmax) / 2.0, kmax], label='Convergence '
                                                                                   '$\kappa$')
        return PlotConfig(fig, ax, limits)

    @staticmethod
    def default():
        """
        Returns:
            (KappaMap): default kappamap used in experiments.
        """
        return KappaMap(KAPPA_FILE)


class ShearMap(LensMap):
    """
    Read in, store, transform and interrogate a pair of shear maps.

    Args:
        gamma_files(list(str)): list of the two filenames of shear maps

    """
    def __init__(self, gamma_files):
        super(ShearMap, self).__init__(gamma_files)

    def plot(self, plot_config=None, fig_size=10, limits=None):  # fig_size in inches
        """
        Plot the shear field with shear sticks.

        Args:
            fig (optional matplotlib.pyplot.figure): figure to plot on if want to overlay, defaults to None -> new
                plot
            fig_size (optional float): figure size in inches, defaults to 10
            limits (optional list(float)): list of four plot limits [xi, xf, yi, yf], defaults to
                full file

        Returns:
            PlotConfig: plot configuration object which can be used to overlay plots
        """
        if plot_config is None:
            fig = plt.figure()
            if limits is None:
                limits = Limits.default(self)
            pix_limits = limits.to_pixels(self)
            set_figure_size(fig, fig_size, limits)
            ax = make_wcs_axes(fig, limits, pix_limits, self.headers[0])
            plot_config = PlotConfig(fig, ax, limits)
        else:
            limits = plot_config.limits
            pix_limits = plot_config.limits.to_pixels(self)

        # Retrieve gamma values in desired limits
        gamma1 = self.values[0][int(pix_limits.yi):int(pix_limits.yf),
                                int(pix_limits.xi):int(pix_limits.xf)]
        gamma2 = self.values[1][int(pix_limits.yi):int(pix_limits.yf),
                                int(pix_limits.xi):int(pix_limits.xf)]

        # Create arrays of shear stick positions, one per pixel in world coordinates
        mesh_x, mesh_y = np.meshgrid(np.arange(limits.xi, limits.xf, -self.PIXSCALE[0]),
                                     np.arange(limits.yi, limits.yf, self.PIXSCALE[0]))

        # Calculate the modulus and angle of each shear
        mod_gamma = np.sqrt(gamma1 * gamma1 + gamma2 * gamma2)
        phi_gamma = np.arctan2(gamma2, gamma1) / 2.0

        # Sticks in world coords need x reversed, to account for left-handed system
        pix_l_mean = np.mean([pix_limits.lx, pix_limits.ly])
        dx = mod_gamma * np.cos(phi_gamma) * pix_l_mean
        dy = mod_gamma * np.sin(phi_gamma) * pix_l_mean

        # Pixel sampling rate for plotting of shear maps
        downsampling_rate = 40.0
        shear_spacing = int(np.floor(pix_limits.lx / downsampling_rate))

        # Plot downsampled 2D arrays of shear sticks in current axes.
        plot_config.ax.quiver(mesh_x[::shear_spacing, ::shear_spacing],
                              mesh_y[::shear_spacing, ::shear_spacing],
                              dx[::shear_spacing, ::shear_spacing],
                              dy[::shear_spacing, ::shear_spacing],
                              alpha=0.8,
                              color='b',
                              headwidth=0,
                              pivot='mid',
                              transform=plot_config.ax.get_transform('world'))

        return plot_config

    @staticmethod
    def default():
        """
        Returns:
            (ShearMap): default shear maps used in experiments.
        """
        return ShearMap([GAMMA_1_FILE, GAMMA_2_FILE])

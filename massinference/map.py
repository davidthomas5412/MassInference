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
from .plot import set_figure_size, set_axes


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

    def at(self, x, y, mapfile=0, coordinate_system='world'):
        """
        Interpolating the map to return a single value at a specified point.

        Args:
            x (float): the x coordinate
            y (float): the y coordinate
            mapfile (optional int): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.
            coordinate_system (optional str): coordinate system of (x,y). Defaults to
                'world'.

        Returns:
            float: map value at provided coordinates in provided coordinate system.
        """
        if coordinate_system == 'world':
            i, j = self.world_to_image(x, y, mapfile)
        elif coordinate_system == 'physical':
            i, j = self.physical_to_image(x, y, mapfile)
        else:
            i, j = x, y
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

    def physical_to_image(self, x, y, mapfile=0):
        """
        Args:
            x (float): physical x-coordinate
            y (float): physical y-coordinate
            mapfile (optional int): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.

        Returns:
            float, float: image coordinates
        """
        i = self.wcs[mapfile]['LTV1'] + self.wcs[mapfile]['LTM1_1']*x # x in rad
        j = self.wcs[mapfile]['LTV2'] + self.wcs[mapfile]['LTM2_2']*y # y in rad
        return i, j

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

    #TODO: fill in all these docstrings
    #TODO: add tests for these
    def image_to_physical(self,i,j,mapfile=0):
        x = (i - self.wcs[mapfile]['LTV1'])/self.wcs[mapfile]['LTM1_1'] # x in rad
        y = (j - self.wcs[mapfile]['LTV2'])/self.wcs[mapfile]['LTM2_2'] # y in rad
        return x,y

     # Only approximate WCS transformations - assumes dec=0.0 and small field
    def image_to_world(self,i,j,mapfile=0):
        """
        Note: only approximate WCS transformations - assumes dec=0.0 and small field

        :param i:
        :param j:
        :param mapfile:
        :return:
        """
        a = self.wcs[mapfile]['CRVAL1'] + self.wcs[mapfile]['CD1_1']*(i - self.wcs[mapfile]['CRPIX1'])
        #if a < 0.0: a += 360.0 : We are using WCS with negative RAs in degrees, now
        d = self.wcs[mapfile]['CRVAL2'] + self.wcs[mapfile]['CD2_2']*(j - self.wcs[mapfile]['CRPIX2'])
        return a,d

    def physical_to_world(self,x,y,mapfile=0):
        a = -np.rad2deg(x) - self.map_x[mapfile]*self.field[mapfile]
        #if a < 0.0: a += 360.0 :we are using nRA instead now
        d = np.rad2deg(y) + self.map_y[mapfile]*self.field[mapfile]
        return a,d

    def world_to_physical(self,a,d,mapfile=0):
        x = -np.deg2rad(a + self.map_x[mapfile]*self.field[mapfile])
        y = np.deg2rad(d - self.map_y[mapfile]*self.field[mapfile])
        return x,y

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
             float: weighted mean of 4 neighbouring pixels, as suggested by Stefan.
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

    def setup_plot(self, subplot=None, coords='world'):
        """
        Set up a plot of the map.

        Args:
            subplot (list(float)): list of four plot limits [xmin, xmax, ymin, ymax].
            coords (string): input coordinate system for the subplot: 'pixel', 'physical',
                or 'world'.

        Returns:

        """
        # TODO: fill this in

        if subplot is None:
            # Default subplot is entire image
            ai, di = self.image_to_world(0, 0)
            af, df = self.image_to_world(self.NX[0], self.NX[0])
            subplot = [ai, af, di, df]

        xi, xf = subplot[0], subplot[1]  # x-limits for subplot
        yi, yf = subplot[2], subplot[3]  # y-limits for subplot

        if coords == 'world':
            # Subplot is already in world coordinates
            lx = 1.0 * abs(xf - xi)  # length of x-axis subplot
            ly = 1.0 * abs(yf - yi)  # length of y-axis subplot

        elif coords == 'physical':
            # Convert subplot bounds to world coordinates
            xi, yi = self.physical_to_world(xi, yi)
            xf, yf = self.physical_to_world(xf, yf)
            lx = 1.0 * abs(xf - xi)
            ly = 1.0 * abs(yf - yi)
            subplot = [xi, xf, yi, yf]

        elif coords == 'pixel':
            # Convert subplot bounds to world coordinates
            xi, yi = self.image_to_world(xi, yi)
            xf, yf = self.image_to_world(xf, yf)
            lx = 1.0 * abs(xf - xi)
            ly = 1.0 * abs(yf - yi)
            subplot = [xi, xf, yi, yf]

        else:
            raise IOError('Error: Subplot bounds can only be in pixel, physical, or world '
                          'coordinates.')

        # Convert subplot bounds to (floating point) pixel values
        pix_xi, pix_yi = self.world_to_image(xi, yi)
        pix_xf, pix_yf = self.world_to_image(xf, yf)

        # Pixel length of subplot
        pix_lx = pix_xf - pix_xi
        pix_ly = pix_yf - pix_yi

        return pix_xi, pix_xf, pix_yi, pix_yf, lx, ly, pix_lx, pix_ly, subplot


class KappaMap(LensMap):
    """
    Read in, store, transform and interrogate a convergence map.

    Args:
        kappa_file (str): name of file containing a convergence map
    """
    def __init__(self, kappa_file):
        super(KappaMap, self).__init__(kappa_file)

    def plot(self, fig=None, fig_size=10, subplot=None, coords='world'):
        """
        Plot the convergence as a heatmap image.

        Note:

        :param fig:
        :param fig_size:
        :param subplot:
        :param coords:
        :return:

        Parameters
        ----------
        fig_size : float
            Figure size in inches
        subplot : list, float
            Plot limits [xmin,xmax,ymin,ymax]
        coords : string
            Input coordinate system for the subplot: 'pixel', 'physical', or 'world'
        """
        #TODO: docstring

        check_limits(subplot, cords)
        if fig is None:
            fig = plt.figure('KappaMap')

        pix_xi, pix_xf, pix_yi, pix_yf, lx, ly, pix_lx, pix_ly, subplot = self.setup_plot(
            subplot, coords)
        set_figure_size(fig, fig_size, lx, ly)
        imsubplot = [pix_xi, pix_xf, pix_yi, pix_yf]
        ax = set_axes(fig, lx, ly, self.headers[0], imsubplot)
        values_to_display = self.values[0]
        # Get the colormap limits
        kmin = np.min(values_to_display)
        kmax = np.max(values_to_display)
        cax = ax.imshow(values_to_display, vmin=kmin, vmax=kmax, cmap='inferno', origin='lower')
        fig.colorbar(cax, ticks=[kmin, (kmin+kmax) / 2.0, kmax], label='Convergence '
                                                                               '$\kappa$')
        return fig, ax, subplot

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

    def plot(self, fig=None, fig_size=10, subplot=None, coords='world'):  # fig_size in inches
        """
        Plot the shear field with shear sticks.

        Args:
            fig_size        Figure size in inches
            subplot         List of four plot limits [xmin,xmax,ymin,ymax]
            coords          Type of coordinates inputted for the subplot:
                            'pixel', 'physical', or 'world'
        """
        #TODO: docstring
        pix_xi, pix_xf, pix_yi, pix_yf, lx, ly, pix_lx, pix_ly, subplot = self.setup_plot(
            subplot, coords)
        imsubplot = [pix_xi, pix_xf, pix_yi, pix_yf]

        if fig is None:
            fig = plt.figure('ShearMap')
            set_figure_size(fig, fig_size, lx, ly)
            ax = set_axes(fig, lx, ly, self.headers[0], imsubplot)
        else:
            ax = fig.get_axes()[0]

        # Retrieve gamma values in desired subplot
        gamma1 = self.values[0]#[pix_yi:pix_yf, pix_xi:pix_xf] TODO:
        gamma2 = self.values[1]#[pix_yi:pix_yf, pix_xi:pix_xf] TODO:

        # Create arrays of shear stick positions, one per pixel in world coordinates
        mesh_x, mesh_y = np.meshgrid(np.arange(subplot[0], subplot[1], -self.PIXSCALE[0]),
                                     np.arange(subplot[2], subplot[3], self.PIXSCALE[0]))

        # Calculate the modulus and angle of each shear
        mod_gamma = np.sqrt(gamma1 * gamma1 + gamma2 * gamma2)
        phi_gamma = np.arctan2(gamma2, gamma1) / 2.0

        # Sticks in world coords need x reversed, to account for left-handed
        # system:
        pix_l = np.mean([pix_lx, pix_ly])
        dx = mod_gamma * np.cos(phi_gamma) * pix_l
        dy = mod_gamma * np.sin(phi_gamma) * pix_l
        # Plot downsampled 2D arrays of shear sticks in current axes.
        # Pixel sampling rate for plotting of shear maps:
        shear_spacing = np.floor(pix_lx / 40.0)

        ax.quiver(mesh_x[::shear_spacing, ::shear_spacing],
                  mesh_y[::shear_spacing, ::shear_spacing],
                  dx[::shear_spacing, ::shear_spacing],
                  dy[::shear_spacing, ::shear_spacing],
                  alpha=0.8,
                  color='b',
                  headwidth=0,
                  pivot='middle',
                  transform=ax.get_transform('world'))

        return fig, ax, subplot

    @staticmethod
    def default():
        """
        Returns:
            (ShearMap): default shear maps used in experiments.
        """
        return ShearMap([GAMMA_1_FILE, GAMMA_2_FILE])
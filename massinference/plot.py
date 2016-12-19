#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common plotting objects and functions.
"""
from .angle import Angle

import numpy as np
from astropy.wcs import WCS


class PlotConfig(object):
    """
    Stores plot references for overlay plotting

    Attributes:
        fig (matplotlib.pyplot.figure): plot figure
        ax (matplotlib.pyplot.axes): plot axes
        limits (Limits): plot limits
    """
    def __init__(self, fig, ax, limits):
        self.fig = fig
        self.ax = ax
        self.limits = limits


class Limits(object):
    """
    Stores the limits for the x-axis and y-axis of a plot
    """
    def __init__(self, xi, xf, yi, yf, check=False):
        """
        TODO
        """
        self.xi = xi
        self.xf = xf
        self.yi = yi
        self.yf = yf
        if check:
            self.check_limits()
            #TODO: rethink limits
        # self.lx = float(abs(xf - xi))
        # self.ly = float(abs(yf - yi))

    def area(self):
        return (self.xf.arcmin - self.xi.arcmin) * (self.yf.arcmin - self.yi.arcmin)

    def add_perimeter(self, radius):
        ret = Limits(
            # left handed coordinate system
            Angle.from_arcmin(self.xi.arcmin + radius.arcmin),
            Angle.from_arcmin(self.xf.arcmin - radius.arcmin),
            Angle.from_arcmin(self.yi.arcmin - radius.arcmin),
            Angle.from_arcmin(self.yf.arcmin + radius.arcmin)
                    )
        return ret

    def check_limits(self):
        """
        Confirms limits are reasonable

        Raises:
            Exception if the limits would cause errors in plotting code
        """
        if self.xi < self.xf:
            raise Exception("x_i, x_f are in left handed right ascension coordinates, so x_i > x_f")
        elif self.yi > self.yf:
            raise Exception("y_i, y_f are in declination coordinates, so y_i < y_f")

    def to_pixels(self, lens_map):
        """
        TODO
        :param lens_map:
        :return:
        """
        pix_xi, pix_yi = lens_map.world_to_image(self.xi, self.yi)
        pix_xf, pix_yf = lens_map.world_to_image(self.xf, self.yf)
        return Limits(pix_xi, pix_xf, pix_yi, pix_yf, check=False)

    @staticmethod
    def default(lens_map):
        """
        TODO
        :param lens_map:
        :return:
        """
        xi, yi = lens_map.image_to_world(0, 0)
        xf, yf = lens_map.image_to_world(lens_map.NX[0], lens_map.NX[0])
        return Limits(xi, xf, yi, yf)


def set_figure_size(fig, fig_size, limits):
    """
    Takes inputted figure and changes its size according to Lx and Ly given in
    inches.

    TODO

    Args:
        fig (matplotlib.figure.Figure): figure to size
        fig_size (int): total figure size
        lx (float): length of x axis
        ly (float): length of y axis
    """

    if limits.lx > limits.ly:
        fig.set_size_inches(fig_size, fig_size * (1.0 * limits.ly / limits.lx))
    else:
        fig.set_size_inches(fig_size * (1.0 * limits.lx / limits.ly), fig_size)


def make_wcs_axes(fig, limits, pix_limits, header):
    """
    TODO: might be able to move into PlotConfig

    Sets wcs and pixel axes for plotting maps and catalogs. Both sets of axes are
    contained in the current figure instance, and also returned for ease of use.

    Args:
        fig (matplotlib.figure.Figure): figure to add axes to
        lx (float): length of x axis
        ly (float): length of y axis
        header (astropy.io.fits.header.Header): fits header for axes projection
        pix_limits (list(float)): list of four limits for plot [x_min, x_max, y_min, y_max]

    Returns:
        (matplotlib.axes.Axes): axes for plotting
    """

    # Create axis, always in world coordinates
    wcs = WCS(header)
    viewport = [0.1, 0.1, 0.8, 0.8]
    ax = fig.add_axes(viewport, projection=wcs, label='wcs')
    ax.set_autoscale_on(False)
    ra = ax.coords['ra']
    dec = ax.coords['dec']

    # Set pixel limits on axis
    ax.set_xlim(pix_limits.xi - 0.5, pix_limits.xf - 0.5)
    ax.set_ylim(pix_limits.yi - 0.5, pix_limits.yf - 0.5)

    # Set labels
    ra.set_axislabel('Right Ascension (deg)')
    dec.set_axislabel('Declination (deg)')

    # Set x/y-axis unit formatter
    if 0.03 < limits.lx <= 0.3:
        ra.set_major_formatter('d.dd')
    elif limits.lx <= 0.03:
        ra.set_major_formatter('d.ddd')
    else:
        ra.set_major_formatter('d.d')
    if 0.03 < limits.ly <= 0.3:
        dec.set_major_formatter('d.dd')
    elif limits.ly <= 0.03:
        dec.set_major_formatter('d.ddd')
    else:
        dec.set_major_formatter('d.d')

    # max number of ticks
    num = 8
    if limits.lx > limits.ly:
        ra.set_ticks(number=num)
        dec.set_ticks(number=np.ceil(num * limits.ly / limits.lx))
    elif limits.lx < limits.ly:
        ra.set_ticks(number=np.ceil(num * limits.lx / limits.ly))
        dec.set_ticks(number=num)
    else:
        ra.set_ticks(number=num)
        dec.set_ticks(number=num)

    return ax


def check_limits(limits):
    """
    Confirms limits are reasonable

    Args:
        limits (optional list(float)): list of four plot limits [xi, xf, yi, yf], defaults to
            full file

    Raises:
        Exception if the limits would cause errors in plotting code
    """
    if len(limits) != 4:
        raise Exception("limits must be length 4 array [x_i, x_f, y_i, y_f]")
    elif limits[0] < limits[1]:
        raise Exception("x_i, x_f are in left handed right ascension coordinates, so x_i > x_f")
    elif limits[2] > limits[3]:
        raise Exception("y_i, y_f are in declination coordinates, so y_i < y_f")


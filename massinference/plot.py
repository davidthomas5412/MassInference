#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common plotting functions.
"""

import numpy as np
from astropy.wcs import WCS


def set_figure_size(fig, fig_size, lx, ly):
    """
    Takes inputted figure and changes its size according to Lx and Ly given in
    inches.

    Args:
        fig (matplotlib.figure.Figure): figure to size
        fig_size (int): total figure size
        lx (float): length of x axis
        ly (float): length of y axis
    """

    if lx > ly:
        fig.set_size_inches(fig_size, fig_size * (1.0 * ly / lx))
    else:
        fig.set_size_inches(fig_size * (1.0 * lx / ly), fig_size)


def set_axes(fig, lx, ly, header, imsubplot):
    """
    Sets wcs and pixel axes for plotting maps and catalogs. Both sets of axes are
    contained in the current figure instance, and also returned for ease of use.

    Args:
        fig (matplotlib.figure.Figure): figure to add axes to
        lx (float): length of x axis
        ly (float): length of y axis
        header (astropy.io.fits.header.Header): fits header for axes projection
        imsubplot (list(float)): list of four limits for plot [x_min, x_max, y_min, y_max]

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
    pix_xi = imsubplot[0]
    pix_xf = imsubplot[1]
    pix_yi = imsubplot[2]
    pix_yf = imsubplot[3]
    ax.set_xlim(pix_xi-0.5, pix_xf-0.5)
    ax.set_ylim(pix_yi-0.5, pix_yf-0.5)

    # Set labels
    ra.set_axislabel('Right Ascension (deg)')
    dec.set_axislabel('Declination (deg)')

    # Set x/y-axis unit formatter
    if 0.03 < lx <= 0.3:
        ra.set_major_formatter('d.dd')
    elif lx <= 0.03:
        ra.set_major_formatter('d.ddd')
    else:
        ra.set_major_formatter('d.d')

    if 0.03 < ly <= 0.3:
        dec.set_major_formatter('d.dd')
    elif ly <= 0.03:
        dec.set_major_formatter('d.ddd')
    else:
        dec.set_major_formatter('d.d')

    # max number of ticks
    num = 8
    if lx > ly:
        ra.set_ticks(number=num)
        dec.set_ticks(number=np.ceil(num * ly / lx))

    elif lx < ly:
        ra.set_ticks(number=np.ceil(num * lx / ly))
        dec.set_ticks(number=num)

    else:
        ra.set_ticks(number=num)
        dec.set_ticks(number=num)

    return ax

def check_limits(limits, coords):

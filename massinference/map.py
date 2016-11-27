#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lens maps for lensing a background.
"""

import struct
import numpy as np


class LensMap(object):
    """
    Stores lensing values in a 2D world-coordinate map.
    Superclass of shearmap and kappamap.
    """
    def __init__(self, mapfiles):
        """
        Args:
             mapfiles (str of list(str)): files to make map from. One file for kappamap,
             list of two files for shearmap.
        """
        if type(mapfiles) != list:
            mapfiles = [mapfiles]

        # Declare needed attributes as lists
        self.hdr = []
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
        for i in range(len(mapfiles)):
            input_parse = mapfiles[i].split(
                '_')  # Creates list of filename elements separated by '_'
            self.map_x.append(int(input_parse[3]))  # The x location of the map grid
            self.map_y.append(int(input_parse[4]))  # The y location of the map grid

            # Initialise some necessary WCS parameters for Stefan
            # Hilbert's binary data files:
            self.field.append(4.0)  # degrees
            self.NX.append(4096)  # pixels
            self.PIXSCALE.append(self.field[i] / (1.0*self.NX[i]))  # degrees
            self.set_wcs(i)

            mapfile = open(mapfiles[i], 'rb')
            data = mapfile.read()
            mapfile.close()
            fmt = str(self.NX[i] * self.NX[i])+'f'
            start = 0
            stop = struct.calcsize(fmt)
            values = struct.unpack(fmt, data[start:stop])
            self.values.append(np.array(values, dtype=np.float32).reshape(self.NX[i], self.NX[i])
                               .transpose())
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

        # Get pixel indices of desired point,
        # and also work out other positions for completeness, if verbose:
        if coordinate_system == 'world':
            i, j = self.world_to_image(x, y, mapfile)
        elif coordinate_system == 'physical':
            i, j = self.physical_to_image(x, y, mapfile)
        elif coordinate_system == 'image':
            i = x
            j = y
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
        return i,j

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


class KappaMap(LensMap):
    def __init__(self):
        raise NotImplementedError()


class ShearMap(LensMap):
    def __init__(self):
        raise NotImplementedError()
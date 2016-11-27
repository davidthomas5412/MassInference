#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lens maps for lensing a background.
"""
import struct
import numpy as np


class LensMap(object):
    """

    """
    def __init__(self, mapfiles):
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
            self.setwcs(i)

            mapfile = open(self.input[i], 'rb')
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
            mapfile (:obj:'int', optional): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.
            coordinate_system (:obj:'str', optional): coordinate system of (x,y). Defaults to
                'world'.

        Returns:
            float: map value at provided coordinates in provided coordinate system.
        """

        # Get pixel indices of desired point,
        # and also work out other positions for completeness, if verbose:
        if coordinate_system == 'world':
            i, j = self.world2image(x, y, mapfile)
        elif coordinate_system == 'physical':
            i, j = self.physical2image(x, y, mapfile)
        elif coordinate_system == 'image':
            i = x
            j = y
        return self.lookup(i, j, mapfile)

    def physical2image(self, x, y, mapfile=0):
        """
        Args:
            x (float): physical x-coordinate
            y (float): physical y-coordinate
            mapfile (:obj:'int', optional): if shearmap then index of shearmap to retrieve values
                from, otherwise ignore.

        Returns:
            float, float: image coordinates
        """
        i = self.wcs[mapfile]['LTV1'] + self.wcs[mapfile]['LTM1_1']*x # x in rad
        j = self.wcs[mapfile]['LTV2'] + self.wcs[mapfile]['LTM2_2']*y # y in rad
        return i,j

    def world2image(self, a, d, mapfile=0):
        """
        Args:
            a (float): physical right-ascension-coordinate
            d (float): world declination-coordinate
            mapfile (:obj:'int', optional): if shearmap then index of shearmap to retrieve values
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

class KappaMap(LensMap):
    def __init__(self):
        raise NotImplementedError()

class ShearMap(LensMap):
    def __init__(self):
        raise NotImplementedError()
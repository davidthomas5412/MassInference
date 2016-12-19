#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import pi


class Angle(object):
    """
    TODO
    """

    def __init__(self, degrees):
        self._degrees = degrees

    @staticmethod
    def radian_to_degree(radian):
        return radian * (180.0 / pi)

    @staticmethod
    def radian_to_arcmin(radian):
        return radian * (180.0 / pi) * 60.0

    @staticmethod
    def arcmin_to_degree(arcmin):
        return arcmin * (1.0 / 60.0)

    @staticmethod
    def arcmin_to_radian(arcmin):
        return arcmin * (pi / (60.0 * 180.0))

    @staticmethod
    def degree_to_radian(degree):
        return degree * (pi / 180.0)

    @staticmethod
    def degree_to_arcmin(degree):
        return degree * 60.0

    @staticmethod
    def from_radian(radians):
        return Angle(Angle.radian_to_degree(radians))

    @staticmethod
    def from_degree(degree):
        return Angle(degree)

    @staticmethod
    def from_arcmin(arcmin):
        return Angle(Angle.arcmin_to_degree(arcmin))

    @property
    def degree(self):
        return self._degrees

    @property
    def radian(self):
        return Angle.degree_to_radian(self._degrees)

    @property
    def arcmin(self):
        return Angle.degree_to_arcmin(self._degrees)

    def __add__(self, other):
        return Angle(self.degree + other.degree)

    def __sub__(self, other):
        return Angle(self.degree - other.degree)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for lens maps.
"""

import unittest
from massinference.map import LensMap, KappaMap, ShearMap
from massinference.data import fetch, KAPPA_FILE, GAMMA_1_FILE, GAMMA_2_FILE

DELTA = 10e-7


class TestMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fetch()
        cls.lm = LensMap([GAMMA_1_FILE, GAMMA_2_FILE])
        cls.km = KappaMap.default()
        cls.sm = ShearMap.default()

    def test_at(self):
        val = self.lm.at(0.0, 0.0, mapfile=1, coordinate_system='world')
        self.assertAlmostEqual(val, -0.0084187807515263557, delta=DELTA)

    def test_world_to_image(self):
        val = self.lm.world_to_image(-1.0, -1.0, mapfile=0)[0]
        self.assertAlmostEqual(val, 3072.0, delta=DELTA)

    def test_physical_to_image(self):
        val = self.lm.physical_to_image(5, 5, mapfile=0)[0]
        self.assertAlmostEqual(val, 295401.89110698149, delta=DELTA)

    # TODO: plotting functional tests
    def test_plotting(self):
        fig, ax, limits = self.km.plot()
        self.sm.plot()
        self.sm.plot()
        self.assertTrue(True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_utils.py - Unittest for utilities
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 09th January 2021, by alessior@kth.se
#
# Copyright [2017-2021] [Alessio Russo - alessior@kth.se]  
# This file is part of PythonVRFT.
# PythonVRFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# PythonVRFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PythonVRFT.  If not, see <http://www.gnu.org/licenses/>.
#

from unittest import TestCase
import numpy as np
import scipy.signal as scipysig
from vrft.utils import *
from vrft.extended_tf import ExtendedTF
from vrft.vrft_algo import virtual_reference
from vrft.iddata import iddata


class TestUtils(TestCase):
    def test_deconvolve(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        sys = ExtendedTF([0.5], [1, -0.9], dt=t_step)
        u = np.random.normal(size=t.size)
        _, y = scipysig.dlsim(sys, u, t)
        y = y[:, 0]
        data = iddata(y, u, t_step, [0])
        r1, _ = virtual_reference(data, sys.num, sys.den)
        r2 = deconvolve_signal(sys, data.y)
        self.assertTrue(np.linalg.norm(r2-r1[:r2.size], np.infty) <  1e-3)


    def test_check_system(self):
        a = [1, 0, 1]
        b = [1, 0, 2]
        self.assertTrue(check_system(a,b))

        b = [1, 0, 2, 4]
        self.assertTrue(check_system(a,b))

        a = [1]
        self.assertTrue(check_system(a,b))

        a = [1, 0, 1]
        b = [1,0]
        with self.assertRaises(ValueError):
            check_system(a,b)

        b = [1]
        with self.assertRaises(ValueError):
            check_system(a,b)

    def test_system_order(self):
        self.assertEqual(system_order(0, 0), (0, 0))
        self.assertEqual(system_order(1, 0), (0, 0))
        self.assertEqual(system_order([1],[1]), (0, 0))
        self.assertEqual(system_order([1, 1],[1, 1]), (1,1))
        self.assertEqual(system_order([1, 1, 3],[1, 1]), (2,1))
        self.assertEqual(system_order([1, 1, 3],[1]), (2,0))
        self.assertEqual(system_order([1, 1],[1, 1, 1]), (1,2))
        self.assertEqual(system_order([1],[1, 1, 1]), (0,2))
        self.assertEqual(system_order([0, 1],[1, 1, 1]), (0,2))
        self.assertEqual(system_order([0,0,1],[1, 1, 1]), (0,2))
        self.assertEqual(system_order([0,0,1],[0, 1, 1, 1]), (0,2))
        self.assertEqual(system_order([0,0,1],[0, 0, 1, 1, 1]), (0,2))

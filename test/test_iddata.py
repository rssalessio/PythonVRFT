# Copyright [2017-2020] [Alessio Russo - alessior@kth.se]  
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
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 06th January 2020, by alessior@kth.se
#

from unittest import TestCase
from vrft.iddata import iddata
import numpy as np

class TestIDData(TestCase):
    def test_type(self):
        a = iddata(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            a.checkData()

        a =  iddata(0.0, [1], 0.0)
        with self.assertRaises(ValueError):
            a.checkData()

        a = iddata(np.zeros(10), 1, 0.0)
        with self.assertRaises(ValueError):
            a.checkData()

        a = iddata([0 for i in range(10)], [0 for i in range(10)], 1.0)
        self.assertTrue(a.checkData())

        a = iddata(np.zeros(10), np.zeros(10), 1.0)
        self.assertTrue(a.checkData())

    def test_size(self):
        a = iddata(np.zeros(10), np.zeros(10), 0.0)
        self.assertEqual(len(a.y), 10)
        self.assertEqual(len(a.u), 10)
        self.assertEqual(len(a.y), len(a.u))

        a = iddata([0 for i in range(10)], [1 for i in range(0,10)], 0.0)
        self.assertEqual(len(a.y), 10)
        self.assertEqual(len(a.u), 10)
        self.assertEqual(len(a.y), len(a.u))

        a = iddata(np.zeros(10), np.zeros(9), 0.0)
        with self.assertRaises(ValueError):
            a.checkData()

        a = iddata(np.zeros(8), np.zeros(9), 0.0)
        with self.assertRaises(ValueError):
            a.checkData()


    def test_sampling_time(self):
        a = iddata(np.zeros(10), np.zeros(10), 0.0)
        with self.assertRaises(ValueError):
            a.checkData()

        a = iddata(np.zeros(10), np.zeros(10), 1e-9)
        with self.assertRaises(ValueError):
            a.checkData()

        a = iddata(np.zeros(10), np.zeros(10), -0.1)
        with self.assertRaises(ValueError):
            a.checkData()

        a = iddata(np.zeros(10), np.zeros(10), 0.1)
        self.assertTrue(a.checkData())


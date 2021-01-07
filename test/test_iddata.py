# test_iddata.py - Unittest for the iddata object
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2020, by alessior@kth.se
#
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


from unittest import TestCase
from vrft.iddata import iddata
import numpy as np
import scipy.signal as scipysig

class TestIDData(TestCase):
    def test_type(self):
        a = iddata(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            a.check()

        a =  iddata(0.0, [1], 0.0)
        with self.assertRaises(ValueError):
            a.check()

        a = iddata(np.zeros(10), 1, 0.0)
        with self.assertRaises(ValueError):
            a.check()

        a = iddata([0 for i in range(10)], [0 for i in range(10)], 1.0)
        self.assertTrue(a.check())

        a = iddata(np.zeros(10), np.zeros(10), 1.0)
        self.assertTrue(a.check())

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
            a.check()

        a = iddata(np.zeros(8), np.zeros(9), 0.0)
        with self.assertRaises(ValueError):
            a.check()


    def test_sampling_time(self):
        a = iddata(np.zeros(10), np.zeros(10), 0.0)
        with self.assertRaises(ValueError):
            a.check()

        a = iddata(np.zeros(10), np.zeros(10), 1e-9)
        with self.assertRaises(ValueError):
            a.check()

        a = iddata(np.zeros(10), np.zeros(10), -0.1)
        with self.assertRaises(ValueError):
            a.check()

        a = iddata(np.zeros(10), np.zeros(10), 0.1)
        self.assertTrue(a.check())

    def test_copy(self):
        a = iddata(np.zeros(10), np.zeros(10), 0.1)
        b = a.copy()
        self.assertTrue(a.check())
        self.assertTrue(b.check())

        self.assertTrue(np.all(a.y == b.y))
        self.assertTrue(np.all(a.u == b.u))
        self.assertTrue(np.all(a.y0 == b.y0))
        self.assertTrue(a.ts == b.ts)

    def test_filter(self):
        a = iddata(np.zeros(10), np.zeros(10), 0.1)
        L = scipysig.dlti([1], [1], dt=0.1)
        b = a.copy()
        a.filter(L)
        self.assertTrue(np.all(a.y == b.y))
        self.assertTrue(np.all(a.u == b.u))
        self.assertTrue(np.all(a.y0 == b.y0))
        self.assertTrue(a.ts == b.ts)

    def test_split(self):
        n = 10
        a = iddata(np.random.normal(size=n), np.random.normal(size=n), 0.1)

        b, c = a.split()

        self.assertTrue(b.y.size == c.y.size)
        self.assertTrue(b.u.size == c.u.size)
        self.assertTrue(b.y0 == c.y0)
        self.assertTrue(b.ts == c.ts)
        self.assertTrue(b.ts == a.ts)
        self.assertTrue(np.all(b.y == a.y[:n//2]))
        self.assertTrue(np.all(b.u == a.u[:n//2]))
        self.assertTrue(b.y0 == a.y0)

        self.assertTrue(np.all(c.y == a.y[n//2:]))
        self.assertTrue(np.all(c.u == a.u[n//2:]))
        self.assertTrue(c.y0 == a.y0)

        y0 = [-1, 2]
        a = iddata(np.random.normal(size=n), np.random.normal(size=n), 0.1, y0)
        n0 = len(y0)
        n1 = (n + n0) // 2
        b, c = a.split()

        self.assertTrue(b.y.size == c.y.size)
        self.assertTrue(b.u.size == c.u.size)
        self.assertTrue(b.ts == c.ts)
        self.assertTrue(b.ts == a.ts)
        self.assertTrue(np.all(b.y == a.y[:n1 - n0]))
        self.assertTrue(np.all(b.u == a.u[:n1 - n0]))
        self.assertTrue(np.all(b.y0 == a.y0))

        self.assertTrue(np.all(c.y == a.y[n1:]))
        self.assertTrue(np.all(c.u == a.u[n1:]))
        self.assertTrue(np.all(c.y0 == a.y[n1 - n0:n1]))


        y0 = [-1, 2]
        n = 9
        a = iddata(np.random.normal(size=n), np.random.normal(size=n), 0.1, y0)
        n0 = len(y0)
        n -= 1
        n1 = (n + n0) // 2
        b, c = a.split()

        self.assertTrue(b.y.size == c.y.size)
        self.assertTrue(b.u.size == c.u.size)
        self.assertTrue(b.ts == c.ts)
        self.assertTrue(b.ts == a.ts)
        self.assertTrue(np.all(b.y == a.y[:n1 - n0]))
        self.assertTrue(np.all(b.u == a.u[:n1 - n0]))
        self.assertTrue(np.all(b.y0 == a.y0))

        self.assertTrue(np.all(c.y == a.y[n1:n]))
        self.assertTrue(np.all(c.u == a.u[n1:n]))
        self.assertTrue(np.all(c.y0 == a.y[n1 - n0:n1]))

        y0 = [-1]
        n = 10
        a = iddata(np.random.normal(size=n), np.random.normal(size=n), 0.1, y0)
        n0 = len(y0)
        n -= 1
        n1 = (n + n0) // 2
        b, c = a.split()

        self.assertTrue(b.y.size == c.y.size)
        self.assertTrue(b.u.size == c.u.size)
        self.assertTrue(b.ts == c.ts)
        self.assertTrue(b.ts == a.ts)
        self.assertTrue(np.all(b.y == a.y[:n1 - n0]))
        self.assertTrue(np.all(b.u == a.u[:n1 - n0]))
        self.assertTrue(np.all(b.y0 == a.y0))

        self.assertTrue(np.all(c.y == a.y[n1:n]))
        self.assertTrue(np.all(c.u == a.u[n1:n]))
        self.assertTrue(np.all(c.y0 == a.y[n1 - n0:n1]))
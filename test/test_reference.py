# test_reference.py - Unittest for virtual reference algorithm
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2021, by alessior@kth.se
#
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
from vrft.iddata import *
from vrft.utils import *
from vrft.vrft_algo import *
import numpy as np
import scipy.signal as scipysig

class TestReference(TestCase):
    def test_virtualReference(self):
        # wrong system
        with self.assertRaises(ValueError):
            virtualReference(1, 1, 0)

        # cant be constant the system
        with self.assertRaises(ValueError):
            virtualReference([1],[1], 0)

        # cant be constant the system
        with self.assertRaises(ValueError):
            virtualReference(np.array(2), np.array(3), 0)

        # wrong data
        with self.assertRaises(ValueError):
            virtualReference([1], [1, 1], 0)

        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.ones(len(t)).tolist()

        num = [0.1]
        den = [1, -0.9]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t,y = scipysig.dlsim(sys, u, t)
        y = y[:,0]
        data = iddata(y,u,t_step,[0,0])
        
        # wrong initial conditions
        with self.assertRaises(ValueError):
            r, _ = virtualReference(data, num, den)
    
        #test good data, first order
        data = iddata(y,u,t_step,[0])

        r, _ = virtualReference(data, num, den)

        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))


        num = [0, 1-1.6+0.63]
        den = [1, -1.6, 0.63]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t, y = scipysig.dlsim(sys, u, t)
        y = y[:,0]
        data = iddata(y,u,t_step,[0,0])
        #test second order
        r, _ = virtualReference(data, num, den)
        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))
 
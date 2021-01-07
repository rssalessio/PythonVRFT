# test_vrft.py - Unittest for VRFT
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
from vrft.iddata import *
from vrft.vrft_algo import *
from vrft.extended_tf import ExtendedTF
import numpy as np
import scipy.signal as scipysig

class TestVRFT(TestCase):
    def test_vrft(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.ones(len(t)).tolist()

        num = [0.1]
        den = [1, -0.9]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t, y = scipysig.dlsim(sys, u, t)
        y = y[:,0]
        data = iddata(y,u,t_step,[0])

        refModel = ExtendedTF([0.2], [1, -0.8], dt=t_step)
        prefilter = refModel * (1-refModel)

        control = [ExtendedTF([1], [1,0], dt=t_step),
                ExtendedTF([1], [1,0,0], dt=t_step),
                ExtendedTF([1], [1,0,0,0], dt=t_step),
                ExtendedTF([1, 0], [1,1], dt=t_step)]

        theta1, _, loss1, _ = compute_vrft(data, refModel, control, prefilter)
        theta2, _, loss2, _ = compute_vrft([data], refModel, control, prefilter)
        theta3, _, loss3, _ = compute_vrft([data, data], refModel, control, prefilter)
        
        self.assertTrue(np.isclose(loss1, loss2))
        self.assertTrue(np.isclose(loss1, loss3))
        self.assertTrue(np.linalg.norm(theta1-theta2)<1e-15)
        self.assertTrue(np.linalg.norm(theta1-theta3)<1e-15)

    def test_iv(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        u = np.ones(len(t)).tolist()

        num = [0.1]
        den = [1, -0.9]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t, y = scipysig.dlsim(sys, u, t)
        y = y.flatten() + np.random.normal(size=t.size)
        data = iddata(y,u,t_step,[0])

        refModel = ExtendedTF([0.2], [1, -0.8], dt=t_step)
        prefilter = refModel * (1-refModel)

        control = [ExtendedTF([1], [1,0], dt=t_step),
                ExtendedTF([1], [1,0,0], dt=t_step),
                ExtendedTF([1], [1,0,0,0], dt=t_step),
                ExtendedTF([1, 0], [1,1], dt=t_step)]

        #import pdb
        #pdb.set_trace()
        
        #theta, _, loss, _ = compute_vrft(data, refModel, control, prefilter, iv=True)




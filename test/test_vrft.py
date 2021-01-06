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

        control = [ExtendedTF([1], [1,0], dt=t_step),
                ExtendedTF([1], [1,0,0], dt=t_step),
                ExtendedTF([1], [1,0,0,0], dt=t_step),
                ExtendedTF([1, 0], [1,1], dt=t_step)]

        theta, _, _, loss, _ = compute_vrft(data, refModel, control, refModel * (1-refModel))




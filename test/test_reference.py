# test_reference.py - Unittest for virtual reference algorithm
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 10th January 2021, by alessior@kth.se
#
# Copyright (c) [2017-2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of PythonVRFT.
# PythonVRFT is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with PythonVRFT.
# If not, see <https://opensource.org/licenses/MIT>.
#

from unittest import TestCase
import numpy as np
import scipy.signal as scipysig
from vrft.iddata import *
from vrft.utils import *
from vrft.vrft_algo import *


class TestReference(TestCase):
    def test_virtualReference(self):
        # wrong system
        with self.assertRaises(ValueError):
            virtual_reference(1, 1, 0)

        # cant be constant the system
        with self.assertRaises(ValueError):
            virtual_reference([1],[1], 0)

        # cant be constant the system
        with self.assertRaises(ValueError):
            virtual_reference(np.array(2), np.array(3), 0)

        # wrong data
        with self.assertRaises(ValueError):
            virtual_reference([1], [1, 1], 0)

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
            r, _ = virtual_reference(data, num, den)
    
        #test good data, first order
        data = iddata(y,u,t_step,[0])

        r, _ = virtual_reference(data, num, den)

        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))


        num = [1-1.6+0.63]
        den = [1, -1.6, 0.63]
        sys = scipysig.TransferFunction(num, den, dt=t_step)
        t, y = scipysig.dlsim(sys, u, t)
        y = y[:,0]
        data = iddata(y,u,t_step,[0,0])
        #test second order
        r, _ = virtual_reference(data, num, den)
        for i in range(len(r)):
            self.assertTrue(np.isclose(r[i], u[i]))
 
# test_vrft.py - Unittest for VRFT
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
from vrft.vrft_algo import *
from vrft.extended_tf import ExtendedTF


class TestVRFT(TestCase):
    def test_vrft(self):
        t_start = 0
        t_step = 1e-2
        t_ends = [10, 10 + t_step]

        expected_theta = np.array([1.93220784, -1.05808206, 1.26623764, 0.0088772])
        expected_loss = 0.00064687904235295
        
        for t_end in t_ends:
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
            self.assertTrue(np.linalg.norm(theta1-expected_theta, np.infty) < 1e-5)
            self.assertTrue(abs(expected_loss - loss1) < 1e-5)

    def test_iv(self):
        t_start = 0
        t_step = 1e-2
        t_ends = [10, 10 + t_step]

        
        for t_end in t_ends:
            t = np.arange(t_start, t_end, t_step)
            u = np.ones(len(t)).tolist()

            num = [0.1]
            den = [1, -0.9]
            sys = scipysig.TransferFunction(num, den, dt=t_step)
            _, y = scipysig.dlsim(sys, u, t)
            y = y.flatten() + 1e-2 * np.random.normal(size=t.size)
            data1 = iddata(y,u,t_step,[0])

            _, y = scipysig.dlsim(sys, u, t)
            y = y.flatten() + 1e-2 * np.random.normal(size=t.size)
            data2 = iddata(y,u,t_step,[0])
            

            refModel = ExtendedTF([0.2], [1, -0.8], dt=t_step)
            prefilter = refModel * (1-refModel)

            control = [ExtendedTF([1], [1,0], dt=t_step),
                    ExtendedTF([1], [1,0,0], dt=t_step),
                    ExtendedTF([1], [1,0,0,0], dt=t_step),
                    ExtendedTF([1, 0], [1,1], dt=t_step)]

            with self.assertRaises(ValueError):
                compute_vrft(data1, refModel, control, prefilter, iv=True)

            compute_vrft([data1, data2], refModel, control, prefilter, iv=True)

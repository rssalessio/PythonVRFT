from unittest import TestCase
from vrft.utilities.iddata import *
from vrft.vrft.vrft_algo import *
from vrft.utilities.utils import ExtendedTF
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




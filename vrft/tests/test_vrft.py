from unittest import TestCase
from vrft.utilities.iddata import *
from vrft.vrft.reference import *
from vrft.vrft.vrft_algo import *
import numpy as np
import control as ctl

class TestVRFT(TestCase):
	def test_vrft(self):
		t_start = 0
		t_end = 10
		t_step = 1e-2
		t = np.arange(t_start, t_end, t_step)
		u = np.ones(len(t)).tolist()

		num = [0.1]
		den = [1, -0.9]
		sys = ctl.tf(num, den, t_step)
		y,t,x= ctl.lsim(sys, u, t)
		data = iddata(y[0],u,t_step,[0])

		refModel = ctl.tf([0.2], [1, -0.8], t_step)

		base = [ctl.tf([1], [1,0],t_step),
				ctl.tf([1], [1,0,0],t_step),
				ctl.tf([1], [1,0,0,0],t_step),
		        ctl.tf([1, 0], [1,1],t_step)]
		theta = vrftAlgorithm(data, refModel, base)




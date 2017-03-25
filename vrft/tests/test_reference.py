from unittest import TestCase
from vrft.utilities.iddata import iddata
from vrft.vrft.reference import virtualReference
import numpy as np
import control as ctl

class TestReference(TestCase):
	def test_virtualReference(self):
		with self.assertRaises(ValueError):
			virtualReference(1, 1, 0)

		with self.assertRaises(ValueError):
			virtualReference([1],[1], 0)

		with self.assertRaises(ValueError):
			virtualReference(np.array(2), np.array(3), 0)

		with self.assertRaises(ValueError):
			virtualReference([1],[1,1], 0)

		t_start = 0
		t_end = 10
		t_step = 1e-2
		t = np.arange(t_start, t_end, t_step)
		u = np.ones(len(t)).tolist()

		num = [0.1]
		den = [1, -0.9]
		sys = ctl.tf(num, den, t_step)
		y,t,x= ctl.lsim(sys, u, t)
		data = iddata(y[0],u,t_step)
		virtualReference(num, den, data)

		num = [0, 1-1.6+0.63]
		den = [1, -1.6, 0.63]
		sys = ctl.tf(num, den, t_step)
		y,t,x= ctl.lsim(sys, u, t)
		data = iddata(y[0],u,t_step)





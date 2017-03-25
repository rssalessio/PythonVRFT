import numpy as np


class iddata:
	y = None
	u = None
	ts = None
	
	def __init__(self, y=None, u=None, ts=None):
		self.y = y
		self.u = u
		self.ts = ts

	def checkData(self):
		if (type(u) is not list) and (type(u) is not np.array):
			raise ValueError("Input signal is not an array.")

		if (type(y) is not list) and (type(y) is not np.array):
			raise ValueError("Input signal is not an array.")

		if (len(y) != len(u)):
			raise ValueError("Input and output size do not match.")

		if (type(ts) is not float):
			raise ValueError("Sampling time is not float.")

		if (np.isclose(ts, 0.0) == True):
			raise ValueError("Sampling time can not be zero.")
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
		if (type(self.y) is np.ndarray):
			self.y = self.y.tolist()

		if (type(self.u) is np.ndarray):
			self.u = self.u.tolist()

		if (type(self.u) is not list):
			raise ValueError("Input signal is not an array.")

		if (type(self.y) is not list):
			raise ValueError("Input signal is not an array.")

		if (len(self.y) != len(self.u)):
			raise ValueError("Input and output size do not match.")

		if (type(self.ts) is not float):
			raise ValueError("Sampling time is not float.")

		if (np.isclose(self.ts, 0.0) == True):
			raise ValueError("Sampling time can not be zero.")

		if (self.ts < 0.0):
			raise ValueError("Sampling time can not be negative.")

		return True
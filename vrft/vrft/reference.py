from vrft.utilities.iddata import iddata
from vrft.utilities.tf import *
import numpy as np
import scipy.signal as sig
import types

#todo: add initial conditions for the systems, right now is set to 0
def virtualReference(num, den, data):
	try:
		checkSystem(num, den)
	except ValueError:
		raise

	M,N= systemOrder(num, den)

	if (N == 0) and (M == 0):
		raise ValueError("The reference model can not be a constant.")

	if (not isinstance(data, iddata)):
		raise ValueError("The passed data is not of type: ", iddata.__name__)

	try:
		data.checkData()
	except ValueError:
		raise

	lag = N-M #number of initial conditions 

	if (lag > 0 and data.y0 == None):
		raise ValueError("Wrong initial condition.")

	if (lag != len(data.y0)):
		raise ValueError("Wrong initial condition size.")

	data_size = [0 for i in range(len(data.y))]

	return True
	for i in range(0, data_size):
		left_side = 0
		r = 0
		first = True



		for j in range(0, N):
			if (i-j >= 0):
				left_side += den[j]*data.y[i-j]
		for j in range(0, M):
			if (i-j >= 0):
				if (num[j] != 0 and first == True):
					r = num[j]
				else:
					left_side -= num[j]*reference[i-j]
		rk = left_side / num[0]

	return reference
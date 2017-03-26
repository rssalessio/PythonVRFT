from vrft.utilities.iddata import *
from vrft.utilities.tf import *
import numpy as np
import types

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

	offset_M = len(num) - M - 1
	offset_N = len(den) - N - 1

	lag = N-M #number of initial conditions 

	if (lag > 0 and data.y0 == None):
		raise ValueError("Wrong initial condition.")

	if (lag != len(data.y0)):
		raise ValueError("Wrong initial condition size.")

	reference = [0 for i in range(len(data.y))]

	for k in range(0, len(data.y)):
		left_side = 0
		r = 0

		start_i = 0  if k >= M else M-k
		start_j = 0  if k >= N else N-k

		for i in range(start_i, N+1):
			index = k+i-N
			if (index < 0):
				left_side += den[offset_N + abs(i-N)]*data.y0[abs(index)-1]
			else:
				left_side += den[offset_N + abs(i-N)]*data.y[index]

		for j in range(start_j, M+1):
			index = k+j-N
			if (start_j != M):
				left_side += -num[offset_M + abs(j-M)]*reference[index]
			else:
				r = num[offset_M]

		if (np.isclose(r, 0.0) != True) :
			reference[k-lag] = left_side/r
		else:
			reference[k-lag] = 0.0

	#add missing data..just copy last N-M points
	for i in range(lag):
		reference[len(data.y)+i-lag] =reference[len(data.y)+i-1-lag]

	return reference
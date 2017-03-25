import collections
import numpy as np
import scipy.signal as sig


def virtualReference(num, den, data):
	try:
		checkSystem(num, den)
	except ValueError:
		raise

	N, M = systemOrder(num, den)

	if (N == 1) and (M == 1):
		raise ValueError("The reference model can not be a constant.")

	if (type(data) is not iddata):
		raise ValueError("The passed data is not of type: ".iddata)

	try:
		data.checkData()
	except ValueError:
		raise

	lag = N-M
	data_size = []

	reference  = [0 for i in range(data_size)]

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



d = iddata(1, 1, 0.1)
c = iddata()

virtualReference([1],[1],d)
virtualReference(1,1,c)
print c.ts
import numpy as np

def checkSystem(num, den):
	try:
		N, M = systemOrder(num, den)
	except ValueError:
		raise

	if (N < M):
		raise ValueError("The system is not causal.")



def systemOrder(num, den):
	if (type(num) is not list) and (type(num) is not np.array):
		raise ValueError("Numerator is not an array.")

	if (type(den) is not list) and (type(den) is not np.array):
		raise ValueError("Denominator is not an array.")

	N = len(den)
	M = len(num)

	denOrder = -1
	numOrder = -1

	for i in range (0,N):
		if (den[i] != 0):
			denOrder = N - i - 1
			break

	for i in range (0,M):
		if num[i] != 0:
			numOrder = M - i - 1
			break

	if (denOrder == -1):
		raise ValueError("Denominator can not be zero.")

	if (numOrder == -1):
		raise ValueError("Numerator can not be zero.")

	return denOrder, numOrder

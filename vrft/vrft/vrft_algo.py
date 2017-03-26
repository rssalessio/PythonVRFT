import numpy as np
from numpy.linalg import inv
import control as ctl
from vrft.utilities.iddata import *
from vrft.vrft.reference import *


def calcFinalController(theta, base):
	return np.dot(theta, base).item((0,0))

def calcMinimum(phi, data):
	phi = np.mat(phi)

	#least squares
	theta = inv(phi * phi.T)*phi
	theta = np.dot(theta, data.u)

	return theta

def calcControllerResponse(base, error, data):
	t_start = 0
	t_step = data.ts
	t_end = len(data.y)*t_step
	

	t =  np.arange(t_start, t_end, t_step)

	phi = np.zeros((len(base), len(t)))
	
	for i in range(len(base)):
		y, t, x = ctl.lsim(base[i], error.tolist(), t)
		phi[i, :] = y

	return phi

def vrftAlgorithm(data, referenceModel, base):
	if (not isinstance(data, iddata)):
		raise ValueError("The passed data is not of type: ", iddata.__name__)

	if (not isinstance(referenceModel, ctl.TransferFunction)):
		raise ValueError("The passed reference model is not of type: ", ctl.TransferFunction.__name__)

	if (type(base) is not list):
		raise ValueError("Should pass a list of controllers.")

	for i in range(len(base)):
		if (not isinstance(base[i], ctl.TransferFunction)):
			raise ValueError("Some of the components of the controller are not transfer functions.")

	r = virtualReference(referenceModel.num[0][0], referenceModel.den[0][0], data)
	
	e =  np.subtract(r,data.y)

	phi = calcControllerResponse(base, e, data)

	theta = calcMinimum(phi, data)

	return calcFinalController(theta, base), theta





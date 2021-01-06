from vrft.utilities.iddata import iddata
from vrft.utilities.utils import systemOrder, checkSystem, filter_iddata, deconvolve_signal
import numpy as np
import scipy.signal as scipysig

def virtualReference(data: iddata, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    try:
        checkSystem(num, den)
    except ValueError:
        raise ValueError('Error in check system')

    M, N = systemOrder(num, den)

    if (N == 0) and (M == 0):
        raise ValueError("The reference model can not be a constant.")

    data.checkData()
    offset_M = len(num) - M - 1
    offset_N = len(den) - N - 1

    lag = N - M  #number of initial conditions

    if (lag > 0 and data.y0 is None):
        raise ValueError("Wrong initial condition.")

    if (lag != len(data.y0)):
        raise ValueError("Wrong initial condition size.")

    reference = np.zeros_like(data.y)
    L = len(data.y)

    for k in range(0, len(data.y) + lag):
        left_side = 0
        r = 0

        start_i = 0 if k >= M else M - k
        start_j = 0 if k >= N else N - k

        for i in range(start_i, N + 1):
            index = k + i - N
            if (index < 0):
                left_side += den[offset_N +
                                 abs(i - N)] * data.y0[abs(index) - 1]
            else:
                left_side += den[offset_N + abs(i - N)] * (
                    data.y[index] if index < L else 0)

        for j in range(start_j, M + 1):
            index = k + j - N
            if (start_j != M):
                left_side += -num[offset_M + abs(j - M)] * reference[index]
            else:
                r = num[offset_M]

        if (np.isclose(r, 0.0) != True):
            reference[k - lag] = left_side / r
        else:
            reference[k - lag] = 0.0

    #add missing data..just copy last N-M points
    #for i in range(lag):
    #    reference[len(self.data.y)+i-lag] =0 #reference[len(self.data.y)+i-1-lag]

    return reference[:-lag], len(reference[:-lag])


def compute_vrft_loss(data: iddata, phi: np.ndarray, theta: np.ndarray):
    z = np.dot(phi, theta.T).flatten()
    L = z.size
    return np.linalg.norm(data.u[:L] - z)**2 / L

def calc_minimum(data: iddata, phi: np.ndarray):
    phi = np.mat(phi)
    nk = phi.shape[1]
    #least squares
    theta = np.linalg.inv(phi.T @ phi) @ phi.T

    L = theta.shape[1]
    theta = np.array(np.dot(theta, data.u[:L])).flatten()
    return theta, phi

def control_response(data: iddata, error: np.ndarray, control: list):
    t_start = 0
    t_step = data.ts
    t_end = len(error) * t_step
    t = np.arange(t_start, t_end, t_step)

    phi = np.zeros((len(control), len(t)))
    for i in range(len(control)):
        t, y = scipysig.dlsim(control[i], error, t)
        phi[i, :] = y[:, 0]

    phi = phi.T
    return phi

def compute_vrft(data: iddata, refModel: scipysig.dlti, control: list, L: scipysig.dlti):
    # import pdb
    # import matplotlib.pyplot as plt
    # pdb.set_trace()
    data = filter_iddata(data, L)
    r, n = virtualReference(data,
                         refModel.num,
                         refModel.den)
    # r2=deconvolve_signal(refModel, data.y, data.ts)
    # plt.plot(r)
    # plt.plot(r2)
    # plt.show()

    phi = control_response(data, np.subtract(r, data.y[:n]), control)
    theta, phi = calc_minimum(data, phi)
    loss = compute_vrft_loss(data, phi, theta)
    print(phi)
    #theta[0] = -theta[0]
    final_control = np.dot(theta, control)
    return theta, r, phi, loss, final_control

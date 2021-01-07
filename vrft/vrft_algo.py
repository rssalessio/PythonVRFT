# vrft_algo.py - VRFT algorithm implementation
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2020, by alessior@kth.se
#
# Copyright(c) [2017-2020] [Alessio Russo - alessior@kth.se]  
# This file is part of PythonVRFT.
# PythonVRFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# PythonVRFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PythonVRFT.  If not, see <http://www.gnu.org/licenses/>.
#


from vrft.iddata import iddata
from vrft.utils import systemOrder, checkSystem, \
    filter_signal
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

    data.check()
    offset_M = len(num) - M - 1
    offset_N = len(den) - N - 1

    lag = N - M  # number of initial conditions
    y0 = data.y0

    if y0 is None:
        y0 = [0.] * lag

    if y0 is not None and (lag != len(y0)):
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
                                 abs(i - N)] * y0[abs(index) - 1]
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

def calc_minimum(data: iddata, phi1: np.ndarray,
                 phi2: np.ndarray = None):
    """Compute least squares minimum
    Parameters
    ----------
    data : iddata
        iddata object containing data from experiments
    phi1 : np.ndarray
        Regressor
    phi2 : np.ndarray, optional
        Second regressor (used only with instrumental variables)

    Returns
    -------
    theta : np.ndarray
        Coefficients computed for the control basis
    """
    phi1 = np.mat(phi1)
    phi2 = np.mat(phi2) if phi2 is not None else phi1
    nk = phi1.shape[1]
    #least squares
    theta = np.linalg.inv(phi2.T @ phi1) @ phi2.T

    L = theta.shape[1]
    theta = np.array(np.dot(theta, data.u[:L])).flatten()
    return theta

def control_response(data: iddata, error: np.ndarray, control: list):
    t_start = 0
    t_step = data.ts
    t_end = len(error) * t_step
    t = np.arange(t_start, t_end, t_step)

    phi = np.zeros((len(control), len(t)))
    for i in range(len(control)):

        t, y = scipysig.dlsim(control[i], error, t)
        if y.size != error.size:
            phi[i, :] = [0, y.flatten()]
        else:
            phi[i, :] = y.flatten()

    phi = phi.T
    return phi

def compute_vrft(data: iddata, refModel: scipysig.dlti,
                 control: list, prefilter: scipysig.dlti = None,
                 iv: bool =  False):
    """Compute VRFT Controller
    Parameters
    ----------
    data : iddata or list of iddata objects
        Data used to identify theta.
        - If data is an iddata object and iv is set to True,
          then the data will be split into half in order
          to compute the instrumental variable.

        - If data a list of iddata objects and iv is False, then
           only the first element will be used to identify theta. 
           If iv is True the first two elements will be used.
    refModel : scipy.signal.dlti
        Discrete Transfer Function representing the reference model
    control : list
        list of discrete transfer functions, representing the control basis
    prefilter : scipy.signal.dlti, optional
        Filter used to pre-filter the data
    iv : bool, optiona;
        Instrumental variable option. If true, the dataset will be split in two, and
        the instrumental variable will be constructed based on the second half of the
        dataset 

    Returns
    -------
    theta : np.ndarray
        Coefficients computed for the control basis
    r : np.ndarray
        Virtual reference signal
    loss: float
        VRFT loss
    final_control: scipy.signal.dlti
        Final controller
    """
    if not isinstance(data, iddata):
        if not isinstance(data, list):
            raise ValueError('data should be an iddata object or a list of iddata objects')
        else:
            for d in data:
                if not isinstance(d, iddata):
                    raise ValueError('data should be a list of iddata objects')

    # Prefilter the data
    if prefilter is not None and isinstance(prefilter, scipysig.dlti):
        if isinstance(data, list):
            for i, d in enumerate(data):
                data[i] = d.copy().filter(prefilter)
        else:
            data = data.copy().filter(prefilter)


    if not iv:
        if isinstance(data, list):
            data = data[0]
        data.check()

        # Compute virtual reference
        r, n = virtualReference(data, refModel.num, refModel.den)

        # Compute control response given the virtual reference
        phi = control_response(data, np.subtract(r, data.y[:n]), control)

        # Compute MSE minimizer
        theta = calc_minimum(data, phi)
    else:
        # Retrieve the two datasets
        if isinstance(data, list):
            d1 = data[0]
            d2 = data[1]
            d1.check()
            d2.check()
            # check if the two datasets have same size
            if d1.y.size != d2.y.size:
                raise ValueError('The two datasets should have same size!')
        else:
            d1, d2 = data.split()
        r1, n1 = virtualReference(d1, refModel.num, refModel.den)
        r2, n2 = virtualReference(d2, refModel.num, refModel.den)
        phi1 = control_response(d1, np.subtract(r1, d1.y[:n1]), control)
        phi2 = control_response(d2, np.subtract(r2, d2.y[:n2]), control)
        theta = calc_minimum(data, phi1, phi2)

    # Compute VRFT loss
    loss = compute_vrft_loss(data, phi, theta)
    # Final controller
    final_control = np.dot(theta, control)

    return theta, r, loss, final_control

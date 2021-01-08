# vrft_algo.py - VRFT algorithm implementation
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2021, by alessior@kth.se
#
# Copyright(c) [2017-2021] [Alessio Russo - alessior@kth.se]  
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

from typing import overload
import numpy as np
import scipy as sp
import scipy.signal as scipysig

from vrft.iddata import iddata
from vrft.utils import system_order, check_system, \
    filter_signal


@overload 
def virtual_reference(data: iddata, L: scipysig.dlti) -> np.ndarray:
    """Compute virtual reference signal by performing signal deconvolution
    Parameters
    ----------
    data : iddata
        iddata object containing data from experiments
    L : scipy.signal.dlti
        Discrete transfer function

    Returns
    -------
    r : np.ndarray
        virtual reference signal
    """
    return virtual_reference(data, L.num, L.den)


def virtual_reference(data: iddata, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Compute virtual reference signal by performing signal deconvolution
    Parameters
    ----------
    data : iddata
        iddata object containing data from experiments
    num : np.ndarray
        numerator of a discrete transfer function
    phi2 : np.ndarray
        denominator of a discrete transfer function

    Returns
    -------
    r : np.ndarray
        virtual reference signal
    """
    try:
        check_system(num, den)
    except ValueError:
        raise ValueError('Error in check system')

    M, N = system_order(num, den)

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


def compute_vrft_loss(data: iddata, phi: np.ndarray, theta: np.ndarray) -> float:
    z = np.dot(phi, theta.T).flatten()
    return np.linalg.norm(data.u[:z.size] - z) ** 2 / z.size

def calc_minimum(data: iddata, phi1: np.ndarray,
                 phi2: np.ndarray = None) -> np.ndarray:
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
    phi1 = np.array(phi1)
    L = phi1.shape[0]
    if phi2 is None:
        theta, _, _, _ = sp.linalg.lstsq(phi1, data.u[:L], lapack_driver='gelsy')
    else:
        phi2 = np.array(phi2)
        theta = (np.linalg.inv(phi2.T @ phi1) @ phi2.T).dot(data.u[:L])
    return theta.flatten()

def control_response(data: iddata, error: np.ndarray, control: list) -> np.ndarray:
    t_step = data.ts
    t = [i * t_step for i in range(len(error))]

    phi = [None] * len(control)
    for i, c in enumerate(control):
        _, y = scipysig.dlsim(c, error, t)
        phi[i] = y.flatten()

    phi = np.vstack(phi).T
    return phi

def compute_vrft(data: iddata, refModel: scipysig.dlti,
                 control: list, prefilter: scipysig.dlti = None,
                 iv: bool =  False):
    """Compute VRFT Controller
    Parameters
    ----------
    data : iddata or list of iddata objects
        Data used to identify theta. If iv is set to true,
        then the algorithm expects a list of 2 iddata objects
    refModel : scipy.signal.dlti
        Discrete Transfer Function representing the reference model
    control : list
        list of discrete transfer functions, representing the control basis
    prefilter : scipy.signal.dlti, optional
        Filter used to pre-filter the data
    iv : bool, optiona;
        Instrumental variable option. If true, the instrumental variable will 
        be constructed based on two iddata objets

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

    # Check the data
    if not isinstance(data, iddata):
        if not isinstance(data, list):
            raise ValueError('data should be an iddata object or a list of iddata objects')
        else:
            if iv and len(data) != 2:
                raise ValueError('data should be a list of 2 iddata objects')

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
        # No instrumental variable routine
        if isinstance(data, list):
            data = data[0]
        data.check()

        # Compute virtual reference
        r, n = virtual_reference(data, refModel.num, refModel.den)

        # Compute control response given the virtual reference
        phi = control_response(data, np.subtract(r, data.y[:n]), control)

        # Compute MSE minimizer
        theta = calc_minimum(data, phi)
    else:
        # Instrumental variable routine

        # Retrieve the two datasets
        if isinstance(data, list):
            d1 = data[0]
            d2 = data[1]
            # check if the two datasets have same size
            if d1.y.size != d2.y.size:
                raise ValueError('The two datasets should have same size!')
        else:
            raise ValueError('To use IV the data should be a list of iddata objects')

        # Compute virtual reference
        r1, n1 = virtual_reference(d1, refModel.num, refModel.den)
        r2, n2 = virtual_reference(d2, refModel.num, refModel.den)

        # Compute control response
        phi1 = control_response(d1, np.subtract(r1, d1.y[:n1]), control)
        phi2 = control_response(d2, np.subtract(r2, d2.y[:n2]), control)

        # We use the first dataset to compute statistics (e.g. VRFT Loss)
        phi = phi1
        data = data[0]
        r = r1

        # Compute MSE minimizer
        theta = calc_minimum(data, phi1, phi2)

    # Compute VRFT loss
    loss = compute_vrft_loss(data, phi, theta)

    # Final controller
    final_control = np.dot(theta, control)

    return theta, r, loss, final_control

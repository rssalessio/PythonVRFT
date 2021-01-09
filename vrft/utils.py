#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py - VRFT utility functions
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2021, by alessior@kth.se
#
# Copyright [2017-2021] [Alessio Russo - alessior@kth.se]  
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
import scipy.signal as scipysig


def Doperator(p: int, q: int, x: float) -> np.ndarray:
    """ DOperator, used to compute the overall Toeplitz matrix """
    D = np.zeros((p * q, q))
    for i in range(q):
        D[i * p:(i + 1) * p, i] = x
    return D

@overload
def check_system(tf: scipysig.dlti) -> bool:
    """Returns true if a transfer function is causal
    Parameters
    ----------
    tf : scipy.signal.dlti
        discrete time rational transfer function
    """

    return check_system(tf.num, tf.den)

def check_system(num: np.ndarray, den: np.ndarray) -> bool:
    """Returns true if a transfer function is causal
    Parameters
    ----------
    num : np.ndarray
        numerator of the transfer function
    den : np.ndarray
        denominator of the transfer function

    """
    try:
        M, N = system_order(num, den)
    except ValueError:
        raise

    if (N < M):
        raise ValueError("The system is not causal.")

    return True

@overload
def system_order(tf: scipysig.dlti) -> tuple:
    """Returns the order of the numerator and denominator
       of a transfer function
    Parameters
    ----------
    tf : scipy.signal.dlti
        discrete time rational transfer function

    Returns
    ----------
    (num, den): tuple
        Tuple containing the orders
    """
    return system_order(tf.num, tf.den)

def system_order(num: np.ndarray, den: np.ndarray) -> tuple:
    """Returns the order of the numerator and denominator
       of a transfer function
    Parameters
    ----------
    num : np.ndarray
        numerator of the transfer function
    den : np.ndarray
        denominator of the transfer function

    Returns
    ----------
    (num, den): tuple
        Tuple containing the orders
    """
    den = den if isinstance(den, np.ndarray) else np.array([den]).flatten()
    num = num if isinstance(num, np.ndarray) else np.array([num]).flatten()

    if num.ndim == 0:
        num = np.expand_dims(num, axis=0)

    if den.ndim == 0:
        den = np.expand_dims(den, axis=0)

    return (np.poly1d(num).order, np.poly1d(den).order)

def filter_signal(L: scipysig.dlti, x: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
    """Filter data in an iddata object
    Parameters
    ----------
    L : scipy.signal.dlti
        Discrete-time rational transfer function used to
        filter the signal
    x : np.ndarray
        Signal to filter
    x0 : np.ndarray, optional
        Initial conditions for L
    Returns
    -------
    signal : iddata
        Filtered iddata object
    """
    t_start = 0
    t_step = L.dt
    t_end = x.size * t_step

    t = np.arange(t_start, t_end, t_step)
    _, y = scipysig.dlsim(L, x, t, x0)
    return y.flatten()

def deconvolve_signal(L: scipysig.dlti, x: np.ndarray) -> np.ndarray:
    """Deconvolve a signal x using a specified transfer function L(z)
    Parameters
    ----------
    L : scipy.signal.dlti
        Discrete-time rational transfer function used to
        deconvolve the signal
    x : np.ndarray
        Signal to deconvolve

    Returns
    -------
    signal : np.ndarray
        Deconvolved signal
    """
    dt = L.dt
    impulse = scipysig.dimpulse(L)[1][0].flatten()
    idx1 = np.argwhere(impulse != 0)[0].item()
    idx2 = np.argwhere(np.isclose(impulse[idx1:], 0.) == True)
    idx2 = -1 if idx2.size == 0 else idx2[0].item()
    signal, _ = scipysig.deconvolve(x, impulse[idx1:idx2])
    return signal[np.argwhere(impulse != 0)[0].item():]

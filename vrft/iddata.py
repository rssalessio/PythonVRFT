#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# iddata.py - iddata object definition
# Analogous to the iddata object in Matlab sysid
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 09th January 2021, by alessior@kth.se
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

import numpy as np
import scipy.signal as scipysig
from vrft.utils import filter_signal
import copy


class iddata(object):
    """
     iddata is a class analogous to the iddata object in Matlab
     It is used to save input/output data.

     @NOTE: y0, the initial conditions, are in general not used.
            The only reason to specify y0 is in case the system is non linear.
            In that case y0 needs to be specified (for the equilibria condition)
    """

    def __init__(self, y: np.ndarray,
                 u: np.ndarray,
                 ts: float,
                 y0: np.ndarray = None):
        """
        Input/output data (suppors SISO systems only)
        Parameters
        ----------
        y: np.ndarray
            Output data
        u: np.ndarray
            Input data
        ts: float
            sampling time
        y0: np.ndarray, optional
            Initial conditions
        """
        if y is None:
            raise ValueError("Signal y can't be None.")
        if u is None:
            raise ValueError("Signal u can't be None.")
        if ts is None:
            raise ValueError("Sampling time ts can't be None.")

        self.y = np.array(y, copy=True) if not isinstance(y, np.ndarray) else np.array([y], copy=True).flatten()
        self.u = np.array(u, copy=True) if not isinstance(u, np.ndarray) else np.array([u], copy=True).flatten()
        self.ts = float(ts)

        if y0 is None:
            raise ValueError("y0: {} can't be None.".format(y0))
        else:
            self.y0 = np.array(y0, copy=True) if not isinstance(y0, np.ndarray) else np.array([y0], copy=True).flatten()
            if self.y0.size == 0 or self.y0.ndim == 0:
                raise ValueError("y0 can't be None.")


    def check(self):
        """ Checks validity of the data """
        if (self.y.shape != self.u.shape):
            raise ValueError("Input and output size do not match.")

        if (np.isclose(self.ts, 0.0) == True):
            raise ValueError("Sampling time can not be zero.")

        if (self.ts < 0.0):
            raise ValueError("Sampling time can not be negative.")

        if (self.y0 is None):
            raise ValueError("Initial condition can't be zero")

        return True

    def copy(self):
        """ Returns a copy of the object """
        return iddata(self.y, self.u, self.ts, self.y0)

    def filter(self, L: scipysig.dlti):
        """ Filters the data using the specified filter L(z) """
        self.y = filter_signal(L, self.y)
        self.u = filter_signal(L, self.u)
        return self

    def split(self) -> tuple:
        """ Splits the dataset into two equal parts
            Used for the instrumental variable method
        """
        n0 = self.y0.size if self.y0 is not None else 0
        n = self.y.size

        if (n + n0) % 2 != 0:
            print('iddata object has uneven data size. The last data point will be discarded')
            n -= 1

        # First dataset
        n1 = (n + n0) // 2 # floor division
        d1 = iddata(self.y[:n1 - n0], self.u[:n1 - n0], self.ts, self.y0)

        # Second dataset
        d2 = iddata(self.y[n1:n], self.u[n1:n], self.ts, self.y[n1 - n0:n1])

        return (d1, d2)



# extended_tf.py - Extended definition of the discrete
# transfer function implemented in scipy.signal.
# Supports arithmetical operations between transfer function
# and feedback loop computation
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2020, by alessior@kth.se
#
# Copyright [2020] [Alessio Russo - alessior@kth.se]  
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

from __future__ import division

import numpy as np
import scipy.signal as scipysig
from scipy.signal.ltisys import TransferFunction as TransFun
from numpy import polymul, polyadd


class ExtendedTF(scipysig.ltisys.TransferFunctionDiscrete):
    """
    Extended definition of the discrete transfer function implemented in scipy.signal.
    Supports arithmetical operations between transfer function and feedback loop
    computation
    """

    def __init__(self, num: np.ndarray, den: np.ndarray, dt: float):
        self._dt = dt
        super().__init__(num, den, dt=dt)

    def __neg__(self):
        return ExtendedTF(-self.num, self.den, dt=self._dt)

    def __floordiv__(self, other):
        # can't make sense of integer division right now
        return NotImplemented

    def __mul__(self, other):
        if type(other) in [int, float]:
            return ExtendedTF(self.num*other, self.den, dt=self._dt)
        elif type(other) in [TransFun, ExtendedTF]:
            numer = polymul(self.num, other.num)
            denom = polymul(self.den, other.den)
            return ExtendedTF(numer, denom, dt=self._dt)

    def __truediv__(self, other):
        if type(other) in [int, float]:
            return ExtendedTF(self.num,self.den*other, dt=self._dt)
        if type(other) in [TransFun, ExtendedTF]:
            numer = polymul(self.num, other.den)
            denom = polymul(self.den, other.num)
            return ExtendedTF(numer, denom, dt=self._dt)

    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            return ExtendedTF(other*self.den, self.num, dt=self._dt)
        if type(other) in [TransFun, ExtendedTF]:
            numer = polymul(self.den, other.num)
            denom = polymul(self.num, other.den)
            return ExtendedTF(numer, denom, dt=self._dt)

    def __add__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(polyadd(self.num, self.den*other), self.den, dt=self._dt)
        if type(other) in [TransFun, type(self)]:
            if len(self.den) == len(other.den) and np.all(self.den == other.den):
                numer = polyadd(self.num, other.num)
                denom = self.den
            else:
                numer = polyadd(polymul(self.num, other.den), polymul(self.den, other.num))
                denom = polymul(self.den, other.den)
            return ExtendedTF(numer, denom, dt=self._dt)

    def __sub__(self, other):
        if type(other) in [int, float]:
            return ExtendedTF(polyadd(self.num, -self.den*other), self.den, dt=self._dt)
        if type(other) in [TransFun, type(self)]:
            if len(self.den) == len(other.den) and np.all(self.den == other.den):
                numer = polyadd(self.num, -other.num)
                denom = self.den
            else:
                numer = polyadd(polymul(self.num, other.den), -polymul(self.den, other.num))
                denom = polymul(self.den, other.den)
            return ExtendedTF(numer, denom, dt=self._dt)

    def __rsub__(self, other):
        if type(other) in [int, float]:
            return ExtendedTF(polyadd(-self.num, self.den*other), self.den, dt=self._dt)
        if type(other) in [TransFun, type(self)]:
            if len(self.den) == len(other.den) and np.all(self.den == other.den):
                numer = polyadd(self.num, -other.num)
                denom = self.den
            else:
                numer = polyadd(polymul(self.num, other.den), -polymul(self.den, other.num))
                denom = polymul(self.den, other.den)
            return ExtendedTF(numer, denom, dt=self._dt)

    def feedback(self):
        """ Computes T(z)/(1+T(z)) """
        num = self.num
        den = self.den
        den = polyadd(num, den)
        self = ExtendedTF(num, den, dt=self.dt)
        return self

    __rmul__ = __mul__
    __radd__ = __add__



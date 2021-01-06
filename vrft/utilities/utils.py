from __future__ import division

import numpy as np
import scipy.signal as scipysig
from .iddata import iddata

from scipy.signal.ltisys import TransferFunction as TransFun
from numpy import polymul, polyadd

def Doperator(p: int, q: int, x: float) -> np.ndarray:
    D = np.zeros((p * q, q))
    for i in range(q):
        D[i * p:(i + 1) * p, i] = x
    return D

def checkSystem(num: np.ndarray, den: np.ndarray) -> bool:
    try:
        M, N = systemOrder(num, den)
    except ValueError:
        raise

    if (N < M):
        raise ValueError("The system is not causal.")

    return True

def systemOrder(num: np.ndarray, den: np.ndarray) -> tuple:
    den = den if isinstance(den, np.ndarray) else np.array([den]).flatten()
    num = num if isinstance(num, np.ndarray) else np.array([num]).flatten()

    if num.ndim == 0:
        num = np.expand_dims(num, axis=0)

    if den.ndim == 0:
        den = np.expand_dims(den, axis=0)

    N = den.size
    M = num.size

    denOrder = -1
    numOrder = -1

    for i, d in enumerate(den):
        if not np.isclose(d, 0):
            denOrder = N - i - 1
            break

    for i, n in enumerate(num):
        if not np.isclose(n, 0):
            numOrder = M - i - 1
            break

    if (denOrder == -1):
        raise ValueError("Denominator can not be zero.")

    if (numOrder == -1):
        raise ValueError("Numerator can not be zero.")

    return (numOrder, denOrder)


def filter_iddata(data: iddata, L: scipysig.dlti) -> iddata:
    t_start = 0
    t_step = data.ts
    t_end = len(data.y) * t_step

    t = np.arange(t_start, t_end, t_step)

    if (L != None):
        t, y = scipysig.dlsim(L, data.y, t)
        t, u = scipysig.dlsim(L, data.u, t)
        return iddata(y[:, 0], u[:, 0], data.ts, data.y0)
    return data

def deconvolve_signal(T: scipysig.dlti, x: np.ndarray, dt: float) -> np.ndarray:
    impulse = scipysig.dimpulse(T)[1][0].flatten()
    idx1 = np.argwhere(impulse != 0)[0].item()
    idx2 = np.argwhere(np.isclose(impulse[idx1:], 0.) == True)
    idx2 = -1 if not idx2 else idx2[0].item()
    recovered, _ = scipysig.deconvolve(x, impulse[idx1:idx2])
    return recovered[np.argwhere(impulse != 0)[0].item():]

class ExtendedTF(scipysig.ltisys.TransferFunctionDiscrete):
    def __init__(self, num, den, dt):
        self._dt = dt
        super().__init__(num, den, dt=dt)

    def __neg__(self):
        return ExtendedTF(-self.num, self.den, dt=self._dt)

    def __floordiv__(self,other):
        # can't make sense of integer division right now
        return NotImplemented

    def __mul__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(self.num*other, self.den, dt=self._dt)
        elif type(other) in [TransFun, ExtendedTF]:
            numer = polymul(self.num,other.num)
            denom = polymul(self.den,other.den)
            return ExtendedTF(numer,denom, dt=self._dt)

    def __truediv__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(self.num,self.den*other, dt=self._dt)
        if type(other) in [TransFun, ExtendedTF]:
            numer = polymul(self.num,other.den)
            denom = polymul(self.den,other.num)
            return ExtendedTF(numer,denom, dt=self._dt)

    def __rtruediv__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(other*self.den,self.num, dt=self._dt)
        if type(other) in [TransFun, ExtendedTF]:
            numer = polymul(self.den,other.num)
            denom = polymul(self.num,other.den)
            return ExtendedTF(numer,denom, dt=self._dt)

    def __add__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(polyadd(self.num,self.den*other), self.den, dt=self._dt)
        if type(other) in [TransFun, type(self)]:
            if np.all(self.den == other.den):
                numer = polyadd(self.num, other.num)
                denom = self.den
            else:
                numer = polyadd(polymul(self.num,other.den),polymul(self.den,other.num))
                denom = polymul(self.den,other.den)
            return ExtendedTF(numer,denom, dt=self._dt)

    def __sub__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(polyadd(self.num,-self.den*other),self.den, dt=self._dt)
        if type(other) in [TransFun, type(self)]:
            if np.all(self.den == other.den):
                numer = polyadd(self.num, -other.num)
                denom = self.den
            else:
                numer = polyadd(polymul(self.num,other.den),-polymul(self.den,other.num))
                denom = polymul(self.den,other.den)
            return ExtendedTF(numer,denom, dt=self._dt)

    def __rsub__(self,other):
        if type(other) in [int, float]:
            return ExtendedTF(polyadd(-self.num,self.den*other),self.den, dt=self._dt)
        if type(other) in [TransFun, type(self)]:
            if np.all(self.den == other.den):
                numer = polyadd(self.num, -other.num)
                denom = self.den
            else:
                numer = polyadd(polymul(self.num,other.den),-polymul(self.den,other.num))
                denom = polymul(self.den,other.den)
            return ExtendedTF(numer,denom, dt=self._dt)

    # sheer laziness: symmetric behaviour for commutative operators
    __rmul__ = __mul__
    __radd__ = __add__

def feedback(L):
    num = L.num
    den = L.den
    den = polyadd(num, den)
    return ExtendedTF(num, den, dt=L.dt)
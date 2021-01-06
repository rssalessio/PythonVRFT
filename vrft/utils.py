import numpy as np
import scipy.signal as scipysig
from .iddata import iddata

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
    idx2 = -1 if idx2.size == 0 else idx2[0].item()
    recovered, _ = scipysig.deconvolve(x, impulse[idx1:idx2])
    return recovered[np.argwhere(impulse != 0)[0].item():]

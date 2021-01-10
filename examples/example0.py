import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipysig
from vrft import *

dt = 0.05
t_start = 0
t_end = 10
t = np.array([i * dt for i in range(int(t_end/dt))])

# Plant P(z) 
num_P = [0.28261, 0.50666]
den_P = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = ExtendedTF(num_P, den_P, dt=dt)

def generate_data(sys, u, t):
    t, y = scipysig.dlsim(sys, u, t)
    y = y.flatten() + 0.5 * np.random.normal(size = t.size)
    return iddata(y, u, dt, [0, 0, 0])

u = np.random.normal(size=t.size)
data1 = generate_data(sys, u, t)

# Reference Model
#            z^-3 (1-alpha)^2
# M(z) =   ---------------------
#           (1 - alpha z^-1)^2 
# 
#       with alpha = e^{-dt omega}, omega = 10
#
omega = 10
alpha = np.exp(-dt*omega)
num_M = [(1-alpha)**2] 
den_M = [1, -2*alpha, alpha**2, 0]
refModel = ExtendedTF(num_M, den_M, dt=dt)

# Controller C(z,O) where O is $\theta$
#
#             O_0 z^5 + O_1 z^4 + O_2 z^3 + O_3 z^2 + O_4 z^1 + O_5 
# C(z,O) =   -------------------------------------------------------
#                                  z^5 - z^4
#
control = [ExtendedTF([1, 0], [1, -1], dt=dt),
           ExtendedTF([1], [1, -1], dt=dt),
           ExtendedTF([1], [1, -1, 0], dt=dt),
           ExtendedTF([1], [1, -1, 0, 0], dt=dt),
           ExtendedTF([1], [1, -1, 0, 0, 0], dt=dt),
           ExtendedTF([1], [1, -1, 0, 0, 0, 0], dt=dt)]

# Experiment filter
#
# L(z) = M(z) ( 1 - M(z) )
#
prefilter = refModel * (1 - refModel)

def my_convolve(sequence: np.ndarray, M: ExtendedTF) -> np.ndarray:
    num = M.num 
    den = M.den 
    zi = scipysig.lfilter_zi(num, den)
    z, _ = scipysig.lfilter(num, den, sequence, zi=zi*sequence[0])
    return z

def my_deconvolve(sequence: np.ndarray, M: ExtendedTF) -> np.ndarray:
    num = M.num 
    den = M.den 
    zi = scipysig.lfilter_zi(den, num)
    z, _ = scipysig.lfilter(den, num, sequence, zi=zi*sequence[0])
    return z

y_output = np.ones(200)
y_output[0:20] = np.zeros(20)

r_bar = my_deconvolve(y_output, refModel)
y_reconstruct = my_convolve(r_bar, refModel)
refModel_output = my_convolve(y_output, refModel)

plt.figure(1)
plt.plot(r_bar, label='r_bar')
plt.plot(y_output, label='y')
plt.legend()

plt.figure(2)
plt.plot(y_output, label='y_output')
plt.plot(y_reconstruct, label='y_reconstruct')
plt.legend()

plt.figure(3)
plt.plot(y_output, label='y_output')
plt.plot(refModel_output, label='refModel_output')
plt.legend()
plt.show()


# r_bar_2 = virtual_reference(data1, prefilter)

print("done")
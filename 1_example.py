import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from vrft import *

t_start = 0
t_end = 10
t_step = 1e-2
t = np.arange(t_start, t_end, t_step)
u = np.ones(len(t)).tolist()

num = [0.5]
den = [1, -1]
sys = ctl.tf(num, den, t_step)
y,t,x= ctl.lsim(sys, u, t)
data = vrft.iddata(y[0],u,t_step,[0])

refModel = ctl.tf([0.2], [1, -0.8], t_step)

#PI Controller
base = [ctl.tf([1], [1],t_step),
		ctl.tf([1, 0], [1, -1],t_step)]

C, theta = vrft.vrftAlgorithm(data, refModel, base)

print "Controller:", C
L = C*sys
L = L/(1+L)

yr,t = ctl.step(refModel, t)
yc,t = ctl.step(L, t)

plt.plot(t,yr[0],label='Ref System')
plt.plot(t,yc[0], label='CL Sustem')
plt.show()

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
den = [1, -0.9]
sys = ctl.tf(num, den, t_step)
y,t,x= ctl.lsim(sys, u, t)

om = 2*np.pi*100.7
xi = 0.9
noise =  np.random.normal(0,0.1,y.size)
yn,t,x = ctl.lsim(ctl.tf([10*om**2], [1, 2*xi*om, om**2]), noise, t)
y += yn
print y.size
data = vrft.iddata(y[0],u,t_step,[0])

refModel = ctl.tf([0.6], [1, -0.4], t_step)

#PI Controller
base = [ctl.tf([1], [1],t_step),
		ctl.tf([1, 0], [1, -1],t_step)]

omega = 2*np.pi*0.1
L =  ctl.tf([1], [1/omega, 1])
L =  ctl.sample_system(L, 0.01)

C, theta = vrft.vrftAlgorithm(data, refModel, base, L)

print "Controller:", C
L = C*sys
L = L/(1+L)

yr,t = ctl.step(refModel, t)
yc,t = ctl.step(L, t)


fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(t,yr[0],label='Ref System')
ax[0].plot(t,yc[0], label='CL System')
ax[0].set_title('Systems response')
ax[1].plot(t,y[0], label='Y Data')
ax[1].set_title('Experiment data')

# Now add the legend with some customizations.
legend = ax[0].legend(loc='lower right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()

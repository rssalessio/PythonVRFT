import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from vrft import *

def generateNoise(t):
	omega = 2*np.pi*100
	xi = 0.9
	noise =  np.random.normal(0,0.1,t.size)
	# Second order system
	yn,t,x = ctl.lsim(ctl.tf([omega**2], [1, 2*xi*omega, omega**2]), \
		noise, t)
	return yn

#Generate time and u(t) signals
t_start = 0
t_end = 10
t_step = 1e-2
t = np.arange(t_start, t_end, t_step)
u = np.ones(len(t)).tolist()
u[200:400] = np.zeros(200)
u[600:800] = np.zeros(200)

#Experiment
num = [0.5]
den = [1, -1.6, 0.8]
sys = ctl.tf(num, den, t_step)
y,t,x= ctl.lsim(sys, u, t)
y += generateNoise(t)
data = vrft.iddata(y[0],u,t_step,[0])

#Reference Model
refModel = ctl.tf([0.2], [1, -0.8], t_step)

#PI Controller
base = [ctl.tf([1], [1],t_step),
	ctl.tf([1, 0], [1, -1],t_step),
	ctl.tf([1, -1], [1, 0],t_step)]

#Experiment filter
omega = 2*np.pi*1
L =  ctl.tf([1], [1/omega, 1])
L =  ctl.sample_system(L, 0.01)

#VRFT
C, theta, r = vrft.vrftAlgorithm(data, refModel, base, L)

#Obtained controller
print "Controller:", C
L = C*sys
L = L/(1+L)

L = L.minreal()
#this is due to a bug in the control library
L = ctl.tf(L.num, L.den, t_step)

print "Theta:", theta

#Analysis
yr,t = ctl.step(refModel, t)
yc,t = ctl.step(L, t)
ys,t = ctl.step(sys, t)


fig, ax = plt.subplots(4, sharex=True)
ax[0].plot(t,yr[0],label='Ref System')
ax[0].plot(t,yc[0], label='CL System')
ax[0].set_title('Systems response')
ax[0].grid(True)
ax[1].plot(t,ys[0], label='OL System')
ax[1].set_title('OL Systems response')
ax[1].grid(True)
ax[2].plot(t,y[0])
ax[2].grid(True)
ax[2].set_title('Experiment data')
ax[3].plot(t,r)
ax[3].grid(True)
ax[3].set_title('Virtual Reference')

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

import control as ctl 
import numpy as np 
import matplotlib.pyplot as plt

def calculateReferenceSignal(t, y,referenceModel):
  # unpack the state vector
  num = referenceModel.num[0][0]
  den =  referenceModel.den[0][0]
  r = 0*t
  



  y = np.fliplr([y])[0]
  
  r,t,x = ctl.lsim(referenceModel, y, t)
  x = state[0]
  xd = state[1]

  # these are our constants
  k = -2.5 # Newtons per metre
  m = 1.5 # Kilograms
  g = 9.8 # metres per second

  # compute acceleration xdd
  xdd = ((k*x)/m) + g

  # return the two state derivatives
  return [xd, xdd]

x0 = [0.0, 0.0]
t = np.arange(0.0, 10, 0.01)
A = [ [-0.8060, 1.0], [-9.1486, -4.59]]
B = [[-0.04],[-4.59]]
C = [1, 0]
D = [0]
Ktheta = [7.5*-9.1486, 2.5*-4.59]

#A = A + np.multiply(B,Ktheta)

sys = ctl.ss(A,B,C,D)
u = t*0+1
y,t,x = ctl.lsim(sys, u, t)

y =  y + np.random.normal(0,1e-2, len(y))

plt.subplot(2,1,1)
plt.plot(t, x)
plt.xlabel('TIME (sec)')
plt.ylabel('x')
plt.title('States x')
plt.xlim(t[0], t[-1])
plt.ylim(-1,0.3)
plt.grid(1)

plt.subplot(2,1,2)
plt.plot(t, y)
plt.xlabel('TIME (sec)')
plt.ylabel('y')
plt.title('Measurement y')
plt.xlim(t[0], t[-1])
plt.ylim(-1,0.3)
plt.grid(1)
plt.show()


refModel =  ctl.tf([1], [1, -0.8], 0.01)
calculateReferenceSignal(t,y, refModel)
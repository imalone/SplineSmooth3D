from scipy.interpolate import splev
import numpy.linalg as linalg
import numpy as np

points=100
Nphi=4
#xi = np.arange(0,1,0.1)
# xk=xi # Previously used Nphi = xi.shape-4 for exact point fitting

xi= np.linspace(0,1,points, endpoint=True)

xk = np.linspace(0,1,Nphi, endpoint=True)
xk = np.concatenate(([xk[0]]*2,xk,[xk[-1]]*2))


f = np.sin(xi * np.pi)

B = np.zeros((xi.shape[0], Nphi))
for p in range(0,Nphi):
   xp = np.zeros(Nphi)
   xp[p]=1
   B[:,p] = splev(xi,(xk,xp,3))
orth=np.matmul(B.transpose(),B)
proj=np.matmul(linalg.inv(orth),B.transpose())
phi = np.matmul(proj, f)

testf = splev(xi,(xk,phi,3))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(xi,testf,"lightblue")
plt.plot(xi,f,"orangered")
plt.show(block=False)

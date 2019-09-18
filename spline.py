from scipy.interpolate import splev
import numpy.linalg as linalg
import numpy as np


#xi = np.arange(0,1,0.1)
xi= np.linspace(0,1,10)
f = np.sin(xi * np.pi)


Nphi = xi.shape[0] - 4
B = np.zeros((xi.shape[0], Nphi))
xk=xi
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


M=[4,5,6]

d=[1,1,1]
order=3
nk = [ x + (order+1) for x in M ]

k = [ np.linspace(0,thisd,thisk) for thisd, thisk in zip(d,nk) ]

N=[20,20,20]
loc=[ np.linspace(0,thisd,thisN) for thisd, thisN in zip(d,N) ]

def BforData1D(knots,datapoints, order=3):
  Nphi = len(knots) - (order+1)
  B = np.zeros((len(datapoints), Nphi))
  for p in range(0,Nphi):
    xp = np.zeros(Nphi)
    xp[p] = 1
    B[:,p] = splev(datapoints, (knots, xp, order))
  return B

def orderedProduct(X,Y,Z):
  prod = [ z * y * x for z in Z for y in Y for x in X ]
  prod = np.array(prod)
  return prod
  

B3d = [ BforData1D(k[dim],loc[dim]) for dim in range(0,3) ]

Bfull = np.zeros((np.prod(N), np.prod(M)))
for Z in range(0,N[2]) :
  BZ = B3d[2][Z,]
  for Y in range(0,N[1]) :
    BY = B3d[1][Y,]
    for X in range(0,N[0]) :
      BX = B3d[0][X,]
      row=X + Y * N[0] + Z * N[0] * N[1]
      Bfull[row,] = orderedProduct(BX,BY,BZ)

b1=np.matmul(Bfull.transpose(),Bfull)
b2 = linalg.inv(b1)
b3 = np.matmul(b2, Bfull.transpose())

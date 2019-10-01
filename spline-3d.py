from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time
import sys
import argparse

from scipy.interpolate import splev
import numpy.linalg as linalg
import numpy as np
import nibabel as nib

# Not used yet, but keep in case
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

infile='adni3-1006.nii.gz'
inimg = nib.load(infile)
inimgdata = inimg.get_fdata()


M=[4,5,6]

d=[1,1,1]
order=3
nk = [ x + (order+1) for x in M ]

k = [ np.linspace(0,thisd,thisk) for thisd, thisk in zip(d,nk) ]

N=inimgdata.shape
loc=[ np.linspace(0,thisd,thisN) for thisd, thisN in zip(d,N) ]

def BforData1D(knots,datapoints, order=3):
  Nphi = len(knots) - (order+1)
  B = np.zeros((len(datapoints), Nphi))
  for p in range(0,Nphi):
    xp = np.zeros(Nphi)
    xp[p] = 1
    B[:,p] = splev(datapoints, (knots, xp, order))
  return B

# Fastest index is last in numpy (the order you get with .flat),
# so stick with this convention, but N.B. it doesn't match the
# on-disc storage order for the image!

def orderedProduct_model(Z,Y,X):
  prod = [ z * y * x for z in Z for y in Y for x in X ]
  prod = np.array(prod)
  return prod
  

def orderedProduct_asOuter(Z,Y,X):
  return np.outer(np.outer(Z,Y),X)

orderedProduct = orderedProduct_asOuter


B3d = [ BforData1D(k[dim],loc[dim]) for dim in range(0,3) ]

Bfull = np.zeros((np.prod(N), np.prod(M)))

t_start=time.time()
print(t_start)
for Z in range(0,N[0]) :
  BZ = B3d[0][Z,]
  for Y in range(0,N[1]) :
    BY = B3d[1][Y,]
    for X in range(0,N[2]) :
      BX = B3d[2][X,]
      row=X + Y * N[2] + Z * N[2] * N[1]
      Bfull[row,] = orderedProduct(BZ,BY,BX).flat
t_end=time.time()
print(t_end)
print(t_end-t_start)


# P = (A' A + w J)^-1 * A' * Z
# A is big, as is (A' * A + w J)^-1 * A'
# so instead of calculation in order
# do (A' * A + w J)^-1 and A' * Z
# then multiply
# If A is too large to store, can evaluate
# A' * A once and then A' * Z using B3d,
# but this is slower than using a
# pre-calculated A as it requires the outer
# products each time rather than the single
# matrix multiplication

b1=np.matmul(Bfull.transpose(),Bfull)

b2 = linalg.inv(b1) # Not implemented J yet

b4 = np.matmul(Bfull.transpose(),inimgdata.flat)

phi = np.matmul(b2,b4)
resdata = Bfull.dot(phi)
resdata = resdata.reshape(inimgdata.shape)
imgresnii = nib.Nifti1Image(resdata, inimg.affine, inimg.header)
nib.save(imgresnii,"test-spline3d.nii.gz") 

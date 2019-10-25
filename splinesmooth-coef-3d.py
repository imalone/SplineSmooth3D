from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time
import sys
import argparse

import numpy.linalg as linalg
import numpy as np
import nibabel as nib

def orderedProduct_asOuter(Z,Y,X):
  return np.outer(np.outer(Z,Y),X)

infile="adni3-1006.nii.gz"
outfile="test-splinesmooth3d.nii.gz"
testfile="test-model.nii.gz"
realdata=False
if realdata:
    inimg = nib.load(infile)
    inimgdata = inimg.get_fdata()
else:
    testshape=(50,100,150)
    #testshape=(50,100,1)
    inimgdata=np.zeros(testshape)
    for Z in range(0,testshape[0]) :
      print (Z)
      dz2 = (Z-testshape[0]/2.0)**2
      for Y in range(0,testshape[1]) :
        dy2 = (Y-testshape[1]/2.0)**2
        for X in range(0,testshape[2]) :
          dx2 = (X-testshape[2]/2.0)**2
          inimgdata[Z][Y][X] = dx2 + dy2 + dz2
          #print("{} {} {} : {} {} {} : {}".format(X,Y,Z,dx2,dy2,dz2,inimgdata[Z][Y][X]))
    aff = np.diag([1]*4)
    inimg = nib.nifti1.Nifti1Image(inimgdata,aff)
    nib.save(inimg,testfile) 


q = 3
spacing = 50

shape=inimg.shape
voxsizes=nib.affines.voxel_sizes(inimg.affine)

domain = [[0,(nvox-1)*voxsize]
          for (nvox, voxsize) in zip(shape, voxsizes)]

kntsArr = [knots_over_domain(bound[0],bound[1],spacing,q=q)
           for bound in domain]

totalPar = 1
for knts in kntsArr:
    totalPar *= knts[0]

# Form the tensor bases
coefArr = [
    [ eval_nonzero_bspl(knts[1],voxsize*v,q=q) for v in range(nvox) ]
  for (knts, nvox, voxsize) in zip (kntsArr, shape, voxsizes)]


AtA = np.zeros((totalPar,totalPar))
Atx = np.zeros(totalPar)
# N.B. shallow copy for flat indexing into AtA
AtAflat = AtA.reshape((AtA.shape[0]*AtA.shape[1]))

# Direct A works.
A=np.zeros((np.prod(inimg.shape),totalPar))

t_start=time.time()
t_last=t_start
print(t_start)
q1=q+1
q13 = q1**3
#indsZpattern = np.tile(range(0,q1),q1*q1) 
#indsYpattern = np.tile(np.repeat(range(0,q1),q1),q1) * kntsArr[0][0]
#indsXpattern = np.repeat(range(0,q1),q1*q1) * kntsArr[0][0] * kntsArr[1][0]

# X is fastest changing: LAST index
indsXpattern = np.tile(range(0,q1),q1*q1) 
indsYpattern = np.tile(np.repeat(range(0,q1),q1),q1) * kntsArr[2][0]
indsZpattern = np.repeat(range(0,q1),q1*q1) * kntsArr[2][0] * kntsArr[1][0]


## 2d
#indsZpattern = np.tile(range(0,q1),q1) 
#indsYpattern = np.repeat(range(0,q1),q1) * kntsArr[0][0]


needAtA=True
timeSteps=True
for Z in range(0,shape[0]) :
  print (Z)
  (cIndZ, cZ) = coefArr[0][Z]
  for Y in range(0,shape[1]) :
    (cIndY, cY) = coefArr[1][Y]
    for X in range(0,shape[2]) :
      (cIndX, cX) = coefArr[2][X]
      # 5us
      # Times include ~ 5us overhead for loop and print.

      # striping the BX, BY, BZ tensor into the outer-product order.
      # Doesn't matter what choice we make so long as consistent.
      # Going to end up with (q+1) * (q+1) * (q+1) coefficients,
      # corresponding to the support cube, they then need to go in
      # at intervals, q+1 long runs.

      # Using pre-assigned out=array gives only about 1us improvement.
      # pre-assigning arrays and using (a,b,out=preassigned)
      ## orderedProduct(Z,Y,X), x changes fastest in resulting list
      coefs = orderedProduct_asOuter(cZ,cY,cX)
      coefsx = np.multiply(coefs, inimgdata[Z,Y,X])
      # 15us

      #indsZ = indsZpattern + cIndZ
      #indsY = indsYpattern + cIndY * kntsArr[0][0]
      #indsX = indsXpattern + cIndX * kntsArr[0][0] * kntsArr[1][0]
      indsX = indsXpattern + cIndX
      indsY = indsYpattern + cIndY * kntsArr[2][0]
      indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]

      tgtinds = indsZ + indsY + indsX
      # 18us
      
      Atx[tgtinds] += coefsx.reshape(q13)
      # 20us
      A[X+shape[2]*Y+shape[2]*shape[1]*Z,tgtinds]=coefs.reshape(q13)
      if needAtA:
        ## Pre-assigning array makes little difference
        AtAadd = np.outer(coefs,coefs)
        # 25-26us

        ## Different AtA adding strategies to try to improve speed,
        ## ix_ indexed adding step takes about 42us.
        #AtA[np.ix_(tgtinds,tgtinds)] += AtAadd
        ## Linear indexed adding for AtA whole iteration takes about
        ## 20us, 9us for flatind, 11us for addition. Using .reshape()
        ## around 4us faster than using .flat()

        ## Pre-assigning seemingly slower?
        flatinds = tgtinds.reshape((q13,1)) + \
          totalPar * tgtinds.reshape((1,q13))
        # 32-33 us

        AtAflat[flatinds.reshape(q13**2)] += AtAadd.reshape((q13**2))
        # 45us
      if timeSteps:
        t_now = time.time()
        print(t_now-t_last)
        t_last=t_now
t_end=time.time()
print(t_end)
print(t_end-t_start)

# A hack, but just to get something invertible. Should be invertible
# as full coverage, but not, indicates an earlier mistake.
# Possibly was an error in setup of indsXpattern.
cheatsmoothing = False
if cheatsmoothing:
    J=np.diag([0.01]*AtA.shape[0])
else:
    J=np.zeros(AtA.shape)

AtAinv = np.linalg.inv(AtA+J)
P = np.matmul(AtAinv,Atx)

## Now fully supported, but clearly wrong when reconstructing.
## Definitely some indexing mix-up somewhere.

## Need to calc AP now...
pred = np.zeros(inimgdata.shape)

for Z in range(0,shape[0]) :
  print (Z)
  (cIndZ, cZ) = coefArr[0][Z]
  for Y in range(0,shape[1]) :
    (cIndY, cY) = coefArr[1][Y]
    for X in range(0,shape[2]) :
      (cIndX, cX) = coefArr[2][X]
      # striping the BX, BY, BZ tensor into
      # the outer-product order. Doesn't matter
      # what choice we make so long as consistent
      # going to end up with (q+1) * (q+1) * (q+1)
      # coefficients, corresponding to the support
      # cube, they then need to go in at intervals,
      # q+1 long runs.
      coefs = orderedProduct_asOuter(cZ,cY,cX).reshape(q13)
      #indsZ = indsZpattern + cIndZ
      #indsY = indsYpattern + cIndY * kntsArr[0][0]
      #indsX = indsXpattern + cIndX * kntsArr[0][0] * kntsArr[1][0]
      indsX = indsXpattern + cIndX
      indsY = indsYpattern + cIndY * kntsArr[2][0]
      indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      pred[Z,Y,X] = np.inner(P[tgtinds],coefs)

imgresnii = nib.Nifti1Image(pred, inimg.affine, inimg.header)
nib.save(imgresnii,outfile) 

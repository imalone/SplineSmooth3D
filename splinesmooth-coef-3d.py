#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time
import sys
import argparse

import numpy.linalg as linalg
import numpy as np
import nibabel as nib

from test_scipyspline import knots_over_domain, eval_nonzero_bspl

def orderedProduct_asOuter(Z,Y,X):
  return np.outer(np.outer(Z,Y),X)

infile="adni3-1006.nii.gz"
outfile="test-splinesmooth3d.nii.gz"
testfile="test-model.nii.gz"
realData=True
# Storing the whole A matrix is potentially faster, but needs *lots* of
# memory, so only practical for small images and testing
useA=False
# Timing information during voxel addition loop.
reportTimeSteps=False

q = 3
spacing = 50

if realData:
    inimg = nib.load(infile)
    inimgdata = inimg.get_fdata()
else:
    testshape=(50,100,150)
    inimgdata=np.zeros(testshape)
    for Z in range(0,testshape[0]) :
      print ("Building test: slice {} of {}".format(Z,testshape[0]))
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


def invertCoefList (coefList):
  # return: list indexed by first supporting coef number
  # each element contains: first voxel index,
  #                        list of (q) coeffs for that vox
  # not necessarily a matrix due to possibility of unequal
  # number of voxels with intervals (particularly at ends)?
  cIndList = [x[0] for x in coefList]
  q1 = coefList[0][1].shape[0]
  coefValList = np.array([x[1] for x in coefList]).reshape((-1,q1))
  firstC, voxinds, coefinds = np.unique(cIndList,
                             return_index=True, return_inverse=True)
  Alocs =[
    [firstvox, coefValList[coefinds==C,:]]
    for (C, firstvox) in zip (firstC, voxinds)
  ]
  return Alocs


invCoefArr = [
  invertCoefList(coefList)
  for coefList in coefArr
]

AtA = np.zeros((totalPar,totalPar))
Atx = np.zeros(totalPar)
# N.B. shallow copy for flat indexing into AtA
AtAflat = AtA.reshape((AtA.shape[0]*AtA.shape[1]))

# Direct A works.
if useA:
  A=np.zeros((np.prod(inimg.shape),totalPar))

t_start=time.time()
t_last=t_start
print(t_start)
q1=q+1
q13 = q1**3

# X is fastest changing: LAST index
indsXpattern = np.tile(range(0,q1),q1*q1) 
indsYpattern = np.tile(np.repeat(range(0,q1),q1),q1) * kntsArr[2][0]
indsZpattern = np.repeat(range(0,q1),q1*q1) * kntsArr[2][0] * kntsArr[1][0]

needAtA=True
slow=False

for cIndZ in range(len(invCoefArr[0])):
  if slow: break
  firstZ, coefsZ = invCoefArr[0][cIndZ]
  nindZ=coefsZ.shape[0]
  for cIndY in range(len(invCoefArr[1])):
    firstY, coefsY = invCoefArr[1][cIndY]
    nindY=coefsY.shape[0]
    for cIndX in range(len(invCoefArr[2])):
      firstX, coefsX = invCoefArr[2][cIndX]
      nindX=coefsX.shape[0]
      print("{} {} {}".format(cIndZ,cIndY,cIndX))

      localData = inimgdata[firstZ:firstZ+nindZ,
                            firstY:firstY+nindY,
                            firstX:firstX+nindX]

      # This is what we're actually doing, below, and may help
      # explain what might seem an odd order of indices
      #localAtens = np.einsum("zi,yj,xk->zyxijk",
      #                       coefsZ,coefsY,coefsX)
      #localA = localAtens.reshape((-1,q13))
      #localAtA = np.matmul(localA.transpose(),localA)
      #localAtx = np.matmul(localA.transpose(),localData.reshape(-1))

      # .encode("ascii","ignore") is necessary to avoid a TypeError
      # due to  from __future__ import (unicode_literals)
      localAtA = np.einsum(
         "xc,yb,za,zi,yj,xk->abcijk".encode("ascii","ignore"),
         coefsX,coefsY,coefsZ,                                          
         coefsZ,coefsY,coefsX, optimize=True ).reshape((q13,q13))       
      localAtx = np.einsum(                                            
        "xc,yb,za,zyx->abc".encode("ascii","ignore"),
        coefsX,coefsY,coefsZ,                                          
        localData, optimize=True).reshape((-1))                        

      indsX = indsXpattern + cIndX
      indsY = indsYpattern + cIndY * kntsArr[2][0]
      indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      ## Need to re-write A building if we still want to have
      ## the option: needs indexing of vox locations into final A,
      ## alternatively store an array of the non-zero A for supported
      ## regions, which will only ever be q13 * image size
      ##if useA:
      ##  # Not included in times
      ##  A[X+shape[2]*Y+shape[2]*shape[1]*Z,tgtinds]=coefs.reshape(q13)
      
      Atx[tgtinds] += localAtx
      if needAtA:
        flatinds = tgtinds.reshape((q13,1)) + \
          totalPar * tgtinds.reshape((1,q13))
        AtAflat[flatinds.reshape(q13**2)] += localAtA.reshape((q13**2))
      if reportTimeSteps:
        t_now = time.time()
        print(t_now-t_last)
        t_last=t_now
t_end=time.time()
print(t_end)
print(t_end-t_start)
 


for Z in range(0,shape[0]) :
  if not slow: break
  print ("Fitting, slice {} of {}".format(Z,shape[0]))
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

      indsX = indsXpattern + cIndX
      indsY = indsYpattern + cIndY * kntsArr[2][0]
      indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      # 18us
      if useA:
        # Not included in times
        A[X+shape[2]*Y+shape[2]*shape[1]*Z,tgtinds]=coefs.reshape(q13)
      
      Atx[tgtinds] += coefsx.reshape(q13)
      # 20us
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
      if reportTimeSteps:
        t_now = time.time()
        print(t_now-t_last)
        t_last=t_now
t_end=time.time()
print(t_end)
print(t_end-t_start)

# A hack, but useful in testing to get something invertible.
# With data covering the full domain AtA by itself should be
# invertible, if it's not that indicates an earlier error.
# With incomplete coverage some J is required to regularise
# the uncovered areas.
cheatsmoothing = False
if cheatsmoothing:
    J=np.diag([0.01]*AtA.shape[0])
else:
    J=np.zeros(AtA.shape)

# Direct inverse:
#AtAinv = np.linalg.inv(AtA+J)
#P = np.matmul(AtAinv,Atx)

# Cholesky and solver (can also just use solver, but
# since we know AtA+J is symmetric.
L = np.linalg(AtA+J)
p1=np.solve(L, Atx)
P=np.solve(L.T.conj(),p1)

## Need to calc AP now...
pred = np.zeros(inimgdata.shape)

# Same process as earlier, but without AtA calculation.
for Z in range(0,shape[0]) :
  if not slow: break
  print ("Projection, slice {} of {}".format(Z,shape[0]))
  (cIndZ, cZ) = coefArr[0][Z]
  for Y in range(0,shape[1]) :
    (cIndY, cY) = coefArr[1][Y]
    for X in range(0,shape[2]) :
      (cIndX, cX) = coefArr[2][X]

      coefs = orderedProduct_asOuter(cZ,cY,cX).reshape(q13)

      indsX = indsXpattern + cIndX
      indsY = indsYpattern + cIndY * kntsArr[2][0]
      indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      pred[Z,Y,X] = np.inner(P[tgtinds],coefs)


for cIndZ in range(len(invCoefArr[0])):
  if slow: break
  firstZ, coefsZ = invCoefArr[0][cIndZ]
  nindZ=coefsZ.shape[0]
  for cIndY in range(len(invCoefArr[1])):
    firstY, coefsY = invCoefArr[1][cIndY]
    nindY=coefsY.shape[0]
    for cIndX in range(len(invCoefArr[2])):
      firstX, coefsX = invCoefArr[2][cIndX]
      nindX=coefsX.shape[0]
      print("{} {} {}".format(cIndZ,cIndY,cIndX))

      indsX = indsXpattern + cIndX
      indsY = indsYpattern + cIndY * kntsArr[2][0]
      indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      localP = P[tgtinds]

      # Again, finding the local tensor, then vectorising,
      # without the overall einsum:
      # localAtens = np.einsum("zi,yj,xk->zyxijk",
      #                       coefsZ,coefsY,coefsX)
      # localA = localAtens.reshape((-1,q13))
      # localAp = np.matmul(localA,localP)
      # localAp = localAp.reshape((nindZ,nindY,nindX))

      localAp = np.einsum(
        "zi,yj,xk,ijk->zyx".encode("ascii","ignore"),
        coefsZ,coefsY,coefsX,
        localP.reshape((q1,q1,q1)),
        optimize=True )

      pred[firstZ:firstZ+nindZ,
           firstY:firstY+nindY,
           firstX:firstX+nindX] = localAp

t_end=time.time()
print(t_end)
print(t_end-t_start)



imgresnii = nib.Nifti1Image(pred, inimg.affine, inimg.header)
nib.save(imgresnii,outfile) 

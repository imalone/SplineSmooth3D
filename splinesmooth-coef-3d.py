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

infile='adni3-1006.nii.gz'
inimg = nib.load(infile)
inimgdata = inimg.get_fdata()

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

t_start=time.time()
t_last=t_start
print(t_start)
q1=q+1
q13 = q1**3
indsZpattern = np.tile(range(0,q1),q1*q1) 
indsYpattern = np.tile(np.repeat(range(0,q1),q1),q1) * kntsArr[0][0]
indsXpattern = np.repeat(range(0,q1),q1*q1) * kntsArr[0][0] * kntsArr[1][0]

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
      coefs = orderedProduct_asOuter(cZ,cY,cZ)
      coefsx = np.multiply(coeffs, inimgdata[Z,Y,X])
      indsZ = indsZpattern + cIndZ
      indsY = indsYpattern + cIndY * kntsArr[0][0]
      indsX = indsXpattern + cIndX * kntsArr[0][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      Atx[tgtinds] += coefsx
      ## About 20us per voxel to reach this point.
      AtAadd = np.outer(coefs,coefs)
      ## 25us, outer products is about 5us

      ## A number of different AtA adding strategies to try to improve speed
      ## With ix_ indexed adding, got about 66us per voxel, addition step
      ## with ix_ is about 42us.
      #AtA[np.ix_(tgtinds,tgtinds)] += AtAadd
      ## Flat indexed adding for AtA whole iteration about 44us per
      ## voxel (reshape) or 48us per vox (.flat):
      ## flatind calculation takes about 9us, addition about 11us.
      flatinds = tgtinds.reshape((q13,1)) + totalPar * tgtinds.reshape((1,q13))
      AtAflat[flatinds.reshape(q13**2)] += AtAadd.reshape((q13**2))
      #AtAflat[flatinds.flat] += AtAadd.flat
      #t_now = time.time()
      #print(t_now-t_last)
      #t_last=t_now
t_end=time.time()
print(t_end)
print(t_end-t_start)

# A hack, but just to get something invertible. Should be invertible
# as full coverage, but not, indicates an earlier mistake.
# Possibly was an error in setup of indsXpattern.
cheatsmoothing = False:
if cheatsmoothing:
    J=np.diag([0.01]*AtA.shape[0])
else:
    J=np.zeros(AtA.shape)

AtAinv = np.linalg.inv(AtA+J)
P = np.matmul(AtAinv,Atx)

## At this point P appears to have a heck of a lot of zero elements...
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
      coefs = orderedProduct_asOuter(cZ,cY,cZ).flat
      indsZ = indsZpattern + cIndZ
      indsY = indsYpattern + cIndY * kntsArr[0][0]
      indsX = indsXpattern + cIndX * kntsArr[0][0] * kntsArr[1][0]
      tgtinds = indsZ + indsY + indsX
      pred[Z,Y,X] = np.inner(P[tgtinds],coefs)

imgresnii = nib.Nifti1Image(pred, inimg.affine, inimg.header)
nib.save(imgresnii,"test-splinesmooth3d.nii.gz") 

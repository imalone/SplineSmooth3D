#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time
import sys
import argparse

import nibabel as nib
from splinesmooth3d import SplineSmooth3D

infile="adni3-1006.nii.gz"
outfile="test-splinesmooth3d.nii.gz"
testfile="test-model.nii.gz"
realData=True
# Storing the whole A matrix is potentially faster, but needs *lots* of
# memory, so only practical for small images and testing
useA=False
# Timing information during voxel addition loop.
reportTimeSteps=False

print("Start")

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
    aff = np.diag([1]*4)
    inimg = nib.nifti1.Nifti1Image(inimgdata,aff)
    nib.save(inimg,testfile) 



shape=inimg.shape
voxsizes=nib.affines.voxel_sizes(inimg.affine)
print (voxsizes)
print("Data loaded, set up class")

q = 3


for spacing in [75]:
    dm="minc"
    splsm3d = SplineSmooth3D(inimgdata, voxsizes, spacing,
                             Lambda=0.01, dofit=False,
                             domainMethod=dm)
    print("Fitting")
    splsm3d.fit(reportingLevel=2)
    #for L in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    for L in [0.00099, 0.0010, 0.00101]:
        print("Solving")
        splsm3d.solve(reportingLevel=1,Lambda=L)
        print("Predicting S:{} L:{}".format(spacing,L))
        pred = splsm3d.predict(reportingLevel=1)

        imgresnii = nib.Nifti1Image(pred, inimg.affine, inimg.header)
        nib.save(imgresnii,"test-5-dm{}-S{}-L{:.05f}.nii.gz".format(dm,spacing,L))

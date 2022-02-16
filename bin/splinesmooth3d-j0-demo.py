#!/usr/bin/env python

from builtins import *

import time
import sys
import argparse

import nibabel as nib
from splinesmooth3d import SplineSmooth3D

infile="adni3-1006.nii.gz"
outfile="test-splinesmooth3d.nii.gz"
maskfile="adni3-1006_mask.nii.gz"
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
    maskimg = nib.load(maskfile)
    mask = maskimg.get_fdata()
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
    mask=None

shape=inimg.shape
voxsizes=nib.affines.voxel_sizes(inimg.affine)
print (voxsizes)
print("Data loaded, set up class")

q = 3


for spacing in [75]:
    dm="minc"
    splsm3d = SplineSmooth3D(inimgdata, voxsizes, spacing,
                             dofit=False, costDerivative=0, Lambda=1e-7,
                             voxelsLambda=True,
                             domainMethod=dm, mask=mask)
    print("Fitting")
    splsm3d.fit(reportingLevel=2)

    for L in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        print("Solving")
        splsm3d.solve(reportingLevel=1, Lambda=L)
        print("Predicting S:{} L:{}".format(spacing, L))
        pred = splsm3d.predict(reportingLevel=1)

        imgresnii = nib.Nifti1Image(pred, inimg.affine, inimg.header)
        nib.save(imgresnii,"test-j0-l{:.0e}-dm{}-S{}.nii.gz".format(L,dm,spacing))

        dosub=False
        if dosub:
            splsm3dsub = splsm3d.promote()
            print("Fitting S:{} L:{} promoted".format(spacing, L))
            splsm3dsub.fit(reportingLevel=1)
            print("Solving S:{} promoted".format(spacing))
            splsm3dsub.solve(reportingLevel=1)
            print("Predicting S:{} L:{} promoted".format(spacing, L))
            predsub = splsm3dsub.predict(reportingLevel=1)

            print("Saving")
            imgresnii = nib.Nifti1Image(predsub, inimg.affine, inimg.header)
            nib.save(imgresnii,"test-j0-l{:.0e}-sub-dm{}-S{}.nii.gz".format(L,dm,spacing))

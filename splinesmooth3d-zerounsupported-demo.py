#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time
import sys
import argparse

import nibabel as nib
from splinesmooth3d import SplineSmooth3D, SplineSmooth3DUnregularised

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
    splsm3d = SplineSmooth3DUnregularised(inimgdata, voxsizes, spacing,
                             dofit=False,
                             domainMethod=dm, mask=mask)
    print("Fitting")
    splsm3d.fit(reportingLevel=2)
    print("Solving")
    splsm3d.solve(reportingLevel=1)
    print("Predicting S:{}".format(spacing))
    pred = splsm3d.predict(reportingLevel=1)

    imgresnii = nib.Nifti1Image(pred, inimg.affine, inimg.header)
    nib.save(imgresnii,"test-zerounsup-dm{}-S{}.nii.gz".format(dm,spacing))

    dosub=True
    if dosub:
        splsm3dsub = splsm3d.promote()
        print("Fitting S:{} promoted".format(spacing))
        splsm3dsub.fit(reportingLevel=1)
        print("Solving S:{} promoted".format(spacing))
        splsm3dsub.solve(reportingLevel=1)
        print("Predicting S:{} promoted".format(spacing))
        predsub = splsm3dsub.predict(reportingLevel=1)

        print("Saving")
        imgresnii = nib.Nifti1Image(predsub, inimg.affine, inimg.header)
        nib.save(imgresnii,"test-zerounsup-sub-dm{}-S{}.nii.gz".format(dm,spacing))

#!/usr/bin/env python

from builtins import *

import os
import argparse

import nibabel as nib
import numpy as np

FileType=argparse.FileType
parser = argparse.ArgumentParser(description='Test multilevel bias corrector.')
parser.add_argument('--infile','-i', metavar='INIMAGE',
                    help='input file', required=True)
parser.add_argument('--outfile','-o', metavar='CSVFILE',
                    help='outfile file', required=True)
parser.add_argument('--along','-a', metavar='x,y,z(,t)', type=str,
                    help='x,y and z locations for profile, e.g. '+
                    '64,:,97 takes profile along y at x=64, z=97 '+
                    '(locations in zero-indexed voxels.', required=True)

args = parser.parse_args()
innii = nib.load(args.infile)
indata = innii.get_fdata()
along = args.along.split(",")
if 4 > len(along) < 3 :
    print ("--along should be three or four whole numbers and colons (:) "+
    "separated by commas.")
    os.exit(os.syserr)

if len(indata.shape) == 3:
    indata = np.expand_dims(indata,3)

selected = indata
skipaxis=0
for indices in along:
    if indices == ":":
        skipaxis += 1
    else:
        selected = np.take(selected, int(indices), skipaxis)

np.savetxt(args.outfile,selected.reshape(-1),fmt='%1.3e')

#!/usr/bin/env python

from builtins import *

import os
import os.path
import subprocess as sub
import tempfile
import nibabel as nib

infile='adni3-1006.nii.gz'
outfile="test-splineminc.nii.gz"


def applyMINCSmooth(inimg, lmb=0.01,dist=75,subsamp=2):
    # Python 3: tmpdir = tempfile.TemporaryDirectory()
    tmpdir = tempfile.mkdtemp()
    tmpfile=os.path.join(tmpdir,'tmpfile')
    tmpfilenii1="{}1.nii".format(tmpfile)
    tmpfilenii2="{}2.nii".format(tmpfile)
    tmpfilenii3="{}3.nii".format(tmpfile)
    tmpfilemnc1="{}1.mnc".format(tmpfile)
    tmpfilemnc2="{}2.mnc".format(tmpfile)

    nib.save(inimg,tmpfilenii1)
    convcmd = ['nii2mnc',tmpfilenii1,tmpfilemnc1]
    print(convcmd)
    sub.call(convcmd)
    # Can't do this.
    #tmpmncout = nib.Minc1Image(inimgdata, inimg.affine, inimg.header)
    #nib.save(tmpmncout,tmpfile)

    splcmd = ['spline_smooth','-lambda',lmb,
              '-distance',dist,
              '-subsample',subsamp,tmpfilemnc1,tmpfilemnc2]
    splcmd = ["{}".format(x) for x in splcmd]
    print(splcmd)
    sub.call(splcmd)
    convcmd = ['mnc2nii',tmpfilemnc2,tmpfilenii2]
    print(convcmd)
    sub.call(convcmd)
    voffcmd = ['nifti_tool','-mod_hdr',
               '-mod_field','vox_offset','352',
               '-in',tmpfilenii2,
               '-prefix',tmpfilenii3]
    sub.call(voffcmd)

    smimg = nib.load(tmpfilenii3)
    smimgdata = smimg.get_fdata()
    for rmfile in [tmpfilenii1,tmpfilenii2,tmpfilenii3,
              tmpfilemnc1,tmpfilemnc2] :
        os.remove(rmfile)
    os.rmdir(tmpdir)
    return smimgdata


inimg = nib.load(infile)
smimgdata = applyMINCSmooth(inimg)
smimg = nib.Nifti1Image(smimgdata, inimg.affine, inimg.header)
nib.save(smimg,outfile)

#!/usr/bin/env python

import numpy as np
# https://github.com/scipy/scipy/blob/v1.3.0/scipy/interpolate/_bspl.pyx#L163
from scipy.interpolate._bspl import evaluate_all_bspl

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


"""
evaluate_all_bspl(const double[::1] t, int k, double xval, int m, int nu=0):

Evaluate the ``k+1`` B-splines which are non-zero on interval ``m``.
    Parameters
    ----------
    t : ndarray, shape (nt + k + 1,)
        sorted 1D array of knots
    k : int
        spline order 
    xval: float
        argument at which to evaluate the B-splines
    m : int
        index of the left edge of the evaluation interval, ``t[m] <= x < t[m+1]``
    nu : int, optional
        Evaluate derivatives order `nu`. Default is zero.

   Returns
    -------
    ndarray, shape (k+1,)
        The values of B-splines :math:`[B_{m-k}(xval), ..., B_{m}(xval)]` if
        `nu` is zero, otherwise the derivatives of order `nu`.

### k, what scipy calls spline order, is actually degree,
### see their example k=3 for cubic splines
"""
def eval_nonzero_bspl(tfull,x,q=3,nu=0):
  tindstart = np.searchsorted(tfull,x,side="right")-1
  coeffstart = tindstart-q
  nonZeroCoeffs = evaluate_all_bspl(tfull,q,x,tindstart,nu=nu)
  if x == tfull[-(1+q)]:
      # Edge case, last internal knot is technically unsupported
      # since we are supporting from the left edge of the range,
      # but has its last spline coefficient zero, so can trim
      # it off and report coefficients, with a padding zero at
      # the start instead.
      coeffstart = coeffstart - 1
      nonZeroCoeffs = np.concatenate(([0],nonZeroCoeffs[0:q]))
  return coeffstart, nonZeroCoeffs

def knots_over_domain(a,b,spacing,q=3):
  nIntervals = np.ceil((b-a)/spacing).astype("int")
  nInternalKnots = nIntervals + 1
  nCoeffs = nIntervals + q
  tinternal = np.linspace(a,b,nInternalKnots)
  tend = np.linspace(spacing,spacing*(q-1),q)
  tfull = np.concatenate((tinternal[0]-np.flip(tend,0),
                          tinternal,
                          tinternal[-1]+tend))
  return nCoeffs, tfull
    

q=3

t = np.arange(-1,5,dtype="double")
tpre=np.arange(-5,-2,dtype="double")
tpost=np.arange(5,8,dtype="double")
text = np.concatenate((tpre,t,tpost))


x=-1

evaluate_all_bspl(text,q,x,np.searchsorted(text,x,side="right")-1)
x=3.9999
evaluate_all_bspl(text,q,x,np.searchsorted(text,x,side="right")-1)

nC, knts = knots_over_domain(0,2*np.pi,0.5,q=q)

testarr = np.linspace(0,2*np.pi,50)
testclean = np.cos(testarr)
testdat = testclean + 0.2*(np.random.rand(testarr.shape[0])-0.5)

A=np.zeros((nC,testarr.shape[0]))
for N in range(testarr.shape[0]):
    cInd, C = eval_nonzero_bspl(knts,testarr[N],q=q)
    A[cInd:(cInd+q+1),N] = C

Ax = np.matmul(A,testdat)
AAt = np.matmul(A, A.transpose())

AAtinv = np.linalg.inv(AAt)

# Probably better replaced with Cholesky solver after
# testing.
fitcoef = np.matmul(AAtinv, Ax)

pred = np.matmul(A.transpose(),fitcoef)

plt.plot(testarr,testclean,"lightblue")
plt.plot(testarr,testdat,"orangered")
plt.plot(testarr,pred,"green")
plt.show(block=False)

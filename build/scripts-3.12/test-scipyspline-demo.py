#!python

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from test_scipyspline import eval_nonzero_bspl, knots_over_domain
    

q=3

t = np.arange(-1,5,dtype="double")
tpre=np.arange(-5,-2,dtype="double")
tpost=np.arange(5,8,dtype="double")
text = np.concatenate((tpre,t,tpost))

"""
x=-1

evaluate_all_bspl(text,q,x,np.searchsorted(text,x,side="right")-1)
x=3.9999
evaluate_all_bspl(text,q,x,np.searchsorted(text,x,side="right")-1)
"""

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
plt.show(block=True)

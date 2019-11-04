from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time

import numpy.linalg as linalg
import numpy as np

from test_scipyspline import knots_over_domain, eval_nonzero_bspl


class SplineSmooth3D:
  def __init__(self,data,voxsizes,spacing,mask=None,
               q=3,
               domainMethod="centre",
               Lambda=None, dofit=True):
    # data, 3D numpy array of data to fit
    # voxsizes, voxel size in each array direction
    # spacing spacing for control points
    # mask optional mask for fitting
    # q spline degree, q=3 cubic
    # domainMethod, how to place control points, see knots_over_domain
    # Lambda smoothing parameter
    # dofit whether to fit on initialisation
    self.data = data
    self.shape = data.shape
    self.voxsizes = voxsizes
    self.spacing = spacing
    self.mask = mask
    self.q=q
    self.domainMethod = domainMethod
    self.Lambda=Lambda
    self.AtA=None
    self.Atx=None
    self.P=None
    self.Jsub=None
    self.J=None
    if ( np.min(self.shape) < 2 or len(self.shape) !=3 or
         voxsizes.size != 3 ):
      # Not too difficult to extend to 1D/2D, would
      # need to special case the support coefficients
      # a bit, the regularisation would also need some
      # work
      raise ValueError("SplineSmooth3D currently only "+
                       "works with 3D, non-flat arrays")
    if (q !=3):
      # Least squares spline fitting will work, but
      # regularisation needs extending to accomodate
      # non cubic-splines
      raise ValueError("SplineSmooth3D only cubic "+
                       "splines currently supported")
    if (not spacing > 0):
      raise ValueError("SplineSmooth3D, knot spacing "+
                       "must be > 0")

    self.setupKCP()
    if Lambda is not None:
      self.Jsub = self.buildJ()
    if(dofit):
      self.fit()
    return None


  def coefIntervalIter(self, reportingLevel=0):
    # This iterator method is used for the common iteration loop
    # over supported intervals: generates target indices into
    # the parameter array (tgtinds), the coefficient lists to form
    # the local support tensor (coefs*), data ranges (range*)
    # and coefficient starting indices in each dimension (cInd*),
    # the last mostly for debugging purposes
    kntsArr=self.kntsArr
    invCoefArr=self.invCoefArr
    indsXpattern = self.indsXpattern
    indsYpattern = self.indsYpattern
    indsZpattern = self.indsZpattern
    t_start=time.time()
    t_last=t_start
    for cIndZ in range(len(invCoefArr[0])):
      firstZ, coefsZ = invCoefArr[0][cIndZ]
      nindZ=coefsZ.shape[0]
      for cIndY in range(len(invCoefArr[1])):
        firstY, coefsY = invCoefArr[1][cIndY]
        nindY=coefsY.shape[0]
        for cIndX in range(len(invCoefArr[2])):
          firstX, coefsX = invCoefArr[2][cIndX]
          nindX=coefsX.shape[0]

          indsX = indsXpattern + cIndX
          indsY = indsYpattern + cIndY * kntsArr[2][0]
          indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
          tgtinds = indsZ + indsY + indsX
    
          rangeZ =[firstZ,firstZ+nindZ]
          rangeY =[firstY,firstY+nindY]
          rangeX =[firstX,firstX+nindX]
          if reportingLevel >=2 :
            t_new=time.time()
            t_step=t_new-t_last
            t_last=t_new
            print("{} {} {}: {:.2g}seconds".format(cIndZ,cIndY,cIndX,
                                             t_step))
          yield cIndZ,cIndY,cIndX, \
          coefsZ,coefsY,coefsX, \
          rangeZ,rangeY,rangeX, \
          tgtinds


  def fit(self,data=None, reportingLevel=0):
    # reportingLevel: 0 none, 1 overall timing,
    # 2 by interval
    if (data is None):
      data = self.data
    if (not data.shape == self.data.shape):
      raise ValueError("Shape of data passed to fit must"+
                       "match that of data used to initialise")
    self.data=data
    totalPar = self.totalPar
    needAtA = self.AtA is None
    if needAtA:
      self.AtA = np.zeros((totalPar,totalPar))
      AtA = self.AtA
      AtAflat = AtA.reshape((AtA.shape[0]*AtA.shape[1]))
    self.Atx = np.zeros(totalPar)
    Atx = self.Atx
    kntsArr = self.kntsArr
    invCoefArr = self.invCoefArr

    t_start=time.time()
    t_last=t_start

    q1=self.q+1
    q13 = q1**3

    indsXpattern = self.indsXpattern
    indsYpattern = self.indsYpattern
    indsZpattern = self.indsZpattern

    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      localData = data[rangeZ[0]:rangeZ[1],
                       rangeY[0]:rangeY[1],
                       rangeX[0]:rangeX[1]]

      # This is what we're actually doing, below, and may help
      # explain what might seem an odd order of indices
      #localAtens = np.einsum("zi,yj,xk->zyxijk",
      #                       coefsZ,coefsY,coefsX)
      #localA = localAtens.reshape((-1,q13))
      #localAtx = np.matmul(localA.transpose(),localData.reshape(-1))
      #localAtA = np.matmul(localA.transpose(),localA)
      # .encode("ascii","ignore") is necessary to avoid a
      # TypeError due to  from __future__ import (unicode_literals)
      localAtx = np.einsum(
        "xc,yb,za,zyx->abc".encode("ascii","ignore"),
        coefsX,coefsY,coefsZ,
        localData, optimize=True).reshape((-1))

      Atx[tgtinds] += localAtx
      if needAtA:
        localAtA = np.einsum(
          "xc,yb,za,zi,yj,xk->abcijk".encode("ascii","ignore"),
          coefsX,coefsY,coefsZ,
          coefsZ,coefsY,coefsX, optimize=True ).reshape((q13,q13))
        flatinds = tgtinds.reshape((q13,1)) + \
                   totalPar * tgtinds.reshape((1,q13))
        AtAflat[flatinds.reshape(q13**2)] += localAtA.reshape((q13**2))

    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)

    return AtA, Atx


    
  def solve(self,Lambda=None, reportingLevel=0):
    if Lambda is None:
      Lambda = self.Lambda
    if Lambda is not None and self.J is None:
      self.J = self.buildFullJ(reportingLevel=reportingLevel)
    if self.AtA is None:
      fit(data)
    J = self.J
    AtA=self.AtA
    Atx=self.Atx

    if Lambda is None:
      L = np.linalg.cholesky(AtA)
    else:
      L = np.linalg.cholesky(AtA + J * Lambda)
    p1=np.linalg.solve(L, Atx)
    self.P=np.linalg.solve(L.T.conj(),p1)
    return self.P



  def predict(self,reportingLevel=0):
    if self.P is None:
      solve(reportingLevel=reportingLevel)
    P=self.P
    q=self.q
    q1=q+1
    q13=q1**3
    pred = np.zeros(self.shape)

    t_start=time.time()
    t_last=t_start
    # Same process as earlier, but without AtA calculation.
    indsXpattern = self.indsXpattern
    indsYpattern = self.indsYpattern
    indsZpattern = self.indsZpattern

    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
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

      pred[rangeZ[0]:rangeZ[1],
           rangeY[0]:rangeY[1],
           rangeX[0]:rangeX[1]] = localAp
    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)
    return pred


    

  def buildFullJ(self,reportingLevel=0):
    if self.Jsub is None:
      self.Jsub = self.buildJ()
    datashape = self.shape
    Jsub=self.Jsub
    J=np.zeros(self.AtA.shape)
    invCoefArr = self.invCoefArr
    kntsArr=self.kntsArr
    Jflat=J.reshape(-1)
    indsXpattern = self.indsXpattern
    indsYpattern = self.indsYpattern
    indsZpattern = self.indsZpattern
    q = self.q
    q1 = q+1
    q13 = q1**3
    totalPar=self.totalPar
    t_start=time.time()
    t_last=t_start
    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      flatinds = tgtinds.reshape((q13,1)) + \
                 totalPar * tgtinds.reshape((1,q13))
     
      Jflat[flatinds.reshape(-1)] += Jsub.reshape(-1)
    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)
    nIntervals = np.prod(list(map(len,invCoefArr)))
    nPoints = np.prod(datashape) # fix if using mask
    J = J #* spacing
    return J


  def setupKCP(self):
    # setup knots and control points
    q = self.q
    q1 = q+1
    shape = self.shape
    voxsizes = self.voxsizes
    spacing = self.spacing
    domain = [[0,(nvox-1)*voxsize]
                   for (nvox, voxsize) in zip(shape, voxsizes)]

    kntsArr = [knots_over_domain(bound[0],bound[1],spacing,
                                 q=q, method=self.domainMethod)
               for bound in domain]
    self.kntsArr = kntsArr
    totalPar = 1
    for knts in self.kntsArr:
      totalPar *= knts[0]
    self.totalPar = totalPar
    # Form the tensor bases, 
    coefArr = [
      [ eval_nonzero_bspl(knts[1],voxsize*v,q=q) for v in range(nvox) ]
      for (knts, nvox, voxsize) in zip (kntsArr, shape, voxsizes)]
    self.coefArr = coefArr
    # Convert to the list of voxels by supported interval
    invCoefArr = [
      self.invertCoefList(coefList)
      for coefList in coefArr
    ]
    self.invCoefArr = invCoefArr

    # X is fastest changing: LAST index
    self.indsXpattern = np.tile(range(0,q1),q1*q1) 
    self.indsYpattern = np.tile(np.repeat(range(0,q1),q1),q1) * kntsArr[2][0]
    self.indsZpattern = np.repeat(range(0,q1),q1*q1) * kntsArr[2][0] * kntsArr[1][0]

    return


  def buildGammaMat(self, order, spacing=None, q=None):
    # dist knot spacing, order is derivative order, q is spline
    # degree, however only q=3 supported.
    # Implementing Shackleford et al. LNCS 7511 MICCAI 2012
    if spacing is None:
      spacing = self.spacing
    if q is None:
      q=self.q
    if q!=3:
      raise ValueError('buildGammaMat only supports q=3')
    B = np.array([[1, -3, 3, -1],
                  [4, 0, -6, 3],
                  [1, 3, 3, -3],
                  [0, 0, 0, 1]])
    spacing=float(spacing)
    R = np.diag(np.power(spacing,[0,-1,-2,-3]))
    n = np.arange(1,8)
    psi = np.power(spacing,n)/n
    D = [
      np.diag([1,1,1,1]),
      np.diag([1,2,3],-1),
      np.diag([2,6],-2)
    ]
    Q = np.matmul(B,np.matmul(R,D[order]))
    # xsi_{a,b} is a 4x4 matrix corresponding to
    # xsi_{a,b}_{i,j} = (q_a outer q_b)_{i,j}
    # where q_a is row a of Q
    xsi = np.einsum("ai,bj->abij",Q,Q)
    offset = range((Q.shape[0]-1),-(Q.shape[1]),-1)
    # like:
    # 1000 0100 0010
    # 0000 1000 0100  ... etc.
    # 0000 0000 1000
    # 0000 0000 0000
    sigmabases = np.array(
      [np.fliplr(np.diag([1]*(4-abs(thisOff)),thisOff))
       for thisOff in offset])
    # Don't need sigma itself, but it's abij,uij->abu
    gamma = np.einsum("abij,uij,u->ab",xsi,sigmabases,psi)
    return gamma


  def buildBigGammaMat(self, orders, spacing=None, q=None):
    # dist uniform knot spacing (assumed isotropic so far),
    # orders (dz, dy, dx) are derivative orders 0 to 2,
    # q=3 is spline degree (only 3 supported)
    # Implementing Shackleford et al. LNCS 7511 MICCAI 2012
    if q is None:
      q=self.q
    if spacing is None:
      q=self.spacing
    gammaZ = self.buildGammaMat(order=orders[0])
    gammaY = self.buildGammaMat(order=orders[1])
    gammaX = self.buildGammaMat(order=orders[2])
    bigGamma = np.einsum("ai,bj,ck->abcijk",
                         gammaZ,
                         gammaY,
                         gammaX)
    return bigGamma.reshape(np.power(gammaZ.shape,3))


  def buildJ(self,spacing=None,q=None):
    # dist assumed isotropic so far, though relatively simple
    # to generalise if needed (mainly need to do type checking)
    # only spline degree q=3 supported
    # Implementing Shackleford et al. LNCS 7511 MICCAI 2012
    # Analytic Regularization of Uniform Cubic B-spline
    # Deformation Fields
    # An alternative would be Wood 2016 https://arxiv.org/abs/1605.02446v1
    # which sets up a similar thing using evaluation at a number of
    # points of the splines to exactly fit the polynomial. This
    # more easily adapts to q!=3, however it needs a bit of furter work to
    # allow the mixed d2/dxdy that Shackleford has and which we need
    # for N3's thin-plate cost function, so sticking with this one.
    if q is None:
      q=self.q
    if spacing is None:
      q=self.spacing

    from scipy.misc import factorial
    J = None
    # fac: Confirmed with Gregory Sharp that Shackleford paper is
    # missing the 2* on the d2v/dxdy cross terms in eq. 18/19.
    # fac is the appropriate term from thin plate bending energy,
    # e.g. Wahba 1990, Spline Models for Observational Data,
    # eq. 12.1.5, or the Sled N3 paper eq. 26
    for dx in range(0,3):
      for dy in range(0,3):
        for dz in range(0,3):
          if (dx+dy+dz)==2:
            orders=(dx,dy,dz)
            # fac: see above
            fac = factorial(2)/np.prod(factorial([dx,dy,dz]))
            newJ = self.buildBigGammaMat(orders) * fac
            if J is None:
              J = newJ
            else:
              J += newJ
    return J

  @staticmethod
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



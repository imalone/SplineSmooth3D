from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import time

import numpy.linalg as linalg
import numpy as np

from test_scipyspline import knots_over_domain, eval_nonzero_bspl


class SplineSmooth3D:
  """3D smoothing tensor B-spline class for fields defined as numpy
  arrays, with thin-plate spline bending regularisation.

  Intended to replace spline smoothing of Sled et al. TMI 17, 1998,
  follows some details of https://github.com/BIC-MNI/N3/ with regard
  to knot placement. Use of Cholesky decomp for solver, Green and
  Silverman, Nonparametric regression[...] (1994); Wahba, Spline
  Models for Observational Data (1990). Analytic form of
  thin-plate-spline penalty Shackleford et al. LNCS 7511 MICCAI 2012
  (with correction to match form used in Sled, matching Wahba and
  other sources).

  Parameters
  ----------
    file_loc : str
    The file location of the spreadsheet
    print_cols : bool, optional
    A flag used to print the columns to the console (default is False)

  data : ndarray
      3D array matching shape of data to smooth
  voxsizes : tuple
      tuple of floats for voxel size in each dimension
  spacing : float
      knot spacing to use, units should match those of `voxsizes`
  mask : ndarray or None, optional
      ndarray same shape as `data`, 0 for exclude, 1 for include. May
      be of any type, including boolean
  q : int, optional
      Spline degree, only q=3 (cubic) supported by most functions in
      this class
  domainMethod : str, optional
      Method to use for knot placement, see knots_over_domain, default
      'centre'. Certain domainMethod values will produce different
      spacings.
  Lambda : float or None, optional:
      Smoothing parameter. Default None: don't use smoothing. Images
      without full support (when using masks) may fail to solve.
  dofit : bool, optional
      Fit on initialisation (with supplied data), default True.


  Attributes
  ----------
  data : ndarray
      Last data passed to fit
  shape : tuple
      Shape of data, set at instantiation
  voxsizes : tuple
      Dimensions of voxels in each direction
  spacing : float
      Knot spacing
  mask : ndarray or None
      Set at instantion, array same shape as data to fit
  q : int
      Spline degree, only q=3 (cubic) supported by most
      functions in this class
  domainMethod : str
      Method to use for knot placement, see knots_over_domain
  Lambda : float
      Requested spacing (same units as voxsize) for knots/control
      points. Certain domainMethod values will produce different
      spacings.
  mincLambda : bool
      # Not implemented yet, see solve. Whether to use an adjstment
      that makes lambda values compatible with MINC spline_smooth
  P : ndarray
      Fitted parameters as vector
  Atx : ndarray
      data projected to parameter space
  AtA : ndarray
      defines inner product on spline coefficients
  Jsub : ndarray
      thin-plate-energy matrix for the coefficients supporting
      an interval
  J : ndarray
      thin-plate-energy matrix for all coefficients

  Methods
  -------
  fit(data,reportingLevel=0)
      Fit new data. Invalidates current parameter solution. First use
      in a new instance will also build inner product matrix, re-running
      fit for updated data is faster than creating a new instance to fit.
  solve(Lambda=None, mincLambda=True, reportingLevel=0):
      Solve for parameters. Will fit current data first if not yet done.
      Can be re-run with a different value of Lambda to change smoothing.
  predict(reportingLevel=0)
      Predict smoothed data from current parameters. Will run all prior
      steps if not already performed.
  #coefIntervalIter(reportingLevel=0)
  #buildFullJ(reportingLevel=0):
  #setupKCP(self):
  #buildGammaMat(self, order, spacing=None, q=None):
  #buildBigGammaMat(self, orders, spacing=None, q=None):
  #buildJ(self,spacing=None,q=None):
  #invertCoefList (coefList):
  """

  def __init__(self,data,voxsizes,spacing,mask=None,
               q=3,
               domainMethod="centre",
               Lambda=None, dofit=True):
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

    if mask is None:
      self.mask = np.ones(self.shape,dtype=self.data.dtype)
    else:
      self.mask = mask.astype(self.data.dtype,copy=True)

    self.setupKCP()
    if Lambda is not None:
      self.Jsub = self.buildJ()
    if(dofit):
      self.fit()
    return None


  def coefIntervalIter(self, reportingLevel=0):
    """Generator for the common iteration loop over supported intervals

    Parameters
    ----------
    reportingLevel : int, optional
    If >=2 then report timing information for each inner loop

    Yields
    -------
    cIndZ,cIndY,cIndX : int
        Coefficient starting indices in each dimension (mainly
        for debugging)
    coefsZ,coefsY,coefsX : ndarray
        matrices of control point coefficients by direction for
        each grid step along that direction within the supported
        interval
    rangeZ,rangeY,rangeX : list
        pairs of indices representing data range of supported interval
        in each dimension
    tgtinds : ndarray
        target indices into the parameter array for control points
        supporting this interval
    """
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
    """fit current data, or update current data and fit

    fit does not solve for parameters, it calculates the projection of
    data to parameter space `Atx`, and if necessary (on the first run)
    the parameter inner product matrix `AtA`. This means re-using fit
    for updated data is quicker (about by about three times) than
    using a fresh instance. However the `AtA` matrix depends on the
    shape of data, the mask and the knot placement, a new instance is
    required if any of these are to be changed.

    Running fit will invalidate the current parameters solution `P`.

    Parameters
    ----------
    data : ndarray : optional
        New data to fit, if no new data then will fit the previously
        supplied data.
    reportingLevel : int, optional
        If >=1 then report time taken for the fitting loop, if >=2
        then will also report time for each inner loop (each interval)

    Returns
    -------
    AtA : ndarray
        Parameter inner product matrix
    Atx : ndarray
        Projection of data to parameter space

    """
    # reportingLevel: 0 none, 1 overall timing,
    # 2 by interval
    mask = self.mask
    if (data is None):
      data = self.data
    if (not data.shape == self.data.shape):
      raise ValueError("Shape of data passed to fit must"+
                       "match that of data used to initialise")
    self.data=data
    self.P=None # Invalidate P
    totalPar = self.totalPar
    needAtA = self.AtA is None
    if needAtA:
      self.AtA = np.zeros((totalPar,totalPar))
      AtAflat = AtA.reshape((AtA.shape[0]*AtA.shape[1]))
    self.Atx = np.zeros(totalPar)
    AtA = self.AtA
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

    AtAOptPath=None
    AtxOptPath=None

    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      localData = data[rangeZ[0]:rangeZ[1],
                       rangeY[0]:rangeY[1],
                       rangeX[0]:rangeX[1]]
      localMask = mask[rangeZ[0]:rangeZ[1],
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

      optimize="optimal".encode("ascii","ignore") # No, really

      AtxSum = "xc,yb,za,zyx,zyx->abc".encode("ascii","ignore")
      if AtxOptPath is None:
        AtxOptPath = np.einsum_path(AtxSum,
          coefsX,coefsY,coefsZ,
          localData, localMask, optimize=optimize)[0]
      localAtx = np.einsum(
        AtxSum,
        coefsX,coefsY,coefsZ,
        localData, localMask, optimize=AtxOptPath).reshape((-1))

      Atx[tgtinds] += localAtx
      if needAtA:
        AtASum = "xc,yb,za,zi,yj,xk,zyx->abcijk".encode("ascii","ignore") 
        if AtAOptPath is None:
          AtAOptPath = np.einsum_path(
          AtASum,
          coefsX,coefsY,coefsZ,
          coefsZ,coefsY,coefsX,
          localMask, optimize=optimize )[0]
        localAtA = np.einsum(
          AtASum,
          coefsX,coefsY,coefsZ,
          coefsZ,coefsY,coefsX,
          localMask, optimize=AtAOptPath ).reshape((q13,q13))
        flatinds = tgtinds.reshape((q13,1)) + \
                   totalPar * tgtinds.reshape((1,q13))
        AtAflat[flatinds.reshape(q13**2)] += localAtA.reshape((q13**2))

    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)

    return AtA, Atx


    
  def solve(self,Lambda=None, mincLambda=True, reportingLevel=0):
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
      lambdaFac = Lambda
      if mincLambda:
        # MINC calculates its bending energy integrals without
        # using knot distances, so the (d2f/dx2)^2 volume
        # integral that should have units 1/spacing doesn't.
        # Since our J is calculated with spacing, we need to
        # multiply back.
        # Additionally MINC also calculates its AtA matrix
        # with just the cube values in the Bspline coefficents,
        # not the 1/6 normally used (e.g. B1(u) = (1-u)^3/6)
        # this means its AtA (product of three tensors squared)
        # is a factor 6^3^2 higher than ours, to get the
        # equivalent internal Lambda to a requested Lambda_minc
        # we therefore need to multiply by spacing and divide
        # by 6^6.
        # Confirmed: tested this against MINC on a range of
        # image FOV, spacings, lambda, and resolutions.
        lambdaFac *= self.spacing / (6**3)**2
      L = np.linalg.cholesky(AtA + J * lambdaFac)
    p1=np.linalg.solve(L, Atx)
    self.P=np.linalg.solve(L.T.conj(),p1)
    return self.P



  def predict(self,reportingLevel=0):
    if self.P is None:
      self.solve(reportingLevel=reportingLevel)
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
      spacing=self.spacing
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
      spacing=self.spacing
    
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



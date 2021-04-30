from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import multivariate_normal
from numpy.linalg import inv
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from scipy.stats import norm, multivariate_normal


def gaussian(x, mean, var, normed=True):
    """
    returns probability density function (pdf) for x given a Gaussian with the
    specified mean and variance. All must be scalars.
    gaussian (1,2,3) is equivalent to scipy.stats.norm(2, math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.
    Parameters
    ----------
    x : scalar or array-like
        The value(s) for which we compute the distribution
    mean : scalar
        Mean of the Gaussian
    var : scalar
        Variance of the Gaussian
    normed : bool, default True
        Normalize the output if the input is an array of values.
    Returns
    -------
    pdf : float
        probability distribution of x for the Gaussian (mean, var). E.g. 0.101 denotes
        10.1%.
    Examples
    --------
    >>> gaussian(8, 1, 2)
    1.3498566943461957e-06
    >>> gaussian([8, 7, 9], 1, 2)
    array([1.34985669e-06, 3.48132630e-05, 3.17455867e-08])
    """

    pdf = ((2*math.pi*var)**-.5) * np.exp((-0.5*(np.asarray(x)-mean)**2.) / var)
    if normed and len(np.shape(pdf)) > 0:
        pdf = pdf / sum(pdf)

    return pdf


def _to_cov(x, n):
    """
    If x is a scalar, returns a covariance matrix generated from it
    as the identity matrix multiplied by x. The dimension will be nxn.
    If x is already a 2D numpy array then it is returned unchanged.
    Raises ValueError if not positive definite
    """

    if np.isscalar(x):
        if x < 0:
            raise ValueError('covariance must be > 0')
        return np.eye(n) * x

    x = np.atleast_2d(x)
    try:
        # quickly find out if we are positive definite
        np.linalg.cholesky(x)
    except:
        raise ValueError('covariance must be positive definit')

    return x


def multivariate_gaussian(x, mu, cov):
    """
    This is designed to replace scipy.stats.multivariate_normal
    which is not available before version 0.14. You may either pass in a
    multivariate set of data:
    .. code-block:: Python
       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)
    or unidimensional data:
    .. code-block:: Python
       multivariate_gaussian(1, 3, 1.4)
    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov
    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.
    equivalent calls:
    .. code-block:: Python
      multivariate_gaussian(1, 2, 3)
      scipy.stats.multivariate_normal(2,3).pdf(1)
    Parameters
    ----------
    x : float, or np.array-like
       Value to compute the probability for. May be a scalar if univariate,
       or any type that can be converted to an np.array (list, tuple, etc).
       np.array is best for speed.
    mu :  float, or np.array-like
       mean for the Gaussian . May be a scalar if univariate,  or any type
       that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.
    cov :  float, or np.array-like
       Covariance for the Gaussian . May be a scalar if univariate,  or any
       type that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.
    Returns
    -------
    probability : float
        probability for x for the Gaussian (mu,cov)
    """

    # force all to numpy.array type, and flatten in case they are vectors
    x = np.array(x, copy=False, ndmin=1).flatten()
    mu = np.array(mu, copy=False, ndmin=1).flatten()

    nx = len(mu)
    cov = _to_cov(cov, nx)


    norm_coeff = nx*math.log(2*math.pi) + np.linalg.slogdet(cov)[1]

    err = x - mu
    if sp.issparse(cov):
        numerator = spln.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)

    return math.exp(-0.5*(norm_coeff + numerator))

def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.
    Parameters
    ----------
    P : nd.array shape (2,2)
       covariance matrix
    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.
    Returns (angle_radians, width_radius, height_radius)
    """

    U, s, _ = np.linalg.svd(P)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = deviations * math.sqrt(s[0])
    height = deviations * math.sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)

def plot_3d_covariance(mean, cov):
    """ plots a 2x2 covariance matrix positioned at mean. mean will be plotted
    in x and y, and the probability in the z axis.
    Parameters
    ----------
    mean :  2x1 tuple-like object
        mean for x and y coordinates. For example (2.3, 7.5)
    cov : 2x2 nd.array
       the covariance matrix
    """

    # compute width and height of covariance ellipse so we can choose
    # appropriate ranges for x and y
    o, w, h = covariance_ellipse(cov, 3)
    # rotate width and height to x,y axis
    wx = abs(w*np.cos(o) + h*np.sin(o)) * 1.2
    wy = abs(h*np.cos(o) - w*np.sin(o)) * 1.2


    # ensure axis are of the same size so everything is plotted with the same
    # scale
    if wx > wy:
        w = wx
    else:
        w = wy

    minx = mean[0] - w
    maxx = mean[0] + w
    miny = mean[1] - w
    maxy = mean[1] + w

    xs = np.arange(minx, maxx, (maxx-minx)/40.)
    ys = np.arange(miny, maxy, (maxy-miny)/40.)
    xv, yv = np.meshgrid(xs, ys)

    zs = np.array([100.* multivariate_gaussian(np.array([x, y]), mean, cov) \
                   for x, y in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax = plt.gca(projection='3d')
    ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, cmap=cm.autumn)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # For unknown reasons this started failing in Jupyter notebook when
    # using `%matplotlib inline` magic. Still works fine in IPython or when
    # `%matplotlib notebook` magic is used.
    x = mean[0]
    zs = np.array([100.* multivariate_gaussian(np.array([x, y]), mean, cov)
                   for _, y in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    y = mean[1]
    zs = np.array([100.* multivariate_gaussian(np.array([x, y]), mean, cov)
                   for x, _ in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

def plot_correlated_data(X, Y, xlabel=None,
                         ylabel=None, equal=True):

    """Plot correlation between x and y by performing
    linear regression between X and Y.
    X: x data
    Y: y data
    xlabel: str
        optional label for x axis
    ylabel: str
        optional label for y axis
    equal: bool, default True
        use equal scale for x and y axis
    """


    plt.scatter(X, Y)

    if xlabel is not None:
        plt.xlabel(xlabel);

    if ylabel is not None:
        plt.ylabel(ylabel)

    # fit line through data
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, np.asarray(X)*m + b,color='k')
    if equal:
        plt.gca().set_aspect('equal')
    plt.show()

def display_stddev_plot():
    xs = np.arange(10,30,0.1)
    var = 8;
    stddev = math.sqrt(var)
    p2, = plt.plot (xs,[gaussian(x, 20, var) for x in xs])
    x = 20+stddev
    # 1std vertical lines
    y = gaussian(x, 20, var)
    plt.plot ([x,x], [0,y],'g')
    plt.plot ([20-stddev, 20-stddev], [0,y], 'g')

    #2std vertical lines
    x = 20+2*stddev
    y = gaussian(x, 20, var)
    plt.plot ([x,x], [0,y],'g')
    plt.plot ([20-2*stddev, 20-2*stddev], [0,y], 'g')

    y = gaussian(20,20,var)
    plt.plot ([20,20],[0,y],'b')

    x = 20+stddev
    ax = plt.gca()
    ax.annotate('68%', xy=(20.3, 0.045))
    ax.annotate('', xy=(20-stddev,0.04), xytext=(x,0.04),
                arrowprops=dict(arrowstyle="<->",
                                ec="r",
                                shrinkA=2, shrinkB=2))
    ax.annotate('95%', xy=(20.3, 0.02))
    ax.annotate('', xy=(20-2*stddev,0.015), xytext=(20+2*stddev,0.015),
                arrowprops=dict(arrowstyle="<->",
                                ec="r",
                                shrinkA=2, shrinkB=2))


    ax.xaxis.set_ticks ([20-2*stddev, 20-stddev, 20, 20+stddev, 20+2*stddev])
    ax.xaxis.set_ticklabels(['$-2\sigma$', '$-1\sigma$','$\mu$','$1\sigma$', '$2\sigma$'])
    ax.yaxis.set_ticks([])
    ax.grid(None, 'both', lw=0)

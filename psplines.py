

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import pascal



def pascal_triangle_row(n):
    """
    Get the n-essim row of pascal's triangle.


    Arguments
    ---------

        row : int
            The desired deepth level at the triangle.

    Returns
    -------

        r : np.ndarray (n+1,)
    """

    r = pascal(n+1)
    i = np.arange(n+1)
    r = r[i, np.flip(i)]
    return r
#

def alternating_sign(n, start=1):
    """
    Return an array alternating 1, -1.

    Arguments
    ---------

        n : int
            Length of the array

        start : {1, -1}
            The value with which start

    Returns
    -------
        alt : np.ndarray
    """

    alt = np.empty(n)
    alt[::2] = start
    alt[1::2] = -start
    return alt
#

def finite_differences(arr, order=1, mode='centered'):
    """
    Compute the finite differences of a given array.

    Arguments
    ---------

        arr : np.ndarray
            The array to which compute the finite differences

        order : int, optional
            Default 1. The derivative order

        mode : {'forward', 'backward'}, optional
            Default 'forward'. The method to be used.

    Returns
    -------
        fd_arr : np.ndarray
            The computed finite differences.
    """

    b = pascal_triangle_row(order)
    sign = alternating_sign(order+1)

    if mode == 'forward':
        fd_arr = [(sign*b*arr[i:i+order+1]).sum() for i in range(len(arr)-order)]

    elif mode == 'backward':
        fd_arr = [(sign*b*arr[i-order:i]).sum() for i in range(order, len(arr))]

    else:
        raise ValueError(f"Wrong value ({mode}) for mode arguments. Available options are {{'forward', 'backward'}}.")

    return np.array(fd_arr)
    #
#

def get_uniform_knot_vector(xb, xe, n, mode='internal', k=3, ext=None):
    """
    Generates a B-Spline uniform knot vector.

    Given the interval [xb, xe], this function returns the even partition in n internal k-nots.
    The mode argument allows the knot vector to account for the different boundary conditions.
    In 'internal' mode only internal knots are returned, leaving the boundarys undefined.
    In 'complete', the extreme of the interval are repeated k+1 times to make the spline interpolate
    the last control point/coefficient. In 'periodic', the extrema of the interval is extended k+1
    times, preserving the spacing between knots. Additionally, an extra 'extended' metod allows to
    perform a similar extension, but the amount extensions is controlled by the ext argument, that
    is ignored in any other mode.


    Arguments
    ---------

    xb, xe : float
        The begin and end of the definition interval.

    n : int
        Number of internal knots.

    k : int, optional
        Default is 3. The degree of the spline.

    mode : {'internal', 'complete', 'extended', 'periodic'} , optional
        Default is 'internal'.

        If mode == 'internal' then t is the even spaced partition of [xb, xe]
        without the extrema of the interval.

        If mode == 'complete' t contains [xb]*(k+1) at the beginning and
        [xe]*(k+1) at the end.

        If mode = 'extended' (ext must be passed), it extends ext times the
        knot vector from both ends preserving the spacing.

        mode 'periodic', is the equivalent to setting mode='extended' and ext=k.
        It is useful when combined with scipy B-Splines functions.

    ext : int
        Default is None. Ignored if mode != 'extended'. The times to extend the knot vector from
        both ends preserving the separation between nodes.


    Returns
    -------
        t : np.ndarray
            The knot vector.

    """

    t = np.linspace(xb, xe, n+2)
    d = (xe-xb)/(n+1)

    if mode == 'periodic':
        mode = 'extended'
        ext = k

    if mode == 'internal':
        t = t[1:-1]

    elif mode == 'complete':
        t = np.concatenate([[t[0]]*k, t, [t[-1]]*k])

    elif mode == 'extended':
        if ext is None:
            raise ValueError(f"Wrong value ({ext}) for ext argument using extended mode.")

        t = np.concatenate([t[0]+np.arange(-ext, 0)*d, t, t[-1]+np.arange(ext+1)[1:]*d])

    else:
        raise ValueError(f"Wrong value ({mode}) for mode argument. The options are {{'internal', 'complete', 'extended', 'periodic'}}. ")

    return t
#

def optimization_expression(coeffs, y, Bx, l=0.0):
    """
    Function to compute the expression to be optimized.

    Parameter l allows to penalized the curvature of the approximation by adding the 2 order forward finite diference of the coefficients.

    :math:`min_{\theta_j \in \mathbb{R}}\sum_{i=0}^N (y_i - \sum_{j=0}^m c_j*B_{j,3,t}(x))^2 + \lambda (\sum_{j=k}^{m}c_j-2c_{j-1}+c_{j-2})`

    Arguments
    ---------

        coeffs : np.ndarray (m,)
            The coefficients of the splines to use.

        Bx : np.ndarray (N, m)
            The matrix with the evaluation of the domain points by the basis splines.

        y : np.ndarray (N,)
            The observations of the function at the domain points.

    Returns
    -------
        err : float
            The evaluation of the error expression to be minimized.
    """

    err = ((y - Bx @ coeffs.reshape(-1, 1))**2).sum()
    if l:
        err += l*(np.convolve(coeffs, [1, -2, 1], mode='valid')**2).sum()

    return err
#

def optimization_OOP(coeffs, x, y, t, k, l):


    err = ((y - BSpline.construct_fast(t, coeffs, k)(x))**2).sum()
    if l:
        err += (finite_differences(coeffs, order=2, mode='forward')**2).sum()

    return err
#

def get_function_control_points(knots, coeff, k, padded = True):

    """
    Compute the control points (t*_j,c_j)_j=1^n of a spline function
    where c_j are the coefficients and t*_j is the average position
    of the knots for a given coefficient.
            t*_j = (t_{j+1}+...+t_{j+k}) / k,     1<=j<=n

    This functions assumes the coeff array/list to be paded with k+1 zero.
    This is due to scipy

    Args:
    -----
        knots : np.array/list of float
            The knot vector of the given spline function
        coeff : np.array/list of float
            The coefficients of the given spline function
        k : int
            Degree of the given spline function
        padded : bool (optional)
            Default True. Whether the coeff array contains k+1 trailing zeros.

    Returns:
    ---------
        control_points : np.array
    """

    if padded: coeff = coeff[:-(k+1)]
    t_ = [np.mean(knots[i+1 : i+k+1]) for i in range(len(coeff))]
    control_points = np.array((t_, coeff))
    return control_points
#
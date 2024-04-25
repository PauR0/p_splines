


import numpy as np
from scipy.interpolate import LSQBivariateSpline, BivariateSpline
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from psplines import get_uniform_knot_vector, bivariate_optimization_loss, get_bispline_function_control_points


# Semi-periodic BivariateSpline approximation with missing data.
# Given a set of noisy data {(x_i, y_i, f(x_i, y_i))}, our aim is to approximate the function f: [a,b]x[c,d]=D -> R
# by means of uniform cubic bivariate splines. However, we have some restrictions on the approximating
# spline. First, we would like to be able to penalize the wiggling of the approximation. This
# is adressed by penalyzed spline approximation (P-splines) by adding a regulatizing term on
# the optimization expression that penalizes the 2-derivative (curvature) of the function.
# Additionally, we would like the spline to be periodic on the y-axis. To all these requirements
# we now add the extra complexity we will assume that our input data leaves uncovered a patch in
# the functions domain, i.e
#   \exists D' \subsetneq D so that \forall p in {(x_i, y_i)} -> p \not\in D'


# Let's play with the function f(x,y) = log(x)*sin(y) on the rectangle [0.05, 5]x[0, 2*π].
# Let's build artificially noisy data {(x_i, y_i, log(x_i)sin(y_i)+eps)}. However, let us remove
# the square

Nx, Ny    = 51, 51
eps       = 2e-1
kx, ky    = 3, 3

a, b, c, d = 0.05, 4, 0, 2*np.pi
am, bm, cm, dm = 0.5, 2, 2, 4


g = np.log
h = np.sin
f = lambda x, y: g(x)+h(y)

x = np.linspace(a, b, Nx)
y = np.linspace(c, d, Ny)
xy = np.array([a.ravel() for a in np.meshgrid(x, y)])
ids = (xy[0]<am) | (bm<xy[0]) | (xy[1]<cm) | (dm<xy[1])
x = xy[0][ids]
y = xy[1][ids]

N=len(x)
z = f(x, y) + np.random.normal(loc=0.0, scale=eps, size=N)

#fg = plt.figure()
#ax = fg.add_subplot(projection='3d')
#ax.scatter(x, y, z)
#plt.show()

nx_spl, ny_spl = 10, 10 #Numer knots

#The standard scipy least squares approximation
tx_spl = get_uniform_knot_vector(a, b, nx_spl, mode='internal', k=kx)
ty_spl = get_uniform_knot_vector(c, d, ny_spl, mode='internal', k=ky)
spl = LSQBivariateSpline(x=x, y=y, z=z, tx=tx_spl, ty=ty_spl, bbox=[a,b,c,d], kx=kx, ky=ky)

#However, with this approximation, we are not imposing periodicity:
mse = lambda x: ((x**2)/x.size).sum()
print('-'*30)
xaux = np.linspace(a, b)
caux = np.array([c]*xaux.shape[0])
daux = np.array([d]*xaux.shape[0])

print(f"spl(x, c)   - spl(x, d)   = {mse(spl(xaux, caux) - spl(xaux, daux))}")
print(f"spl'(x, c)  - spl'(x, d)  = {mse(spl(xaux, caux, dx=1, dy=1) - spl(xaux, daux, dx=1, dy=1))}")
print(f"spl''(x, c) - spl''(x, d) = {mse(spl(xaux, caux, dx=2, dy=2) - spl(xaux, daux, dx=2, dy=2))}")
print('-'*30)


#Now let us compute step by step the penalized constrained optimization using scipy optimize subpackage.
nx, ny = 10, 10  #Since high frequency is going to be penalized, we can add extra degrees of freedom.
l = 1 #Penalty to laplacian (second derivative analog)
tx_pspl = get_uniform_knot_vector(a, b, nx, mode='complete', k=kx)
ty_pspl = get_uniform_knot_vector(c, d, ny, mode='periodic', k=ky)
n_coeff = (nx+(kx+1)) * (ny+(ky+1))

x0 = np.full(shape=(n_coeff,), fill_value=y.mean())
tck = lambda coeff: (tx_pspl, ty_pspl, coeff, kx, ky)
column = lambda coeff, col: coeff.reshape(nx+(kx+1), ny+(ky+1))[:,col]

cons = [
        {'type':'eq', 'fun': lambda c: column(c, 0) - column(c, -1)}, #Periodicity conditions
        {'type':'eq', 'fun': lambda c: column(c, 1) - column(c, -2)}, #Periodicity conditions
        {'type':'eq', 'fun': lambda c: column(c, 2) - column(c, -3)}, #Periodicity conditions
    ]

res = minimize(fun=bivariate_optimization_loss, x0=x0, args=(x, y, z, tx_pspl, ty_pspl, kx, ky, l),
                        method='SLSQP', constraints=cons)

pbispl         = BivariateSpline()
pbispl.tck     = tx_pspl, ty_pspl, res.x
pbispl.degrees = kx, ky

print('-'*30)
print(f"pbispl(x, c)   - pbispl(x, d)   = {mse(pbispl(xaux, caux) - pbispl(xaux, daux))}")
print(f"pbispl'(x, c)  - pbispl'(x, d)  = {mse(pbispl(xaux, caux, dx=1, dy=1) - pbispl(xaux, daux, dx=1, dy=1))}")
print(f"pbispl''(x, c) - pbispl''(x, d) = {mse(pbispl(xaux, caux, dx=2, dy=2) - pbispl(xaux, daux, dx=2, dy=2))}")
print('-'*30)



fg = plt.figure()
xx = np.linspace(a, b, Nx)
yy = np.linspace(c, d, Ny)
X, Y = np.meshgrid(xx, yy)
Z = f(X, Y)
ax = fg.add_subplot(131, projection='3d')
ax.scatter(x, y, z, label='Data set')

Z_spl = spl(X, Y, grid=False)
ax.plot_surface(X, Y, Z_spl, linewidth=0, color='r', label='bispl')
if False:
    X_cp, Y_cp, Z_cp = get_bispline_function_control_points(tx_spl, ty_spl, spl.get_coeffs().reshape(nx_spl+kx+1, ny_spl+ky+1), kx, ky)
    ax.plot_wireframe(X_cp, Y_cp, Z_cp, linewidth=0, color='r', label='bispl control polygon')



ax = fg.add_subplot(132, projection='3d')
ax.scatter(x, y, z, label='Data set')

Z_pbispl = pbispl(X, Y, grid=False)
ax.plot_surface(X, Y, Z_pbispl, linewidth=0, color='g', label='spl')
if False:
    X_cp, Y_cp, Z_cp = get_bispline_function_control_points(tx_pspl, ty_pspl, pbispl.get_coeffs().reshape(nx+kx+1, ny+ky+1), kx, ky)
    ax.plot_wireframe(X_cp, Y_cp, Z_cp, linewidth=0, color='g', label='pbispl control polygon')


ax = fg.add_subplot(133, projection='3d')
ax.plot_surface(X, Y, Z-Z_spl, linewidth=0, color='r', label='Error bispl')
ax.plot_surface(X, Y, Z-Z_pbispl, linewidth=0, color='g', label='Error p-bispl')

plt.show()

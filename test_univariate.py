

import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from psplines import get_uniform_knot_vector, optimization_expression, optimization_OOP, get_function_control_points

# Clamped LSQ Univariate example
# Given a set of noisy data {(x_i, f(x_i))}, our aim is to approximate the function f: [a,b] -> R
# by means of uniform cubic splines. However, we have some restrictions on the approximating
# spline. First, we would like to be able to penalize the wiggling of the approximation. This
# is adressed by penalyzed spline approximation (P-splines) by adding a regulatizing term on
# the optimization expression that penalizes the 2-derivative (curvature) of the function.
# Additionally, if we have some knowledge about how te function behaves we would like to ad that we know f(a) = p, f'(a) = p' and f(b) = q, f'(b) = q' for some known
# p, q \in R.

# Let's play with the some functions on the interval [a, b]. We start by making some
# artificially noisy data {(x_i, log(x_i)+eps)}. However, let us leave the first and last pairs of
# points clean, so that we have an approximation of f(a), f'(a), f(b) and f'(b) (using finite
# differences for the derivatives).

N    = 201  #Number of samples
eps  = 1e-1 #Noise magnitud
k    = 3    #Spline degree

########To play with logarithm leave these lines uncomented
a, b = 0.05, 5
f  = np.log
fd = lambda x: 1/x

########To play with sinus uncomment these lines
#a, b = 0, 6
#f  = np.sin
#fd = np.cos

x = np.linspace(a, b, N)
y = f(x)
y[2:-2] += np.random.normal(loc=0.0, scale=eps, size=N-4)

n = 10 #Numer knots
#The standard scipy least squares approximation
t_spl = get_uniform_knot_vector(a, b, n, mode='complete', k=k)
spl = make_lsq_spline(x=x, y=y, t=t_spl, k=k)

#However, with this approximation, we are not imposing any of the previous requirements:
print('-'*30)
print(f"f(a) - spl(a)  = {f(a)  - spl(a)}")
print(f"f'(a)- spl'(a) = {fd(a) - spl(a, nu=1)}")
print(f"f(b) - spl(b)  = {f(b)  - spl(b)}")
print(f"f'(b)- spl'(b) = {fd(b) - spl(b, nu=1)}")
print('-'*30)


#Now let us compute step by step the optimization using scipy optimize subpackage.
n = 30 #Since high frequency is going to be penalized, let us provide more degrees of freedom.
l = 1e1 #Penalty to second derivative
t_pspl = get_uniform_knot_vector(a, b, n, mode='complete', k=k)
Bx = BSpline.design_matrix(x=x, t=t_pspl, k=k).toarray() #The evaluation of the Basis splines at the domain.
n_coeff = n+(k+1)
x0 = np.array([y.mean()]*n_coeff)

cons = [
    {'type':'eq', 'fun': lambda c: BSpline.construct_fast(t_pspl, c, k)(a) - f(a)},
    {'type':'eq', 'fun': lambda c: BSpline.construct_fast(t_pspl, c, k)(b) - f(b)},
    {'type':'eq', 'fun': lambda c: BSpline.construct_fast(t_pspl, c, k)(a, nu=1) - fd(a)},
    {'type':'eq', 'fun': lambda c: BSpline.construct_fast(t_pspl, c, k)(b, nu=1) - fd(b)}
]

#res = minimize(fun=optimization_expression, x0=x0, args=(y, Bx, 0.0),
res = minimize(fun=optimization_OOP, x0=x0, args=(x, y, t_pspl, k, l),
#                    method='L-BFGS-B', constraints=cons)
                    method='SLSQP', constraints=cons)

pspl = BSpline(t=t_pspl, c=res.x, k=k)
print('-'*30)
print(f"f(a) - pspl(a)  = {f(a)  - pspl(a)}")
print(f"f'(a)- pspl'(a) = {fd(a) - pspl(a, nu=1)}")
print(f"f(b) - pspl(b)  = {f(b)  - pspl(b)}")
print(f"f'(b)- pspl'(b) = {fd(b) - pspl(b, nu=1)}")
print('-'*30)


fg, ax = plt.subplots(3, 1)
ax[0].scatter(x, y, label='Data set')
ax[0].plot(x, f(x), 'k--', label='f')
spl_cp = get_function_control_points(knots=t_spl, coeff=spl.c, k=3)
ax[0].plot(x, spl(x), 'r', label='spl')
ax[0].plot(spl_cp[0], spl_cp[1], 'ro', linestyle='--', mfc='none', label='spl control polygon')
ax[0].legend()


ax[1].scatter(x, y, label='Data set')
ax[1].plot(x, f(x), 'k--', label='f')
pspl_cp = get_function_control_points(knots=t_pspl, coeff=pspl.c, k=3)
ax[1].plot(x, pspl(x), 'g', label='P-spline')
ax[1].plot(pspl_cp[0], pspl_cp[1], 'go', linestyle='--', mfc='none', label='pspl control polygon')
ax[1].legend()

ax[2].plot(x, f(x), 'k--', label='f')
ax[2].plot(x, spl(x), 'r', label='spl')
ax[2].plot(x, pspl(x), 'g', label='P-spline')
ax[2].legend()
plt.show()

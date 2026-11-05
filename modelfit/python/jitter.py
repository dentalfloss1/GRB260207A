from scipy.optimize import brentq
import numpy as np 


def chi2_red(jitter, f_i, stat_i, dof):
    w_i = (1 / (stat_i**2 + jitter**2))
    f_avg = np.average(f_i, weights=w_i)
    return (1/dof)*np.sum( (f_i - f_avg)**2 / (stat_i**2 + jitter**2))

def fit_jitter(ydata, yerr):
    dof = len(ydata) - 1 
    g = lambda j: 1 - chi2_red(j, ydata, yerr, dof)
    b = 5*np.std(ydata)
    while g(0)*g(b) > 0:
       b *= 2
    return brentq(g, 0, b)

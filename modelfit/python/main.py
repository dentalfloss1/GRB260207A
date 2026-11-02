import numpy as np 
import emcee
import matplotlib.pyplot as plt
from astropy.time import Time
import corner
import lcmodel

CADENCE = 200.0/60.0/60.0/24.0
TRIGGER = Time("2026-02-07 05:42:33.65").jd - 2457000
FORWARDSTOP = 0.02 
k = 0

PRIOR_FS = {
    'logF0':   (1.0, 5.0),
    'logTb':   (-2.523, -1.699),  # 3–30 min, matching the original P1 range
    'p':   (1.5, 3),
}

theta0_FS = np.array([np.log10(1e3), np.log10(0.01), 2.2])



def log_likelihood(theta, x, y, yerr):
    params = theta
    # model = sbpl(x + CADENCE/2.0, *params)
    model = lcmodel.forwardshock(x, k, *params)
    sigma2 = yerr**2 
    return -0.5 * np.sum( (y-model)**2 / sigma2 )

def log_prior(theta):
    for v, (low, hi) in zip(theta, PRIOR_FS.values()):
        if not (low <= v <= hi):
            return -np.inf
    return 0 

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

if __name__=="__main__":
    data = np.loadtxt("lc_GRB260207A_cand41148_geo")
    tbins = data[:,0] - TRIGGER
    tends = tbins + CADENCE
    subtbins = tbins[(tends > 0) & (tbins < FORWARDSTOP)]
    ydata = -data[:,1][(tends >0) & (tbins < FORWARDSTOP)]
    yerr = data[:,2][(tends > 0) & (tbins < FORWARDSTOP)]
        
    ndim = len(theta0_FS)
    nwalkers = 32
    pos = theta0_FS + 1e-3 * np.random.rand(nwalkers, ndim)
    for i in range(nwalkers):
        for j, (lo, hi) in enumerate(PRIOR_FS.values()):
            pos[i,j] = np.clip(pos[i,j], lo+1e-4, hi - 1e-4)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(subtbins, ydata,yerr))
    nburn = 1000
    state = sampler.run_mcmc(pos, nburn, progress = True)
    sampler.reset()
    sampler.run_mcmc(state, 10_000, progress=True)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=list(PRIOR_FS.keys()))
    plt.savefig("corner.png")
    plt.close()
    fig = plt.figure()
    plt.errorbar(subtbins,ydata,yerr=yerr,ls='none')
    inds = np.random.randint(len(flat_samples),size=100)
    x0 = np.geomspace(1e-4, subtbins.max(),1000)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(x0, lcmodel.forwardshock(x0, k, *sample), "C1", alpha=0.1)

    # plt.plot(x0, lcmodel.forwardshock(x0, k, *theta0_FS), color='black',ls=':')
    
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig("lightcurve.png")
    plt.close()



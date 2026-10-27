import numpy as np 
import emcee
import matplotlib.pyplot as plt
from astropy.time import Time
import corner

CADENCE = 200.0/60.0/60.0/24.0
TRIGGER = Time("2026-02-07 05:42:33.65").jd - 2457000
FORWARDSTOP = 0.02 

PRIOR_FS = {
    'logF0':   (1.0, 5.0),
    'logTb':   (-2.523, -1.699),  # 3–30 min, matching the original P1 range
    'k':   (0, 2.0),      
    'p':   (1.5, 3),
}

theta0_FS = np.array([np.log10(1e3), np.log10(0.01), 0.1,2.2])


def sbpl(t, logF0, logtb, k, p):
    F0 = 10**logF0
    tb = 10**logtb
    S = 0.02
    a1=-(2-k)/(4-k); 
    a2=(12*p-12-3*p*k+5*k)/(4*(4-k));
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def log_likelihood(theta, x, y, yerr):
    params = theta
    model = sbpl(x + CADENCE/2.0, *params)
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
    subtbins = tbins[(tbins > 0) & (tbins < FORWARDSTOP)]
    ydata = -data[:,1][(tbins >0) & (tbins < FORWARDSTOP)]
    yerr = data[:,2][(tbins > 0) & (tbins < FORWARDSTOP)]
        
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
    plt.show()
    plt.close()
    fig = plt.figure()
    plt.errorbar(subtbins,ydata,yerr=yerr,ls='none')
    inds = np.random.randint(len(flat_samples),size=100)
    x0 = np.geomspace(subtbins.min(), subtbins.max(),1000)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(x0, sbpl(x0, *sample), "C1", alpha=0.1)

    plt.plot(x0, sbpl(x0, *theta0_FS), color='black',ls=':')
    
    ax = plt.gca()
    ax.axvline(0.01)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    plt.close()



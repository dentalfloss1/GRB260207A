import numpy as np 
import emcee
import matplotlib.pyplot as plt
from astropy.time import Time
import corner
import lcmodel

CADENCE = 200.0/60.0/60.0/24.0
TRIGGER = Time("2026-02-07 05:42:33.65").jd - 2457000
FORWARDSTOP = 0.02 
K = 0
# cadence in 200 seconds, 99% efficiency, 80% retained after cosmic ray mitigation
TOT_CTS_TO_JY = (1/(200*0.99*0.8))*(2416*10**(-8.176))

PRIOR_FS = {
    'logF0':   (1.0, 5.0),
    'logTb':   (-2.523, -1.699),  # 3–30 min, matching the original P1 range
    'p':   (1.5, 3),
}

theta0_FS = np.array([np.log10(1e3), np.log10(0.01), 2.2])

def fs_model(t, theta):
    return lcmodel.forwardshock(t, K, *theta)

class Fit:
    def __init__(self, names, priors, model, t, y, yerr):
        self.names = names
        self.priors = priors
        self.model = model
        self.t = t
        self.y = y 
        self.yerr = yerr
    def log_likelihood(self, theta):
        params = theta
        # model = sbpl(x + CADENCE/2.0, *params)
        model = self.model(self.t, theta)
        sigma2 = self.yerr**2 
        return -0.5 * np.sum( (self.y-model)**2 / sigma2 )
    def log_prior(self,theta):
        for v, (low, hi) in zip(theta, self.priors.values()):
            if not (low <= v <= hi):
                return -np.inf
        return 0 
    def log_probability(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    def run(self, theta0, nwalkers=32, nburn=1000, nsteps=10_000):
        self.theta0 = theta0
        ndim = len(theta0)
        pos = theta0 + 1e-3 * np.random.rand(nwalkers, ndim)
        for i in range(nwalkers):
            for j, (lo, hi) in enumerate(self.priors.values()):
                pos[i,j] = np.clip(pos[i,j], lo+1e-4, hi - 1e-4)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        state = self.sampler.run_mcmc(pos, nburn, progress = True)
        self.sampler.reset()
        self.sampler.run_mcmc(state, nsteps, progress=True)
        return 0
    def samples(self, discard=100, thin=15):
        flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        return flat_samples
    def bestfit(self):
        samples = self.samples()
        return samples[np.argmax([self.log_probability(s) for s in samples])]
    def summary(self):
        bfparams = self.bestfit()
        for i, name in enumerate(self.names):
            print(name, bfparams[i])
        return 0 
         
    

if __name__=="__main__":
    data = np.loadtxt("lc_GRB260207A_cand41148_geo")
    tbins = data[:,0] - TRIGGER
    tends = tbins + CADENCE
    subtbins = tbins[(tends > 0) & (tbins < FORWARDSTOP)]
    ydata = -data[:,1][(tends >0) & (tbins < FORWARDSTOP)]
    yerr = data[:,2][(tends > 0) & (tbins < FORWARDSTOP)]
    fsfitter = Fit(list(PRIOR_FS.keys()), PRIOR_FS, fs_model, subtbins, ydata, yerr)
    fsfitter.run(theta0_FS)
    flat_samples = fsfitter.samples()
    bestfit = fsfitter.bestfit()
    fig = corner.corner(flat_samples, labels=list(PRIOR_FS.keys()))
    plt.savefig("corner.png")
    plt.close()
    fig = plt.figure()
    plt.errorbar(subtbins+CADENCE/2,ydata*TOT_CTS_TO_JY,xerr=CADENCE/2,yerr=yerr*TOT_CTS_TO_JY,ls='none')
    plt.errorbar(subtbins+CADENCE/2,fs_model(subtbins+CADENCE/2,bestfit)*TOT_CTS_TO_JY,xerr=CADENCE/2,ls='none',marker='s',color='red')
    inds = np.random.randint(len(flat_samples),size=100)
    x0 = np.geomspace(1e-4, subtbins.max(),1000)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(x0, fs_model(x0,sample)*TOT_CTS_TO_JY, "C1", alpha=0.1)

    
    ax = plt.gca()
    ax.set_xscale('symlog', linthresh=2e-3)
    ax.set_yscale('log')
    plt.savefig("lightcurve.png")
    plt.close()



import numpy as np
import emcee
from astropy.time import Time
import lcmodel
from jitter import fit_jitter
from plotting import save_corner, save_lightcurve
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--quick",action="store_true")
args = parser.parse_args()

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

PRIOR_INJ = {
    'logF0rev': (np.log10(1), np.log10(1e5)),
    'logdeltatbrev': (np.log10(0.005),np.log10(0.06)),
    'a1': (-4, 0),
    'logtinj': (np.log10(0.015), np.log10(0.03)),
    'logdeltatinj': (np.log10(0.001), np.log10(0.02)),
    'eta': (0,3-K),
}

# Cosmetic only -- corner-plot display labels, keyed by the same names as
# PRIOR_FS/PRIOR_INJ above but kept separate so plotting never touches fitting.
LATEX_LABELS = {
    'logF0': r'$\log_{10} F_0$',
    'logTb': r'$\log_{10} t_b$',
    'p': r'$p$',
    'logF0rev': r'$\log_{10} F_{0,\rm rev}$',
    'logdeltatbrev': r'$\log_{10} \Delta t_{b,\rm rev}$',
    'a1': r'$\alpha_1$',
    'logtinj': r'$\log_{10} t_{\rm inj}$',
    'logdeltatinj': r'$\log_{10} \Delta t_{\rm inj}$',
    'eta': r'$\eta$',
}

theta0_FS = np.array([np.log10(1e2), np.log10(0.01), 2.2])
theta0_INJ = np.array([np.log10(1e3), np.log10(0.02), -1.5,
                       np.log10(0.02),np.log10(0.01), 2])

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
         
def get_preburst_fluxoff(f_i, stat_i, jitter):
    w_i = (1 / (stat_i**2 + jitter**2))
    return np.average(f_i, weights=w_i)
   
def print_red_chi2(data, model, err, dof):
    chi2_r = (1/dof) * np.sum((data - model)**2 / (err**2))
    print("Red. Chi2:", chi2_r)
    return chi2_r

if __name__=="__main__":
    data = np.loadtxt("lc_GRB260207A_cand41148_geo")
    tbins = data[:,0] - TRIGGER
    tends = tbins + CADENCE

    # Pre-trigger, get jitter 
    subtbins = tbins[(tbins > -0.5) & (tends < 0)]
    ydata = -data[:,1][(tbins >-0.5) & (tends < 0 )]
    yerr = data[:,2][(tbins > -0.5) & (tends < 0)]
    
    jitter = fit_jitter(ydata, yerr)
    flux_offset = get_preburst_fluxoff(ydata, yerr, jitter)

    # All post-trigger data, for plotting
    t_post = tbins[tends > 0]
    y_post = -data[:,1][tends > 0] - flux_offset
    yerr_post = np.sqrt(data[:,2][tends > 0]**2 + jitter**2)

    # forward shock
    subtbins = tbins[(tends > 0) & (tbins < FORWARDSTOP)]
    ydata = -data[:,1][(tends >0) & (tbins < FORWARDSTOP)] - flux_offset
    yerr = np.sqrt(data[:,2][(tends > 0) & (tbins < FORWARDSTOP)]**2 + jitter**2)
    fsfitter = Fit(list(PRIOR_FS.keys()), PRIOR_FS, fs_model, subtbins, ydata, yerr)
    fsfitter.run(theta0_FS)
    flat_samples = fsfitter.samples()
    bestfit = fsfitter.bestfit()
    
    chi2_r = print_red_chi2(ydata, fs_model(subtbins, bestfit), yerr, len(ydata)-len(bestfit))

    save_corner("FScorner.png", flat_samples, [LATEX_LABELS[n] for n in fsfitter.names])
    save_lightcurve("FSlightcurve.png", t_post, y_post, yerr_post, t_post < FORWARDSTOP,
                    fs_model, bestfit, flat_samples, chi2_r,
                    title='Forward shock only', color='royalblue',
                    cadence=CADENCE, to_jy=TOT_CTS_TO_JY)

    # full evolution
    bestfwd = bestfit
    def inj_model(t, theta):
        logF0, logtb, p = bestfwd
        return lcmodel.injection(t, K, logF0, logtb, p,  *theta)
    subtbins = tbins[(tends > 0) & (tbins < 1)]
    ydata = -data[:,1][(tends >0) & (tbins < 1)] - flux_offset
    yerr = np.sqrt(data[:,2][(tends > 0) & (tbins < 1)]**2 + jitter**2)
    injfitter = Fit(list(PRIOR_INJ.keys()), PRIOR_INJ, inj_model, subtbins, ydata, yerr)
    if args.quick:
        injfitter.run(theta0_INJ,nburn=500, nsteps=1000)
    else:
        injfitter.run(theta0_INJ)
    flat_samples = injfitter.samples()
    bestfit = injfitter.bestfit()
    
    chi2_r = print_red_chi2(ydata, inj_model(subtbins, bestfit), yerr, len(ydata)-len(bestfit))

    # A log flux of -50 switches a component off
    OFF = -50.0
    components = [
        ('Unrefreshed FS', lambda t: fs_model(t, bestfwd), '--', 'dimgray'),
        ('Refreshed FS', lambda t: inj_model(t, [OFF, *bestfit[1:]]), '--', 'darkorange'),
        ('Reverse shock', lambda t: lcmodel.injection(t, K, OFF, *bestfwd[1:], *bestfit), ':', 'crimson'),
    ]
    tinj = 10**bestfit[injfitter.names.index('logtinj')]

    save_corner("injcorner.png", flat_samples, [LATEX_LABELS[n] for n in injfitter.names])
    save_lightcurve("injlightcurve.png", t_post, y_post, yerr_post, t_post < 1,
                    inj_model, bestfit, flat_samples, chi2_r,
                    title='Forward shock with energy injection and reverse shock', color='navy',
                    cadence=CADENCE, to_jy=TOT_CTS_TO_JY,
                    components=components, vlines=[(tinj, r'$t_{\rm inj}$')])

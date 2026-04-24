import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import math
from datetime import datetime, timezone
from astropy.time import Time

trigger = Time("2026-02-07 05:40:16.947").jd-2457000
trigger_btjd = Time("2026-02-07 05:47:42.796").jd-2457000

def ctstomag(cts):
    return -2.5*np.log10(cts) + 20.44

def magtocts(mag):
    return 10**((20.44 - mag) / 2.5)
df = pd.read_csv('lc_GRB260207A_cand47734_cleaned', comment='#', sep=r'\s+',
    names=['BTJD','TJD','cts_per_s','e_cts_per_s','mag','e_mag',
           'bkg','bkg_model','bkg2','e_bkg2'])
rawdata = np.genfromtxt("lc_GRB260207A_cand41148")
rezero = np.median(df['cts_per_s'][((df['TJD'] - trigger) <(-2/24)) & ((df['TJD'] - trigger) > (-1))])


fig,ax = plt.subplots(nrows=2,ncols=2, figsize=(15,10))
masterT = Time("2026-02-07 05:44:56").jd - Time("2026-02-07 05:40:16.947").jd
masterm = magtocts(17.3)
ax[0,0].scatter(df['TJD']-trigger,df['cts_per_s']-rezero)
ax[0,0].scatter(masterT,masterm,label="MASTER",marker='s')
ax[0,0].set_title("TJD")
ax[1,0].scatter(df['BTJD']-trigger_btjd,df['cts_per_s']-rezero)
ax[1,0].scatter(masterT,masterm,label="MASTER",marker='s')
ax[1,0].set_title("BTJD")
ax[0,1].scatter(rawdata[:,0]-trigger,-rawdata[:,1]/(200*0.8*0.99))
ax[0,1].scatter(masterT,masterm,label="MASTER",marker='s')
ax[0,1].set_title("TJD")
print(df['BTJD']-trigger_btjd)
for a in ax.flatten():
#     a.set_ylim(0,)
    a.set_xlim(-0.1,0.1)
    a.axvline(0)
    # a.set_xscale('log')
    # a.invert_yaxis()
    # a.set_ylim(20,17)

plt.legend()
plt.show()
plt.close()

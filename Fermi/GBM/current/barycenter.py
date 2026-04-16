#!/usr/bin/env python3

import numpy as np
import glob
import re
import argparse

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.constants import c
from astropy.time import Time


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Simple GBM barycenter correction")
    parser.add_argument("--ra", type=float, required=True, help="RA (deg)")
    parser.add_argument("--dec", type=float, required=True, help="Dec (deg)")
    parser.add_argument("--outfile", default="bary_times.npy")
    return parser.parse_args()


# -----------------------------
# Find files
# -----------------------------
def find_tte():
    files = glob.glob("glg_tte_*_bn*.fit")
    if not files:
        raise FileNotFoundError("No TTE file found")
    return sorted(files)[0]


def extract_bn(fname):
    m = re.search(r'bn(\d+)', fname)
    return m.group(1)


def find_poshist(bn):
    yymmdd = bn[:6]
    files = glob.glob(f"glg_poshist_all_{yymmdd}_v*.fit")
    if not files:
        raise FileNotFoundError("No poshist file found")
    return sorted(files)[-1]


# -----------------------------
# Load data
# -----------------------------
def load_tte(tte_file):
    with fits.open(tte_file) as f:
        times = f["EVENTS"].data["TIME"]
        trigtime = f[0].header["TRIGTIME"]
    return times, trigtime


def load_poshist(poshist_file):
    with fits.open(poshist_file) as f:
        data = f[1].data
        times = data["SCLK_UTC"]
        x = data["POS_X"]
        y = data["POS_Y"]
        z = data["POS_Z"]
    return times, np.vstack([x, y, z]).T


# -----------------------------
# Main
# -----------------------------
def main():

    args = parse_args()

    tte_file = find_tte()
    bn = extract_bn(tte_file)
    poshist_file = find_poshist(bn)

    print("TTE:", tte_file)
    print("POSHIST:", poshist_file)

    # Load
    tte_times, trigtime = load_tte(tte_file)
    ph_times, ph_pos = load_poshist(poshist_file)

    print("Events:", len(tte_times))
    print("Trigger time (MET):", trigtime)

    # Source direction
    src = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="icrs")
    n_hat = src.cartesian.xyz.value

    # -----------------------------
    # Interpolate spacecraft pos for events
    # -----------------------------
    print("Interpolating spacecraft position...")

    sc_x = np.interp(tte_times, ph_times, ph_pos[:, 0])
    sc_y = np.interp(tte_times, ph_times, ph_pos[:, 1])
    sc_z = np.interp(tte_times, ph_times, ph_pos[:, 2])

    sc_pos = np.vstack([sc_x, sc_y, sc_z]).T * u.km

    # Compute correction
    dot = sc_pos.value @ n_hat
    dt = (dot * u.km / c).to(u.s).value

    bary_times = tte_times + dt

    # Save
    np.save(args.outfile, bary_times)

    print("\nSaved:", args.outfile)
    print("Correction range (s):", dt.min(), dt.max())

    # -----------------------------
    # 🔥 Barycenter trigger time
    # -----------------------------
    print("\nComputing barycenter-corrected trigger time...")

    # Interpolate spacecraft position at TRIGTIME
    sc_trig_x = np.interp(trigtime, ph_times, ph_pos[:, 0])
    sc_trig_y = np.interp(trigtime, ph_times, ph_pos[:, 1])
    sc_trig_z = np.interp(trigtime, ph_times, ph_pos[:, 2])

    sc_trig = np.array([sc_trig_x, sc_trig_y, sc_trig_z]) * u.km

    # Compute delay
    dot_trig = sc_trig.value @ n_hat
    dt_trig = (dot_trig * u.km / c).to(u.s).value

    bary_trig = trigtime + dt_trig

    print("Trigger correction (s):", dt_trig)
    print("Barycentered trigger (MET):", bary_trig)

    # Convert to UTC
    t0 = Time("2001-01-01T00:00:00", scale="utc")
    t_utc = t0 + bary_trig * u.s
    print("Barycentered trigger (UTC):", t_utc.iso)


if __name__ == "__main__":
    main()


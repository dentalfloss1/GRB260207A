#!/usr/bin/env python3

import numpy as np
import glob
import re
import argparse

from astropy.io import fits
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
import astropy.units as u
from astropy.constants import c
from astropy.time import Time


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="SSB barycenter correction for GBM TTE data"
    )
    parser.add_argument("--ra", type=float, required=True, help="RA (deg)")
    parser.add_argument("--dec", type=float, required=True, help="Dec (deg)")
    parser.add_argument("--outfile", default="bary_times.npy")
    return parser.parse_args()


# -----------------------------
# File helpers
# -----------------------------
def find_tte():
    files = sorted(glob.glob("glg_tte_*_bn*.fit"))
    if not files:
        raise FileNotFoundError("No TTE file found")
    return files[0]


def extract_bn(fname):
    m = re.search(r'bn(\d+)', fname)
    return m.group(1)


def find_poshist(bn):
    yymmdd = bn[:6]
    files = sorted(glob.glob(f"glg_poshist_all_{yymmdd}_v*.fit"))
    if not files:
        raise FileNotFoundError("No poshist file found")
    return files[-1]


# -----------------------------
# Load data
# -----------------------------
def load_tte(tte_file):
    with fits.open(tte_file) as f:
        times = f["EVENTS"].data["TIME"] * u.s
        trigtime = f[0].header["TRIGTIME"] * u.s
    return times, trigtime


def load_poshist(poshist_file):
    with fits.open(poshist_file) as f:
        data = f[1].data
        times = data["SCLK_UTC"] * u.s
        pos = np.vstack([
            data["POS_X"],
            data["POS_Y"],
            data["POS_Z"]
        ]).T * u.km
    return times, pos


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
    print("Trigger time (MET):", trigtime.value)

    # Reference epoch (Fermi MET)
    t0 = Time("2001-01-01T00:00:00", scale="utc")

    # Convert to TDB (important!)
    t_astropy = (t0 + tte_times).tdb

    # Source direction
    src = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="icrs")
    n_hat = src.cartesian.xyz.value  # unit vector

    # -----------------------------
    # Spacecraft position interpolation
    # -----------------------------
    print("Interpolating spacecraft position...")

    sc_x = np.interp(tte_times.value, ph_times.value, ph_pos[:, 0].value) * u.km
    sc_y = np.interp(tte_times.value, ph_times.value, ph_pos[:, 1].value) * u.km
    sc_z = np.interp(tte_times.value, ph_times.value, ph_pos[:, 2].value) * u.km

    sc_pos = u.Quantity(np.vstack([sc_x, sc_y, sc_z]).T)

    # -----------------------------
    # Earth → SSB position
    # -----------------------------
    earth_pos, _ = get_body_barycentric_posvel("earth", t_astropy)
    earth_pos = earth_pos.xyz.to(u.km).T

    # Total position relative to SSB
    r_ssb = earth_pos + sc_pos

    # -----------------------------
    # Barycentric correction
    # -----------------------------
    dot = np.sum(r_ssb * n_hat, axis=1)
    dt = (dot / c).to(u.s)

    bary_times = (tte_times + dt).to(u.s).value

    # Save
    np.save(args.outfile, bary_times)

    print("\nSaved:", args.outfile)
    print("Correction range (s):", dt.min().value, dt.max().value)

    # -----------------------------
    # 🔥 Barycenter trigger time
    # -----------------------------
    print("\nComputing barycentered trigger time...")

    t_trig = (t0 + trigtime).tdb

    # Spacecraft position at trigger
    sc_trig = u.Quantity([
        np.interp(trigtime.value, ph_times.value, ph_pos[:, 0].value),
        np.interp(trigtime.value, ph_times.value, ph_pos[:, 1].value),
        np.interp(trigtime.value, ph_times.value, ph_pos[:, 2].value),
    ], u.km)

    # Earth position at trigger
    earth_trig, _ = get_body_barycentric_posvel("earth", t_trig)
    earth_trig = earth_trig.xyz.to(u.km)

    # Total position
    r_trig_ssb = earth_trig + sc_trig

    # Dot product (explicit)
    dot_trig = (
        r_trig_ssb[0] * n_hat[0] +
        r_trig_ssb[1] * n_hat[1] +
        r_trig_ssb[2] * n_hat[2]
    )

    dt_trig = (dot_trig / c).to(u.s)
    bary_trig = (trigtime + dt_trig).to(u.s)

    print("Trigger correction (s):", dt_trig.value)
    print("Barycentered trigger (MET):", bary_trig.value)

    # Convert to UTC
    t_utc = (t0 + bary_trig).utc
    print("Barycentered trigger (UTC):", t_utc.iso)


if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
from pathlib import Path

h       = 0.6777
Omega_m = 0.315
Omega_L = 1.0 - Omega_m
c_km_s  = 299792.458
H0      = 100.0 * h

def Ez(z):
    return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)

def cumtrapz_np(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    dy = (y[:-1] + y[1:]) * 0.5 * (x[1:] - x[:-1])
    out = np.zeros(len(x))
    out[1:] = np.cumsum(dy)
    return out

def comoving_distance_Mpc(z):
    z = np.asarray(z)
    z_grid = np.linspace(0.0, z.max(), 2000)
    integrand = 1.0 / Ez(z_grid)
    chi_grid = (c_km_s / H0) * cumtrapz_np(integrand, z_grid)
    chi = np.interp(z, z_grid, chi_grid)
    return chi

def dVdz_dOmega_Mpc3(z):
    z = np.asarray(z)
    chi = comoving_distance_Mpc(z)
    Hz  = H0 * Ez(z)
    return (c_km_s / Hz) * chi**2

deg2_per_sr = (180.0 / np.pi)**2

def load_desi_df(DATA_DIR):
    return pd.read_csv(
        Path(DATA_DIR) / "main-800coaddefftime1200-nz-zenodo.ecsv",
        comment="#",
        sep="\s+"
    )

def load_z(DATA_DIR):
    df = load_desi_df(DATA_DIR)
    return 0.5 * (df["ZMIN"] + df["ZMAX"])

def ang_to_vol_density(colname, DATA_DIR):
    df = load_desi_df(DATA_DIR)

    N_bin_deg2 = df[colname].to_numpy()
    N_bin_sr   = N_bin_deg2 * deg2_per_sr

    n_vol = np.zeros(len(df))
    for i in range(len(df)):
        zlow, zhigh = df["ZMIN"][i], df["ZMAX"][i]
        z = np.linspace(zlow, zhigh, 200)
        dVdz = dVdz_dOmega_Mpc3(z)
        V_bin_sr = np.trapz(dVdz, z)

        V_bin_sr *= h**3
        n_vol[i] = N_bin_sr[i] / V_bin_sr

    return n_vol
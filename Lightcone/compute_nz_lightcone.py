import numpy as np
import h5py
import csv
from pathlib import Path
from colossus.cosmology import cosmology
from gfib_forward_model import halomass_to_gfib, make_DA_lookup_2d

cosmology.setCosmology('planck18')

# Path settings (HPC environment)
# Update this path depending on your local environment.
BASE = Path("****")# Parh settings

SNAPS = range(21, 49)
survey_area = [1048.346611175581]#最初複数のareaで試していたものの名残

dz_bin = 0.05
z_min_all = 0.0
z_max_all = 2.5

z_edges = np.arange(z_min_all, z_max_all + dz_bin, dz_bin)
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

N_LOP_bin = np.zeros(len(z_centers), dtype=int)
N_VLO_bin = np.zeros(len(z_centers), dtype=int)

def load_magnitudes(i):
    H5 = BASE / f"Uchuu_nu2GC_lightcone_snap{i}.hdf5"
    with h5py.File(H5, "r") as f:
        hdr   = f["Header"]
        zsnap = float(hdr.attrs["zsnap"])
        zmin  = float(hdr.attrs["zmin"])
        zmax  = float(hdr.attrs["zmax"])

        g = f["g_mag"][:]
        r = f["r_mag"][:]
        z = f["z_mag"][:]
        zred = f["Redshift"][:]

    return zsnap, zmin, zmax, g, r, z, zred

def load_halomass(i):
    H5 = BASE / f"Uchuu_nu2GC_lightcone_snap{i}.hdf5"
    with h5py.File(H5, "r") as f:
        halomass = f["HaloMass"][:]  
    return halomass

def elg_lower_line(rz):
    """下側の境界線: g-r = 0.5*(r-z) + 0.1"""
    return 0.5 * rz + 0.1

def elg_lop_upper_line(rz):
    """LOP の上側境界線: g-r = -1.2*(r-z) + 1.3"""
    return -1.2 * rz + 1.3

def elg_vlo_upper_line(rz):
    """VLO の上側境界線: g-r = -1.2*(r-z) + 1.6"""
    return -1.2 * rz + 1.6
    
def elg_common_region_mask(rz, gr):
    """
    LOP/VLO 共通の基本領域:
      r-z >= 0.15
      g-r <= 0.5*(r-z) + 0.1
    """
    return (rz >= 0.15) & (gr <= elg_lower_line(rz)) 

def elg_lop_mask(rz, gr):
    """
    LOP の領域:
      共通領域 かつ
      g-r <= -1.2*(r-z) + 1.3
    """
    m = elg_common_region_mask(rz, gr)
    m &= (gr <= elg_lop_upper_line(rz))
    return m   

def elg_vlo_mask_total(rz, gr):
    """
    VLO 全体の領域:
      共通領域 かつ
      g-r >= -1.2*(r-z) + 1.3
      g-r <= -1.2*(r-z) + 1.6
    """
    m = elg_common_region_mask(rz, gr)
    m &= (gr <= elg_vlo_upper_line(rz))
    m &= (gr >= elg_lop_upper_line(rz))
    return m

DA_interp = make_DA_lookup_2d(
    zmin=0.0,
    zmax=2.5,   
    N=2000,
)

for i in SNAPS:
    print(f"======[info] processing snap {i}======")

    zsnap, zmin, zmax, g_all, r_all, z_all, zred_all = load_magnitudes(i)

    halomass_all = load_halomass(i)
    m_g = np.isfinite(g_all) & np.isfinite(halomass_all)

    g_cut = g_all[m_g]
    r_cut = r_all[m_g]
    z_cut = z_all[m_g]
    halo_cut = halomass_all[m_g]  

    
    rz = r_cut - z_cut
    gr = g_cut - r_cut
    m_lop = elg_lop_mask(rz, gr)
    m_vlo = elg_vlo_mask_total(rz, gr)

    g_fib_lop, _ = halomass_to_gfib(
        g_cut[m_lop],
        halo_cut[m_lop],
        zsnap,
        DA_interp,
        rbin_width=10,
        chunk=20000,
    )

    m_lop_fib = np.isfinite(g_fib_lop) & (g_fib_lop <= 24.1)
    N_LOP = np.count_nonzero(m_lop_fib)
    N_LOP_cand = np.count_nonzero(m_lop)   
    frac_LOP_keep = N_LOP / N_LOP_cand if N_LOP_cand > 0 else 0.0

    frac_LOP_cut  = 1.0 - frac_LOP_keep

    z_lop = zred_all[m_g][m_lop][m_lop_fib]
    
    idx_lop = np.floor((z_lop - z_min_all) / dz_bin).astype(int)
    good = (idx_lop >= 0) & (idx_lop < len(N_LOP_bin))
    
    for k in idx_lop[good]:
        N_LOP_bin[k] += 1

    g_fib_vlo, _ = halomass_to_gfib(
        g_cut[m_vlo],
        halo_cut[m_vlo],
        zsnap,
        DA_interp,
        rbin_width=10,
        chunk=20000,
    )

    m_vlo_fib = np.isfinite(g_fib_vlo) & (g_fib_vlo <= 24.1)
    N_VLO = np.count_nonzero(m_vlo_fib)
    N_VLO_cand = np.count_nonzero(m_vlo)

    frac_VLO_keep = N_VLO / N_VLO_cand if N_VLO_cand > 0 else 0.0

    frac_VLO_cut  = 1.0 - frac_VLO_keep

    z_vlo = zred_all[m_g][m_vlo][m_vlo_fib]

    idx_vlo = np.floor((z_vlo - z_min_all) / dz_bin).astype(int)
    good = (idx_vlo >= 0) & (idx_vlo < len(N_VLO_bin))
    
    for k in idx_vlo[good]:
        N_VLO_bin[k] += 1

    print(f"[check] snap {i}")

    print(f"  LOP candidates        = {N_LOP_cand}")
    print(f"  LOP kept (g_fib<=24.1)= {N_LOP}")
    print(f"    cut fraction        = {frac_LOP_cut*100:.2f} %")
    
    print(f"  VLO candidates        = {N_VLO_cand}")
    print(f"  VLO kept (g_fib<=24.1)= {N_VLO}")
    print(f"    cut fraction        = {frac_VLO_cut*100:.2f} %")


csv_path = "nz_lightcone_gfibcut.csv"

with open(csv_path, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow([
        "area_deg2", "z_center",
        "N_LOP", "N_VLO",
        "n_LOP[deg^-2]", "n_VLO[deg^-2]"
    ])

    for survey_area_deg2 in survey_area:
        n_LOP_deg2 = N_LOP_bin / survey_area_deg2
        n_VLO_deg2 = N_VLO_bin / survey_area_deg2

        for zc, nlop, nvlo, dnlop, dnvlo in zip(
            z_centers, N_LOP_bin, N_VLO_bin,
            n_LOP_deg2, n_VLO_deg2
        ):
            w.writerow([
                survey_area_deg2,
                f"{zc:.3f}",
                nlop,
                nvlo,
                f"{dnlop:.6e}",
                f"{dnvlo:.6e}",
            ])


print(f"[done] wrote {csv_path}")
import numpy as np
import h5py
import csv
from pathlib import Path
from colossus.cosmology import cosmology
from gfib_forward_model import halomass_to_gfib, make_DA_lookup_2d

cosmology.setCosmology('planck18')

# Path settings (HPC environment)
# Update this path depending on your local environment.
BASE = Path("/data/PFS/Uchuu_nu2GC_DESI_v2/box")

SNAPS = range(21, 43)
L_box = 2000.0                                
V_box = L_box**3                               
vol_unit = "(Mpc/h)^3"

def load_magnitudes(i):
    H5 = BASE / f"Uchuu_nu2GC_box_snap{i}.hdf5"

    with h5py.File(H5, "r") as f:
        zsnap = float(f["Header"].attrs["zsnap"])
        g = f["g_mag"][:]
        r = f["r_mag"][:]
        z = f["z_mag"][:]

    return zsnap, g, r, z

def load_halomass(i):
    H5 = BASE / f"Uchuu_nu2GC_box_snap{i}.hdf5"
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

print(f"[info] Using fixed Uchuu box size: L = {L_box} (Mpc/h), V = {V_box:.3e} {vol_unit}")

DA_interp = make_DA_lookup_2d(
    zmin=0.0,
    zmax=2.5,   
    N=2000,
)

rows = []

for i in SNAPS:
    print(f"======[info] processing snap {i}======")

    zsnap, g_all, r_all, z_all = load_magnitudes(i)

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
    frac_LOP_cut = 1.0 - frac_LOP_keep

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

    print(f"[check] snap {i}")

    print(f"  LOP candidates        = {N_LOP_cand}")
    print(f"  LOP kept (g_fib<=24.1)= {N_LOP}")
    print(f"    cut fraction        = {frac_LOP_cut*100:.2f} %")
    
    print(f"  VLO candidates        = {N_VLO_cand}")
    print(f"  VLO kept (g_fib<=24.1)= {N_VLO}")
    print(f"    cut fraction        = {frac_VLO_cut*100:.2f} %")

    nV_LOP = N_LOP / V_box
    nV_VLO = N_VLO / V_box

    rows.append((
    i, zsnap,
    N_LOP + N_VLO, 
    N_LOP, N_VLO,
    N_LOP_cand, N_VLO_cand,
    frac_LOP_keep, frac_LOP_cut,
    frac_VLO_keep, frac_VLO_cut,
    L_box, V_box,
    nV_LOP, nV_VLO
))


rows.sort(key=lambda t: t[1])

csv_path = "nz_box_gfibcut.csv"
with open(csv_path, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow([
        "snap", "z",
        "N_ELG_total",
        "N_ELG_LOP", "N_ELG_VLO",
        "N_LOP_candidate", "N_VLO_candidate",
        "f_cut_LOP",
        "f_cut_VLO",
        "L_box[Mpc/h]", "V_box[(Mpc/h)^3]",
        "nV_LOP[(h/Mpc)^3]", "nV_VLO[(h/Mpc)^3]"
    ])

    for r in rows:
        w.writerow([
            r[0],                 # snap
            f"{r[1]:.6f}",        # z
            r[2],                 # N_ELG_total
            r[3], r[4],           # N_ELG_LOP, N_ELG_VLO
            r[5], r[6],           # N_LOP_candidate, N_VLO_candidate
            f"{r[8]:.6f}",        # f_cut_LOP
            f"{r[10]:.6f}",       # f_cut_VLO
            f"{r[11]:.6f}",       # L_box
            f"{r[12]:.6e}",       # V_box
            f"{r[13]:.6e}",       # nV_LOP
            f"{r[14]:.6e}",       # nV_VLO
        ])


print(f"[done] wrote {csv_path}")

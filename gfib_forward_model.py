import numpy as np
import mpmath as mp
import colossus
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from colossus.halo import mass_so
"""
g_fib を計算する関数（2D 畳み込み積分）。

畳み込み積分は method="fft" または "con2d" により切り替え可能。
batch 計算では rbin_width を指定することで、r_vir を bin 化し、
同一 bin 内では代表半径で1回のみ畳み込み計算を行うことができる。
rbin_width=None とするとbin化を行わず、各銀河ごとに個別計算を行う。
rbin_widthをbin化する場合は10程度を推奨。

角径距離 DA(z) はあらかじめ make_DA_lookup(z_min, z_max, N) により
補間関数 DA_interp を作成し、各計算関数に渡す必要。

データ数が少ない場合は compute_g_fib、
大量データの場合は compute_g_fib_batch / halomass_to_gfib の使用を推奨。

--------------------------------------------------
Input:
    g          : float or array_like
                 総 g 等級
    r_vir      : float or array_like
                 ハローのビリアル半径 [pkpc]
    z          : float or array_like
                 赤方偏移
    DA_interp  : callable
                 make_DA_lookup により作成した角径距離補間関数

    ※ batch 関数では g_arr, rvir_arr, z_arr に対応

--------------------------------------------------
Returns:
    g_fib : float or ndarray
            ファイバー補正後の g 等級
    f     : float or ndarray
            fiber fraction (= Lf / Lt)
    Lf    : float or ndarray
            ファイバー内の光量
    Lt    : float or ndarray
            全光量

--------------------------------------------------
Default settings (変更可能):
    - 銀河光度分布        : Sérsic プロファイル (n = 4)
    - 半光度半径          : r_half = rhalf_factor × R_vir
    - PSF                : ガウシアン (FWHM = 1.0 arcsec)
    - ファイバー半径      : 1.5 arcsec
    - 畳み込み手法        : FFT 畳み込み ("fft")
    - グリッド分解能      : dx = 0.02 arcsec, N = 512

--------------------------------------------------
Notes:
    - FFT 畳み込みは高速だが、境界条件の影響を受ける可能性がある。
      精度確認には method="con2d" との比較を推奨する。
    - rbin_width を変更することで、計算速度と精度のトレードオフを制御できる。
    - model / numerics / conv はすべて上位関数から一貫して指定可能。
"""

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

DEFAULT_MODEL = dict(
    sersic_n       = 4,
    rhalf_factor   = 0.015,   
)

DEFAULT_NUMERICS = dict(
    FWHM_arcsec = 1.0,
    fiber_arcsec = 1.5,
    dx_arcsec    = 0.02,  #ピクセルサイズ
    N_grid       = 512,   #グリッド数N*N
)

DEFAULT_COSMOLOGY = dict(
    Omega_m = 0.315,
    Omega_L = 0.685,
    h       = 0.6777,
)

B_N_CACHE = {}

def rvir_from_halomass(halomass, z, *, mdef="vir"):
    r_vir = mass_so.M_to_R(halomass, z, mdef=mdef)
    return r_vir
    


def rhalf_from_rvir(r_vir, factor):
    r_vir = np.asarray(r_vir)
    return factor * r_vir

def b_n(n):
    if n not in B_N_CACHE:
        def eq(b):       
            return mp.gammainc(2*n, 0, b) / mp.gamma(2*n) - 0.5
        b0 = 2.0*n - 1.0/3.0
        B_N_CACHE[n] = float(mp.findroot(eq, b0))#最初にb(n)を作ったら後は固定
    return B_N_CACHE[n]

def DA_z(z, *, Omega_m, Omega_L, h):
    c = 299792.458  
    H0 = 100.0 * h  

    def Ez(z):
        return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

    z_grid = np.linspace(0, z, 5000)
    Ez_grid = Ez(z_grid)
    chi = np.trapz(1.0/Ez_grid, z_grid) * (c/H0)
    DA = chi / (1+z)
    return DA*1000 #kpc

def make_DA_lookup_2d(zmin=0.0, zmax=1.5, N=1000, *, cosmo=DEFAULT_COSMOLOGY):##DAをテーブル化する処理
    z_table = np.linspace(zmin, zmax, N)
    DA_table = np.array([DA_z(z, **cosmo) for z in z_table])
    DA_interp = interp1d(z_table, DA_table, kind='cubic', fill_value="extrapolate", bounds_error=False)
    return DA_interp

def make_grid(DA, *, N, dx_arcsec):
    dx = dx_arcsec * ARCSEC_TO_RAD * DA  
    x = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, x)
    return X, Y

def psf_2d(X, Y, DA, *, FWHM_arcsec):
    FWHM_rad = FWHM_arcsec * ARCSEC_TO_RAD 
    sigma_theta = FWHM_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_r = sigma_theta * DA  
    PSF = np.exp(-(X**2 + Y**2) / (2.0 * sigma_r**2))
    PSF /= PSF.sum()
    return PSF

def I_sersic_2d(X, Y, r_star, n):
    R = np.sqrt(X**2 + Y**2)
    return np.exp(-(R / r_star)**(1.0 / n))


def convolve_I_2d(X, Y, r_star, n, DA, *, FWHM_arcsec, method="fft"):
    I = I_sersic_2d(X, Y, r_star, n)
    PSF = psf_2d(X, Y, DA, FWHM_arcsec=FWHM_arcsec) 
    if method == "fft":     #FFTとconvolve2dの切り替え
        return fftconvolve(I, PSF, mode="same")
    elif method == "con2d":
        return convolve2d(I, PSF, mode="same",boundary="fill", fillvalue=0.0)
    else:
        raise ValueError(f"Unknown conv method: {method}")


def L_fiber_2d(X, Y, I_obs, DA, *, arcsec):
    theta = arcsec * ARCSEC_TO_RAD
    r_fib = theta * DA
    R = np.sqrt(X**2 + Y**2)
    mask = R <= r_fib
    return I_obs[mask].sum()


def L_total_2d(I_obs):
    return I_obs.sum() 

def compute_g_fib_core(
    g, r_vir,
    *,
    DA,
    X, Y,
    model,
    numerics,
    conv="fft",
):

    n = model["sersic_n"]
    rhalf_fac = model["rhalf_factor"]
    FWHM_arcsec = numerics["FWHM_arcsec"]
    arcsec = numerics["fiber_arcsec"]

    r_half = rhalf_from_rvir(r_vir, factor=rhalf_fac)
    b = b_n(n)
    r_star = r_half / (b**n)
    I_obs = convolve_I_2d(X, Y, r_star, n, DA, FWHM_arcsec=FWHM_arcsec, method=conv)
    Lf = L_fiber_2d(X, Y, I_obs, DA, arcsec=arcsec)
    Lt = L_total_2d(I_obs)

    if Lt <= 0:
        return np.inf, 0.0, Lf, Lt#Lt_2d=0やf_2d<0を防ぐ
    f = Lf / Lt
    if (not np.isfinite(f)) or (f <= 0):
        return np.inf, 0.0, Lf, Lt

    g_fib = g - 2.5 * np.log10(f)
    return g_fib, f, Lf, Lt


def compute_g_fib_batch_samez(#zが全部同じ場合(snapごとなど)
    g_arr, rvir_arr, z, DA_interp,
    *,
    model=DEFAULT_MODEL,
    numerics=DEFAULT_NUMERICS,
    conv="fft",
):
    g_arr = np.asarray(g_arr)
    rvir_arr = np.asarray(rvir_arr)
    DA = float(DA_interp(z))
    N = numerics["N_grid"]
    dx_arcsec = numerics["dx_arcsec"]
    X, Y = make_grid(DA, N=N, dx_arcsec=dx_arcsec)

    Ng = g_arr.size
    g_fib = np.empty(Ng)
    frac  = np.empty(Ng)
    Lf    = np.empty(Ng)
    Lt    = np.empty(Ng)

    for i in range(Ng):
        g_fib[i], frac[i], Lf[i], Lt[i] = compute_g_fib_core(
            g_arr[i], rvir_arr[i],
            DA=DA,
            X=X, Y=Y,
            model=model,
            numerics=numerics,
            conv=conv,
        )

    return g_fib, frac, Lf, Lt

def _compute_g_fib_batch_samez_rbin(#r_binで使う用
    g_arr, rvir_arr, z, DA_interp,
    *,
    model,
    numerics,
    conv,
    rbin_width,
):
    g_arr = np.asarray(g_arr)
    rvir_arr = np.asarray(rvir_arr) 
    DA = float(DA_interp(z))
    N = numerics["N_grid"]
    dx_arcsec = numerics["dx_arcsec"]
    X, Y = make_grid(DA, N=N, dx_arcsec=dx_arcsec)

    r_key = (np.floor(rvir_arr / rbin_width) + 0.5) * rbin_width
    uniq_r = np.unique(r_key)

    Ng = g_arr.size
    g_fib = np.empty(Ng)
    frac  = np.empty(Ng)
    Lf    = np.empty(Ng)
    Lt    = np.empty(Ng)

    for r0 in uniq_r:
        idx = (r_key == r0)

        r_rep = float(r0)

        _, f_rep, Lf_rep, Lt_rep = compute_g_fib_core(
            g=0.0,
            r_vir=r_rep,
            DA=DA,
            X=X, Y=Y,
            model=model,
            numerics=numerics,
            conv=conv,
        )

        frac[idx] = f_rep
        Lf[idx]   = Lf_rep
        Lt[idx]   = Lt_rep
        g_fib[idx] = g_arr[idx] - 2.5 * np.log10(f_rep)

    return g_fib, frac, Lf, Lt


    
def compute_g_fib_batch_2d(
    g_arr, rvir_arr, z_arr, DA_interp,
    *,
    model=DEFAULT_MODEL,
    numerics=DEFAULT_NUMERICS,
    conv="fft",
    z_round=8,#何桁まで正確に分けるか
    rbin_width=10,#bin幅を決める。Noneならbinにしない
):

    z_arr = np.asarray(z_arr)
    z_key = np.round(z_arr, z_round)
    uniq = np.unique(z_key)

    if uniq.size == 1:#Zが全て同じ場合
        z0 = float(z_arr.flat[0]) 
        if rbin_width is None:
            return compute_g_fib_batch_samez(
                g_arr, rvir_arr, z0, DA_interp,
                model=model,
                numerics=numerics,
                conv=conv,
            )
        else:
            return _compute_g_fib_batch_samez_rbin(#rbinでの切り替え
                g_arr, rvir_arr, z0, DA_interp,
                model=model,
                numerics=numerics,
                conv=conv,
                rbin_width=rbin_width,
            )
    else:
        g_arr = np.asarray(g_arr)
        rvir_arr = np.asarray(rvir_arr)

        Ng = g_arr.size
        g_fib = np.empty(Ng)
        frac  = np.empty(Ng)
        Lf    = np.empty(Ng)
        Lt    = np.empty(Ng)

        for z0 in uniq:
            idx = (z_key == z0)
            
            if rbin_width is None:
                g_sub, f_sub, Lf_sub, Lt_sub = compute_g_fib_batch_samez(
                    g_arr[idx], rvir_arr[idx], float(z0),
                    DA_interp,
                    model=model,
                    numerics=numerics,
                    conv=conv,
                )
            else:
                g_sub, f_sub, Lf_sub, Lt_sub = _compute_g_fib_batch_samez_rbin(
                    g_arr[idx], rvir_arr[idx], float(z0),
                    DA_interp,
                    model=model,
                    numerics=numerics,
                    conv=conv,
                    rbin_width=rbin_width,
                )

            g_fib[idx] = g_sub
            frac[idx]  = f_sub
            Lf[idx]    = Lf_sub
            Lt[idx]    = Lt_sub

        return g_fib, frac, Lf, Lt

def compute_g_fib_from_halomass_batch_wid(
    gmag, rvir, z, DA_interp,
    *,
    model=DEFAULT_MODEL,
    numerics=DEFAULT_NUMERICS,
    conv="fft",
    rbin_width,
    chunk=20000,
):
    n = gmag.size
    g_fib = np.full(n, np.nan)
    frac  = np.full(n, np.nan)

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sl = slice(i0, i1)
        
        gi = gmag[sl]
        ri = rvir[sl]

        m = np.isfinite(gi) & np.isfinite(ri) & (ri > 0)
        if not np.any(m):
            continue
        idx = np.where(m)[0] + i0

        zi = np.full(np.count_nonzero(m), z)

        gf, fr, _, _ = compute_g_fib_batch_2d(
            gi[m],
            ri[m],
            zi,
            DA_interp,
            model=model,
            numerics=numerics,
            conv=conv,
            rbin_width=rbin_width,
        )


        g_fib[idx] = gf
        frac[idx]  = fr

        if i1 % (5 * chunk) == 0 or i1 == n:
            print(f"[info] batch {i1}/{n}")

    return g_fib, frac

def halomass_to_gfib(
    g_arr,
    halomass,
    z,
    DA_interp,
    *,
    model=DEFAULT_MODEL,
    numerics=DEFAULT_NUMERICS,
    conv="fft",
    rbin_width=None,
    chunk=20000,
):

    r_vir = rvir_from_halomass(halomass, z)

    g_fib, frac = compute_g_fib_from_halomass_batch_wid(
        g_arr,
        r_vir,
        z,
        DA_interp,
        model=model,
        numerics=numerics,
        conv=conv,
        rbin_width=rbin_width,
        chunk=chunk,
    )


    return g_fib, frac

"""Additive decompositions used by decomposition-hybrid forecasters.

Each function maps a 1-D series x to a (K, len(x)) array of components that sum back to x.
Whether a pipeline leaks depends on *what* it is applied to: the full series (leaky) or only the
data observed up to the forecast origin (causal). `atrous_haar` is causal by construction: the
component at time t uses x[0..t] only, so applying it to the full series does not leak.
"""
import numpy as np
import pywt
from PyEMD import CEEMDAN


def dwt_mra(x, wavelet='db4', level=4):
    """Discrete-wavelet multiresolution analysis (1 approximation + `level` details)."""
    return np.vstack(pywt.mra(x, wavelet, level=level, transform='dwt', mode='symmetric'))


def ceemdan(x, max_imf=6, trials=30, seed=0):
    """CEEMDAN intrinsic mode functions plus residue."""
    c = CEEMDAN(trials=trials, parallel=False)
    c.noise_seed(seed)
    c.ceemdan(np.asarray(x, float), max_imf=max_imf)
    imfs, residue = c.get_imfs_and_residue()
    return np.vstack([imfs, residue[None, :]])


def atrous_haar(x, level=4):
    """Causal a trous Haar wavelet (Renaud et al., 2005): details w_1..w_J plus smooth c_J."""
    c_prev = np.asarray(x, float)
    comps = []
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        shifted = np.concatenate([np.full(step, c_prev[0]), c_prev[:-step]])
        c = 0.5 * (c_prev + shifted)
        comps.append(c_prev - c)
        c_prev = c
    comps.append(c_prev)
    return np.vstack(comps)


def _even(x):
    """TQWT and vmdpy need an even length: pad one copy of the OLDEST value (keeps the recent end intact)."""
    x = np.asarray(x, float)
    return (np.concatenate([x[:1], x]), 1) if len(x) % 2 else (x, 0)


def _tqwt_trans(T):
    v = np.arange(1, T + 1) / (T + 1) * np.pi
    return (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2


def _afb(X, N0, N1):
    N = len(X)
    P, T, S = (N - N1) // 2, (N0 + N1 - N) // 2 - 1, (N - N0) // 2
    tr = _tqwt_trans(T)
    V0 = np.zeros(N0, complex)
    V0[0] = X[0]
    V0[1:1 + P] = X[1:1 + P]
    V0[1 + P:1 + P + T] = X[1 + P:1 + P + T] * tr
    V0[N0 - P - T:N0 - P] = X[N - P - T:N - P] * tr[::-1]
    V0[N0 - P:] = X[N - P:]
    V1 = np.zeros(N1, complex)
    V1[1:1 + T] = X[1 + P:1 + P + T] * tr[::-1]
    V1[1 + T:1 + T + S] = X[1 + P + T:1 + P + T + S]
    V1[N1 // 2] = X[N // 2]
    V1[N1 - T - S:N1 - T] = X[N - P - T - S:N - P - T]
    V1[N1 - T:] = X[N - P - T:N - P] * tr
    return V0, V1


def _sfb(V0, V1, N):
    N0, N1 = len(V0), len(V1)
    P, T, S = (N - N1) // 2, (N0 + N1 - N) // 2 - 1, (N - N0) // 2
    tr = _tqwt_trans(T)
    Y = np.zeros(N, complex)
    Y[0] = V0[0]
    Y[1:1 + P] = V0[1:1 + P]
    Y[1 + P:1 + P + T] = V0[1 + P:1 + P + T] * tr + V1[1:1 + T] * tr[::-1]
    Y[1 + P + T:1 + P + T + S] = V1[1 + T:1 + T + S]
    Y[N // 2] = V1[N1 // 2]
    Y[N - P - T - S:N - P - T] = V1[N1 - T - S:N1 - T]
    Y[N - P - T:N - P] = V0[N0 - P - T:N0 - P] * tr[::-1] + V1[N1 - T:] * tr
    Y[N - P:] = V0[N0 - P:]
    return Y


def tqwt(x, Q=2, r=3, J=6):
    """Tunable-Q wavelet transform (Selesnick, 2011) as an additive decomposition: each of the J high-pass
    sub-bands and the final low-pass band is reconstructed on its own, so the J + 1 components sum to x."""
    xe, pad = _even(x)
    N = len(xe)
    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    sizes = [(2 * round(alpha ** j * N / 2), 2 * round(beta * alpha ** (j - 1) * N / 2)) for j in range(1, J + 1)]
    X = np.fft.fft(xe)
    bands = []
    for N0, N1 in sizes:
        X, W = _afb(X, N0, N1)
        bands.append(W)
    bands.append(X)  # final low-pass
    comps = []
    for k in range(J + 1):
        Y = bands[J] if k == J else np.zeros(sizes[-1][0], complex)
        for j in range(J, 0, -1):
            M = 2 * round(alpha ** (j - 1) * N / 2)
            W = bands[j - 1] if k == j - 1 else np.zeros(sizes[j - 1][1], complex)
            Y = _sfb(Y, W, M)
        comps.append(np.real(np.fft.ifft(Y))[pad:])
    return np.vstack(comps)


def vmd(x, K=6, alpha=2000.0, tau=0.0, tol=1e-7):
    """Variational mode decomposition (Dragomiretskiy & Zosso, 2014) via vmdpy: K modes plus the
    reconstruction residual as a last component, so the components sum to x."""
    from vmdpy import VMD
    xe, pad = _even(x)
    u, _, _ = VMD(xe, alpha, tau, K, 0, 1, tol)
    u = u[:, pad:]
    return np.vstack([u, (np.asarray(x, float) - u.sum(0))[None, :]])


METHODS = {'dwt': dwt_mra, 'ceemdan': ceemdan, 'atrous': atrous_haar, 'tqwt': tqwt, 'vmd': vmd}
N_COMP = {'dwt': 5, 'ceemdan': 7, 'tqwt': 7, 'vmd': 7}  # fixed component count for the stepwise design

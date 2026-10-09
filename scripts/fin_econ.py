"""
fin_econ.py
-----------
Estimadores minimos (numpy) para el issue #2, compartidos por fin_power.py y
fin_models.py: MCO, logit y Poisson con errores agrupados por empresa (CR1,
inferencia t con G-1 grados de libertad, como en analysis_v3.py), doble
desviacion de efectos fijos empresa+ano y wild cluster bootstrap restringido.

Se implementan a mano para que la simulacion de potencia (decenas de miles de
ajustes) sea rapida y para no depender de linearmodels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _cr1(X: np.ndarray, scores: np.ndarray, bread: np.ndarray, codes: np.ndarray,
         k: int, small: bool = True) -> np.ndarray:
    """Varianza sandwich agrupada. scores: (N, K) contribuciones individuales."""
    G = codes.max() + 1
    S = np.zeros((G, scores.shape[1]))
    np.add.at(S, codes, scores)
    meat = S.T @ S
    V = bread @ meat @ bread
    if small:
        n = X.shape[0]
        V *= (G / (G - 1)) * ((n - 1) / (n - k))
    return V


def ols_cl(y: np.ndarray, X: np.ndarray, codes: np.ndarray, j: int = 0,
           df_absorbed: int = 0) -> tuple[float, float, float]:
    """MCO con SE agrupado (CR1). Devuelve (beta_j, se_j, p_j) con t(G-1).
    df_absorbed: parametros ya absorbidos (efectos fijos quitados por desviacion)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    k = X.shape[1] + df_absorbed
    V = _cr1(X, X * u[:, None], XtX_inv, codes, k)
    se = float(np.sqrt(V[j, j]))
    G = codes.max() + 1
    p = float(2 * stats.t.sf(abs(b[j] / se), G - 1)) if se > 0 else np.nan
    return float(b[j]), se, p


def logit_cl(y: np.ndarray, X: np.ndarray, codes: np.ndarray, j: int = 0,
             max_iter: int = 30) -> tuple[float, float, float]:
    """Logit por Newton-Raphson con SE agrupado. nan si no converge (separacion)."""
    b = np.zeros(X.shape[1])
    ybar = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    # arranque: constante al log-odds medio si hay constante
    const = np.where(np.all(X == 1, axis=0))[0]
    if len(const):
        b[const[0]] = np.log(ybar / (1 - ybar))
    for _ in range(max_iter):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(len(b)), g)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        return np.nan, np.nan, np.nan
    if np.max(np.abs(b)) > 15:
        return np.nan, np.nan, np.nan
    eta = X @ b
    p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (p * (1 - p))[:, None])
    Hinv = np.linalg.pinv(H)
    V = _cr1(X, X * (y - p)[:, None], Hinv, codes, X.shape[1])
    se = float(np.sqrt(V[j, j]))
    G = codes.max() + 1
    return float(b[j]), se, float(2 * stats.t.sf(abs(b[j] / se), G - 1))


def poisson_cl(y: np.ndarray, X: np.ndarray, codes: np.ndarray, j: int = 0,
               max_iter: int = 50) -> tuple[float, float, float]:
    """Poisson (cuasi-verosimilitud) con SE agrupado."""
    b = np.zeros(X.shape[1])
    const = np.where(np.all(X == 1, axis=0))[0]
    if len(const):
        b[const[0]] = np.log(max(y.mean(), 1e-3))
    for _ in range(max_iter):
        mu = np.exp(np.clip(X @ b, -30, 30))
        H = X.T @ (X * mu[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(len(b)), g)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        return np.nan, np.nan, np.nan
    mu = np.exp(X @ b)
    Hinv = np.linalg.pinv(X.T @ (X * mu[:, None]))
    V = _cr1(X, X * (y - mu)[:, None], Hinv, codes, X.shape[1])
    se = float(np.sqrt(V[j, j]))
    G = codes.max() + 1
    return float(b[j]), se, float(2 * stats.t.sf(abs(b[j] / se), G - 1))


def twoway_demean(v: np.ndarray, fc: np.ndarray, yc: np.ndarray, iters: int = 200,
                  tol: float = 1e-12) -> np.ndarray:
    """Quita efectos fijos de empresa y ano (proyecciones alternadas; exacto
    en un paso si el panel es balanceado). v: (N,) o (N, K)."""
    v = np.asarray(v, dtype=float)
    out = v.copy()
    for _ in range(iters):
        prev = out.copy()
        for c in (fc, yc):
            G = c.max() + 1
            if out.ndim == 1:
                s = np.bincount(c, weights=out, minlength=G) / np.bincount(c, minlength=G)
                out = out - s[c]
            else:
                cnt = np.bincount(c, minlength=G)[:, None]
                S = np.zeros((G, out.shape[1]))
                np.add.at(S, c, out)
                out = out - (S / cnt)[c]
        if np.max(np.abs(out - prev)) < tol:
            break
    return out


def dummies(s: pd.Series, drop_first: bool = True) -> np.ndarray:
    return pd.get_dummies(s.astype(str), drop_first=drop_first).to_numpy(dtype=float)


def wild_cluster_p(yd: np.ndarray, Xd: np.ndarray, codes: np.ndarray, j: int = 0,
                   n_boot: int = 9999, seed: int = 20261009) -> float:
    """Wild cluster bootstrap restringido (WCR, Rademacher) para H0: beta_j = 0
    en un modelo lineal ya desviado de efectos fijos. Generaliza el de
    analysis_v3.py a varios regresores: el modelo restringido quita la columna j."""
    G = codes.max() + 1
    Xr = np.delete(Xd, j, axis=1)
    if Xr.shape[1]:
        br = np.linalg.lstsq(Xr, yd, rcond=None)[0]
        fit_r = Xr @ br
    else:
        fit_r = np.zeros_like(yd)
    ur = yd - fit_r
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)

    def tstat(Y):  # (B, N)
        B = Y @ Xd @ XtX_inv.T          # (B, K)
        U = Y - B @ Xd.T
        tj = np.empty(Y.shape[0])
        for r in range(Y.shape[0]):
            S = np.zeros((G, Xd.shape[1]))
            np.add.at(S, codes, Xd * U[r][:, None])
            V = XtX_inv @ (S.T @ S) @ XtX_inv
            tj[r] = B[r, j] / np.sqrt(V[j, j])
        return tj

    t_obs = tstat(yd[None, :])[0]
    rng = np.random.default_rng(seed)
    t_b = np.empty(n_boot)
    chunk = 500
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        W = rng.choice([-1.0, 1.0], size=(m, G))
        Ystar = fit_r[None, :] + ur[None, :] * W[:, codes]
        t_b[s:s + m] = tstat(Ystar)
    return float((np.abs(t_b) >= abs(t_obs)).mean())

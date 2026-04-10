import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SampleData:
    name: str
    ages: np.ndarray
    sigma2_abs: np.ndarray


def _to_1sigma(sigma2_abs: np.ndarray) -> np.ndarray:
    return np.asarray(sigma2_abs, dtype=float) / 2.0


def _weighted_mean_age(ages: np.ndarray, sigma2_abs: np.ndarray) -> Dict[str, float]:
    ages = np.asarray(ages, dtype=float)
    s1 = _to_1sigma(np.asarray(sigma2_abs, dtype=float))
    weights = 1.0 / np.square(s1)
    mu = float(np.sum(weights * ages) / np.sum(weights))
    se1 = math.sqrt(1.0 / np.sum(weights))
    mswd = float(np.sum(weights * np.square(ages - mu)) / max(len(ages) - 1, 1))
    return {"mean": mu, "mean_2sigma": float(2.0 * se1), "mswd": mswd, "n": int(len(ages))}


def _common_overlap_window(ages: np.ndarray, halfwidths: np.ndarray) -> Optional[Tuple[float, float]]:
    low = float(np.max(ages - halfwidths))
    high = float(np.min(ages + halfwidths))
    return (low, high) if low <= high else None


def _sort_by_age(ages: np.ndarray, sigma2_abs: np.ndarray):
    idx = np.argsort(ages)
    return ages[idx], sigma2_abs[idx], idx


def mda_ysg(ages: np.ndarray, sigma2_abs: np.ndarray) -> Dict:
    a, s2, idx = _sort_by_age(ages, sigma2_abs)
    return {
        "abbrev": "YSG",
        "age": float(a[0]),
        "age_2sigma": float(s2[0]),
        "N": 1,
        "indices": [int(idx[0])],
    }


def mda_ydz(ages: np.ndarray, sigma2_abs: np.ndarray, n_sims: int = 200_000, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    s1 = _to_1sigma(sigma2_abs)
    draws = rng.normal(loc=ages, scale=s1, size=(n_sims, len(ages)))
    mins = np.min(draws, axis=1)
    return {
        "abbrev": "YDZ",
        "median": float(np.median(mins)),
        "mean": float(np.mean(mins)),
        "p2_5": float(np.quantile(mins, 0.025)),
        "p97_5": float(np.quantile(mins, 0.975)),
        "N": int(len(ages)),
    }


def mda_yc1sigma(ages: np.ndarray, sigma2_abs: np.ndarray, min_n: int = 2) -> Dict:
    a, s2, idx = _sort_by_age(ages, sigma2_abs)
    s1 = _to_1sigma(s2)
    for k in range(min_n, len(a) + 1):
        win = _common_overlap_window(a[:k], s1[:k])
        if win is not None:
            wm = _weighted_mean_age(a[:k], s2[:k])
            return {
                "abbrev": "YC1σ",
                "age": wm["mean"],
                "age_2sigma": wm["mean_2sigma"],
                "MSWD": wm["mswd"],
                "N": wm["n"],
                "overlap_window_1sigma": win,
                "indices": [int(i) for i in idx[:k]],
            }
    return {"abbrev": "YC1σ", "age": None}


def mda_yc2sigma(ages: np.ndarray, sigma2_abs: np.ndarray, min_n: int = 3) -> Dict:
    a, s2, idx = _sort_by_age(ages, sigma2_abs)
    for k in range(min_n, len(a) + 1):
        win = _common_overlap_window(a[:k], s2[:k])
        if win is not None:
            wm = _weighted_mean_age(a[:k], s2[:k])
            return {
                "abbrev": "YC2σ",
                "age": wm["mean"],
                "age_2sigma": wm["mean_2sigma"],
                "MSWD": wm["mswd"],
                "N": wm["n"],
                "overlap_window_2sigma": win,
                "indices": [int(i) for i in idx[:k]],
            }
    return {"abbrev": "YC2σ", "age": None}


def mda_y3za(ages: np.ndarray, sigma2_abs: np.ndarray) -> Dict:
    a, s2, idx = _sort_by_age(ages, sigma2_abs)
    if len(a) < 3:
        return {"abbrev": "Y3Za", "age": None, "reason": "Need at least 3 grains"}
    wm = _weighted_mean_age(a[:3], s2[:3])
    return {
        "abbrev": "Y3Za",
        "age": wm["mean"],
        "age_2sigma": wm["mean_2sigma"],
        "MSWD": wm["mswd"],
        "N": 3,
        "indices": [int(i) for i in idx[:3]],
    }


def mda_y3zo_2sigma(ages: np.ndarray, sigma2_abs: np.ndarray) -> Dict:
    a, s2, idx = _sort_by_age(ages, sigma2_abs)
    for i in range(0, len(a) - 2):
        win = _common_overlap_window(a[i : i + 3], s2[i : i + 3])
        if win is not None:
            wm = _weighted_mean_age(a[i : i + 3], s2[i : i + 3])
            return {
                "abbrev": "Y3Zo_2σ",
                "age": wm["mean"],
                "age_2sigma": wm["mean_2sigma"],
                "MSWD": wm["mswd"],
                "N": 3,
                "overlap_window_2sigma": win,
                "indices": [int(i) for i in idx[i : i + 3]],
            }
    return {"abbrev": "Y3Zo_2σ", "age": None}


def mda_ysp(ages: np.ndarray, sigma2_abs: np.ndarray, min_n: int = 2) -> Dict:
    from scipy.stats import chi2

    a, s2, idx = _sort_by_age(ages, sigma2_abs)
    best = None
    for k in range(min_n, len(a) + 1):
        wm = _weighted_mean_age(a[:k], s2[:k])
        df = max(k - 1, 1)
        crit = chi2.ppf(0.95, df) / df
        if wm["mswd"] <= crit:
            best = {
                "abbrev": "YSP",
                "age": wm["mean"],
                "age_2sigma": wm["mean_2sigma"],
                "MSWD": wm["mswd"],
                "MSWD_crit_95": float(crit),
                "N": wm["n"],
                "indices": [int(i) for i in idx[:k]],
            }
        else:
            break
    if best is None:
        return {"abbrev": "YSP", "age": None}
    return best


def mda_ypp(ages: np.ndarray, max_age: float = 600, grid_step: float = 0.05) -> Dict:
    from scipy.stats import gaussian_kde

    filt = ages[ages < max_age]
    if len(filt) < 2:
        return {"abbrev": "YPP", "peak_age": None}

    kde = gaussian_kde(filt, bw_method="silverman")
    x = np.arange(float(np.min(filt) - 5), float(np.max(filt) + 5), grid_step)
    y = kde(x)
    dy = np.diff(y)
    peaks = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    if len(peaks) == 0:
        p = int(np.argmax(y))
    else:
        p = int(peaks[np.argmin(x[peaks])])
    return {"abbrev": "YPP", "peak_age": float(x[p]), "used_n": int(len(filt))}


def mda_tau(ages: np.ndarray, sigma2_abs: np.ndarray, max_age: float = 600, grid_step: float = 0.05) -> Dict:
    mask = ages < max_age
    a = ages[mask]
    s1 = _to_1sigma(sigma2_abs[mask])
    if len(a) < 2:
        return {"abbrev": "τ", "age": None}

    x = np.arange(float(np.min(a) - 20), float(np.max(a) + 20), grid_step)
    pdp = np.zeros_like(x)
    for ti, si in zip(a, s1):
        pdp += (1.0 / (si * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.square((x - ti) / si))

    dy = np.diff(pdp)
    peaks = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    p = int(peaks[np.argmin(x[peaks])]) if len(peaks) else int(np.argmax(pdp))
    minima = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0] + 1
    left = minima[minima < p]
    right = minima[minima > p]
    li = int(left[-1]) if len(left) else 0
    ri = int(right[0]) if len(right) else len(x) - 1
    xl, xr = float(x[li]), float(x[ri])

    sel = np.where(mask & (ages >= xl) & (ages <= xr))[0]
    if len(sel) < 2:
        return {"abbrev": "τ", "age": None}
    wm = _weighted_mean_age(ages[sel], sigma2_abs[sel])
    return {
        "abbrev": "τ",
        "age": wm["mean"],
        "age_2sigma": wm["mean_2sigma"],
        "MSWD": wm["mswd"],
        "N": wm["n"],
        "window": (xl, xr),
        "indices": [int(i) for i in sel],
    }


def mda_mla(ages: np.ndarray, sigma2_abs: np.ndarray, max_age: float = 600) -> Dict:
    """
    Minimum age model following Vermeesch (2021) and Galbraith (2005).

    Fits a two-component mixture in log-age space:
      1. Point mass at minimum age  tm = exp(γ)  with proportion π
      2. Truncated normal (z > γ) for older grains, convolved with
         measurement error

    Uses the 3-parameter model (γ = μ) recommended for numerical
    stability with typical detrital-zircon datasets.

    The log-likelihood per grain j is:

        L_j = π · φ((z_j − γ)/s_j)/s_j
            + (1−π) · φ((z_j − μ)/τ_j)/τ_j · Φ((m_j − γ)/v_j) / Φ((μ − γ)/σ)

    where  z_j = ln(t_j),  s_j = σ_j/t_j  (radial-plot transform),
           τ_j = √(σ² + s_j²),
           m_j = (σ²·z_j + s_j²·μ) / τ_j²,
           v_j = σ·s_j / τ_j.

    Standard error of γ is obtained from the numerical Hessian;
    age uncertainty via the delta method: se(t_m) = t_m · se(γ).
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    mask = ages < max_age
    a = ages[mask]
    s1 = _to_1sigma(sigma2_abs[mask])

    if len(a) < 5:
        return {"abbrev": "MLA", "age": None, "reason": "Need \u22655 grains below max_age"}

    # ── Radial-plot transform ────────────────────────────────────────
    zj = np.log(a)
    sj = s1 / a                          # se(log t) ≈ σ/t

    # ── Negative log-likelihood (3-param: γ = μ) ─────────────────────
    def neg_loglik(params):
        gamma = params[0]
        sigma = np.exp(params[1])         # ensure σ > 0
        pi_val = 1.0 / (1.0 + np.exp(-params[2]))  # sigmoid → (0,1)

        mu = gamma                         # 3-parameter constraint

        # Component 1: point mass at minimum age
        f1 = norm.pdf(zj, loc=gamma, scale=sj)

        # Component 2: truncated-normal convolved with measurement error
        tau = np.sqrt(sigma**2 + sj**2)
        m = (sigma**2 * zj + sj**2 * mu) / tau**2
        v = sigma * sj / tau

        # Φ((μ−γ)/σ) = Φ(0) = 0.5  when μ = γ  (3-param model)
        trunc_num = norm.cdf((m - gamma) / np.maximum(v, 1e-15))
        f2 = norm.pdf(zj, loc=mu, scale=tau) * trunc_num / 0.5

        # Mixture
        f = pi_val * f1 + (1.0 - pi_val) * f2
        return -np.sum(np.log(np.maximum(f, 1e-300)))

    # ── Optimise from multiple starting points ───────────────────────
    best = None
    sorted_z = np.sort(zj)
    starts = [
        (sorted_z[0],                np.log(max(np.std(zj), 0.01)), -2.0),
        (sorted_z[min(2, len(a)-1)], np.log(max(np.std(zj)*0.5, 0.01)), -1.0),
        (np.mean(zj[:3]),            np.log(max(np.std(zj)*1.5, 0.01)), -3.0),
    ]
    for x0 in starts:
        try:
            res = minimize(
                neg_loglik, np.array(x0), method="Nelder-Mead",
                options={"maxiter": 15000, "xatol": 1e-10, "fatol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    if best is None or not best.success and best.fun > 1e30:
        return {"abbrev": "MLA", "age": None, "reason": "Optimisation failed"}

    gamma_hat = best.x[0]
    sigma_hat = float(np.exp(best.x[1]))
    pi_hat = float(1.0 / (1.0 + np.exp(-best.x[2])))
    age_mla = float(np.exp(gamma_hat))

    # ── Standard error via numerical Hessian ─────────────────────────
    age_2sigma = None
    try:
        eps = 1e-5
        n_p = len(best.x)
        H = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(i, n_p):
                xpp = best.x.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = best.x.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = best.x.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = best.x.copy(); xmm[i] -= eps; xmm[j] -= eps
                H[i, j] = (neg_loglik(xpp) - neg_loglik(xpm)
                            - neg_loglik(xmp) + neg_loglik(xmm)) / (4 * eps * eps)
                H[j, i] = H[i, j]
        cov = np.linalg.inv(H)
        se_gamma = float(np.sqrt(max(cov[0, 0], 0)))
        # Delta method: se(exp(γ)) = exp(γ) · se(γ)
        age_2sigma = float(2.0 * age_mla * se_gamma)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return {
        "abbrev": "MLA",
        "age": age_mla,
        "age_2sigma": age_2sigma,
        "pi": pi_hat,
        "sigma": sigma_hat,
        "N": int(len(a)),
        "note": "Minimum age model (Vermeesch 2021, 3-param)",
    }


def compute_all_metrics(ages: np.ndarray, sigma2_abs: np.ndarray) -> Dict[str, Dict]:
    return {
        "YSG": mda_ysg(ages, sigma2_abs),
        "YDZ": mda_ydz(ages, sigma2_abs),
        "YC1σ": mda_yc1sigma(ages, sigma2_abs),
        "YC2σ": mda_yc2sigma(ages, sigma2_abs),
        "Y3Za": mda_y3za(ages, sigma2_abs),
        "Y3Zo_2σ": mda_y3zo_2sigma(ages, sigma2_abs),
        "YSP": mda_ysp(ages, sigma2_abs),
        "MLA": mda_mla(ages, sigma2_abs),
        "YPP": mda_ypp(ages),
        "τ": mda_tau(ages, sigma2_abs),
    }

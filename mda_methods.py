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


def mda_mla(ages: np.ndarray, max_age: float = 600) -> Dict:
    """Approximate IsoplotR-style youngest component using Gaussian mixture model with BIC selection."""
    from sklearn.mixture import GaussianMixture

    a = ages[ages < max_age]
    if len(a) < 5:
        return {"abbrev": "MLA", "age": None, "reason": "Need at least 5 grains below max_age"}

    X = a.reshape(-1, 1)
    candidates = []
    max_k = min(6, len(a) - 1)
    for k in range(1, max_k + 1):
        model = GaussianMixture(n_components=k, random_state=0)
        model.fit(X)
        bic = model.bic(X)
        means = np.sort(model.means_.flatten())
        youngest = float(means[0])
        candidates.append((bic, k, youngest))

    bic, best_k, youngest = sorted(candidates, key=lambda r: r[0])[0]

    # Re-fit best model to extract covariance of the youngest component
    best_model = GaussianMixture(n_components=best_k, random_state=0)
    best_model.fit(X)
    youngest_idx = int(np.argmin(best_model.means_.flatten()))
    youngest_std = float(np.sqrt(best_model.covariances_[youngest_idx].flatten()[0]))

    return {
        "abbrev": "MLA",
        "age": youngest,
        "age_2sigma": float(2 * youngest_std),
        "components": int(best_k),
        "bic": float(bic),
        "N": int(len(a)),
        "used_n": int(len(a)),
        "note": "GMM-BIC approximation to IsoplotR youngest component",
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
        "MLA": mda_mla(ages),
        "YPP": mda_ypp(ages),
        "τ": mda_tau(ages, sigma2_abs),
    }

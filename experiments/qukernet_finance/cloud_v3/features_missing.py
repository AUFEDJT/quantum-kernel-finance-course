"""Versioned calendar-aligned missing-feature policy inspired by Qlib operators.

No original price, volume or label is imputed. Ordinary rolling statistics skip
missing observations inside fixed market-day windows. The empirical moment and
weighted-feature definitions remain course definitions, not Qlib Alpha158.
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
try:
    from ..cloud_v2.features80 import FEATURES, FEATURE_DICTIONARY as LEGACY_DICTIONARY
except ImportError:
    from cloud_v2.features80 import FEATURES, FEATURE_DICTIONARY as LEGACY_DICTIONARY

POLICY_ID = "qlib_calendar_missing_train_median_v1"
FEATURE_DICTIONARY = copy.deepcopy(LEGACY_DICTIONARY)
for _row in FEATURE_DICTIONARY:
    _row["missing_policy"] = POLICY_ID
    _name = _row["name"]
    _edge = "Fixed calendar window; ordinary aggregates use finite observations with min_periods=1; all-missing -> NaN; invalid denominator -> NaN"
    if _name.startswith(("log_momentum_", "log1p_volume_change_")) and _name != "log1p_volume_change_sd_20":
        _edge = "Exact market-date endpoints required; no compressed time or forward filling"
    elif _name in ("overnight_gap", "intraday_return", "log_range", "true_range", "body_fraction", "wick_imbalance"):
        _edge = "Required same-day or preceding-market-day OHLC must be finite; valid H=L gives body/wick zero"
    elif "corr_" in _name:
        _edge = "At least two common finite pairs; variances computed on identical pair support; either paired variance zero -> NaN"
    elif "slope_" in _name or "r2_" in _name:
        _edge = "At least two finite responses; original market-position x coordinates; zero response variance R2 -> NaN"
    elif _name.startswith(("return_sd_", "volume_cv_", "log1p_volume_change_sd_")):
        _edge = "At least two finite values for ddof=1 standard deviation; invalid denominator -> NaN"
    elif _name.startswith(("return_skew_", "return_excess_kurtosis_")):
        _edge = "Empirical central moments, not pandas unbiased skew/kurtosis; at least two finite returns and positive second central moment"
    elif _name == "maximum_drawdown_60":
        _edge = "Observed finite closing-price path inside sixty market dates; at least one observation; singleton -> 0; all-missing -> NaN; no claim about unobserved prices"
    elif _name.startswith(("signed_volume_", "clv_volume_", "volume_weighted_return_")):
        _edge = "Value and weight share identical finite-date support in numerator/denominator; nonpositive total weight -> NaN; valid H=L gives CLV zero"
    elif _name.startswith("illiquidity_"):
        _edge = "At least ceil(0.8*h) finite returns with positive volume inside h market dates; incomplete windows permitted; missing/zero-volume values are omitted"
    elif _name in ("zero_volume_fraction_20", "negative_return_fraction_20"):
        _edge = "Missing primitive values remain missing before Boolean conversion; proportion over finite observed values; all-missing -> NaN"
    elif _name.startswith("path_efficiency_"):
        _edge = "Numerator/denominator use common finite returns; valid all-zero observed path -> 0; all-missing -> NaN"
    _row["edge_policy"] = _edge


def _slope(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    x, y = np.arange(len(values), dtype=float)[valid], values[valid]
    if len(y) < 2:
        return np.nan
    x, y = x-x.mean(), y-y.mean()
    return float(x @ y / (x @ x)) if x @ x > 0 else np.nan


def _r2(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    x, y = np.arange(len(values), dtype=float)[valid], values[valid]
    if len(y) < 2 or np.ptp(y) == 0:
        return np.nan
    x, y = x-x.mean(), y-y.mean()
    denominator = (x @ x) * (y @ y)
    return float(np.clip((x @ y)**2 / denominator, 0., 1.)) if denominator > 0 else np.nan


def _moment(values, power):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.ptp(values) == 0:
        return np.nan
    centered = values-values.mean()
    m2 = np.mean(centered**2)
    return float(np.mean(centered**power)/m2**(power/2.) - (3. if power == 4 else 0.)) if m2 > 0 else np.nan


def _drawdown(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(np.max(1-values/np.maximum.accumulate(values)))


def _weighted(value, weight, h):
    joint = np.isfinite(value) & np.isfinite(weight)
    numerator = (value*weight).where(joint).rolling(h, min_periods=1).sum()
    denominator = weight.where(joint).rolling(h, min_periods=1).sum()
    return (numerator/denominator).where(denominator > 0)


def _corr(a, b, h):
    if h < 2:
        return pd.Series(np.nan, index=a.index, dtype=float)
    joint = np.isfinite(a) & np.isfinite(b)
    left, right = a.where(joint), b.where(joint)
    value = left.rolling(h, min_periods=2).corr(right)
    variance_ok = (left.rolling(h, min_periods=2).var(ddof=1) > 0) & (right.rolling(h, min_periods=2).var(ddof=1) > 0)
    return value.where(variance_ok).clip(-1., 1.)


def _supports(O, H, L, C, V, r, u, du, clv):
    """Finite primitive observation counts; not confidence or effective sample size."""
    def count(x, h): return x.where(np.isfinite(x)).rolling(h, min_periods=1).count()
    def paired(a, b, h): return a.where(np.isfinite(a) & np.isfinite(b)).rolling(h, min_periods=1).count()
    point = C.notna().astype(float)
    s = {}
    for h in (1,2,3,5,10,15,20,40,60,120):
        s[f"log_momentum_{h}"] = (C.notna() & C.shift(h).notna()).astype(float)
    for name in ("intraday_return", "log_range", "body_fraction", "wick_imbalance"):
        s[name] = point
    for name in ("overnight_gap", "true_range"):
        s[name] = (O.notna() & C.shift(1).notna()).astype(float)
    for h in (5,10,20,40,60,120): s[f"mean_deviation_{h}"] = count(C,h)
    for h in (20,60):
        for name in ("log_price_slope", "log_price_r2"): s[f"{name}_{h}"] = count(C,h)
        s[f"high_distance_{h}"] = count(H,h)
    for h in (5,10,20,40,60): s[f"return_rms_{h}"] = count(r,h)
    for h in (20,60):
        s[f"return_sd_{h}"] = count(r,h)
        for name in ("parkinson", "garman_klass"): s[f"{name}_{h}"] = count(C,h)
    s["mean_log_range_20"] = count(C,20)
    for h in (5,10,20,60): s[f"volume_ratio_{h}"] = count(V,h)
    for h in (1,5,20): s[f"log1p_volume_change_{h}"] = (u.notna() & u.shift(h).notna()).astype(float)
    for h in (20,60): s[f"volume_cv_{h}"] = count(V,h)
    s["log1p_volume_slope_20"] = count(u,20)
    s["log1p_volume_change_sd_20"] = count(du,20)
    s["zero_volume_fraction_20"] = count(V,20)
    for h in (20,60):
        s[f"return_volume_change_corr_{h}"] = paired(r,du,h)
        s[f"abs_return_volume_corr_{h}"] = paired(r,u,h)
    for h in (5,20,60): s[f"signed_volume_{h}"] = paired(r,V,h)
    for h in (20,60): s[f"clv_volume_{h}"] = paired(clv,V,h)
    s["volume_weighted_return_20"] = paired(r,V,20)
    for h in (10,20,60): s[f"downside_rms_{h}"] = count(r,h)
    for h in (20,60):
        for name in ("return_skew", "return_excess_kurtosis"): s[f"{name}_{h}"] = count(r,h)
    for name in ("negative_return_fraction_20", "maximum_negative_return_20"): s[name] = count(r,20)
    s["maximum_drawdown_60"] = count(C,60)
    for h in (20,60): s[f"illiquidity_{h}"] = count(r.where(V>0),h)
    for h in (5,60): s[f"relative_range_{h}"] = count(C,h)
    for h in (20,60):
        s[f"return_sign_autocorr_{h}"] = paired(r,r.shift(1),h)
        s[f"path_efficiency_{h}"] = count(r,h)
    assert set(s) == set(FEATURES)
    return s


def build_features(raw, calendar):
    """Return 80 raw features and D0 targets, with independent validity flags.

    Invalid bars (nonpositive prices, negative volume, inconsistent OHLC) become
    missing, including targets using their opens. Valid zero-volume bars remain.
    """
    calendar = pd.DatetimeIndex(calendar)
    tables = []
    for symbol, group in raw.groupby("symbol", sort=True):
        bars = group.set_index("date").reindex(calendar)
        p = bars[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(p).all(axis=1)
        valid = (finite & (p[["open", "high", "low", "close"]] > 0).all(axis=1)
                 & (p.volume >= 0) & (p.high >= p[["open", "close", "low"]].max(axis=1))
                 & (p.low <= p[["open", "close", "high"]].min(axis=1)))
        p = p.where(valid, np.nan)
        O, H, L, C, V = [p[name] for name in ("open", "high", "low", "close", "volume")]
        r, u = np.log(C/C.shift(1)), np.log1p(V)
        du, width, log_range = u.diff(), H-L, np.log(H/L)
        f = {}
        def mean(x, h): return x.rolling(h, min_periods=1).mean()
        def total(x, h): return x.rolling(h, min_periods=1).sum()
        def corr(a, b, h):
            return _corr(a, b, h)
        for h in (1, 2, 3, 5, 10, 15, 20, 40, 60, 120): f[f"log_momentum_{h}"] = np.log(C/C.shift(h))
        f["overnight_gap"], f["intraday_return"], f["log_range"] = np.log(O/C.shift(1)), np.log(C/O), log_range
        f["true_range"] = pd.concat([width, (H-C.shift(1)).abs(), (L-C.shift(1)).abs()], axis=1).max(axis=1, skipna=False)/C.shift(1)
        f["body_fraction"] = ((C-O)/width).mask(width.eq(0), 0.)
        f["wick_imbalance"] = ((H-p[["open", "close"]].max(axis=1)-p[["open", "close"]].min(axis=1)+L)/width).mask(width.eq(0), 0.)
        for h in (5, 10, 20, 40, 60, 120): f[f"mean_deviation_{h}"] = C/mean(C, h)-1
        for h in (20, 60):
            f[f"log_price_slope_{h}"] = np.log(C).rolling(h, min_periods=1).apply(_slope, raw=True)
            f[f"log_price_r2_{h}"] = np.log(C).rolling(h, min_periods=1).apply(_r2, raw=True)
            f[f"high_distance_{h}"] = C/H.rolling(h, min_periods=1).max()-1
        for h in (5, 10, 20, 40, 60): f[f"return_rms_{h}"] = np.sqrt(mean(r*r, h))
        for h in (20, 60):
            f[f"return_sd_{h}"] = r.rolling(h, min_periods=1).std(ddof=1)
            f[f"parkinson_{h}"] = np.sqrt(mean(log_range**2, h)/(4*np.log(2)))
            gk = mean(.5*log_range**2-(2*np.log(2)-1)*np.log(C/O)**2, h)
            f[f"garman_klass_{h}"] = np.sqrt(gk.where(gk >= -1e-15).clip(lower=0.))
        f["mean_log_range_20"] = mean(log_range, 20)
        for h in (5, 10, 20, 60): f[f"volume_ratio_{h}"] = V/mean(V, h)-1
        for h in (1, 5, 20): f[f"log1p_volume_change_{h}"] = u-u.shift(h)
        for h in (20, 60): f[f"volume_cv_{h}"] = V.rolling(h, min_periods=1).std(ddof=1)/mean(V, h)
        f["log1p_volume_slope_20"] = u.rolling(20, min_periods=1).apply(_slope, raw=True)
        f["log1p_volume_change_sd_20"] = du.rolling(20, min_periods=1).std(ddof=1)
        f["zero_volume_fraction_20"] = mean(V.eq(0).astype(float).where(V.notna()), 20)
        for h in (20, 60):
            f[f"return_volume_change_corr_{h}"] = corr(r, du, h)
            f[f"abs_return_volume_corr_{h}"] = corr(r.abs(), u, h)
        for h in (5, 20, 60): f[f"signed_volume_{h}"] = _weighted(np.sign(r), V, h)
        clv = ((2*C-H-L)/width).mask(width.eq(0), 0.)
        for h in (20, 60): f[f"clv_volume_{h}"] = _weighted(clv, V, h)
        f["volume_weighted_return_20"] = _weighted(r, V, 20)
        for h in (10, 20, 60): f[f"downside_rms_{h}"] = np.sqrt(mean(r.clip(upper=0)**2, h))
        for h in (20, 60):
            f[f"return_skew_{h}"] = r.rolling(h, min_periods=1).apply(_moment, raw=True, args=(3,))
            f[f"return_excess_kurtosis_{h}"] = r.rolling(h, min_periods=1).apply(_moment, raw=True, args=(4,))
        f["negative_return_fraction_20"] = mean(r.lt(0).astype(float).where(r.notna()), 20)
        f["maximum_negative_return_20"] = (-r).clip(lower=0).rolling(20, min_periods=1).max()
        f["maximum_drawdown_60"] = C.rolling(60, min_periods=1).apply(_drawdown, raw=True)
        for h in (20, 60):
            available = r.notna() & C.notna() & V.notna()
            value = (r.abs()/(C*V)).where(V > 0)
            count = value.rolling(h, min_periods=1).count()
            f[f"illiquidity_{h}"] = value.rolling(h, min_periods=int(np.ceil(.8*h))).mean().where((count >= np.ceil(.8*h)))
        for h in (5, 60): f[f"relative_range_{h}"] = mean(2*width/(H+L), h)
        for h in (20, 60):
            f[f"return_sign_autocorr_{h}"] = corr(np.sign(r), np.sign(r).shift(1), h)
            denominator = total(r.abs(), h)
            f[f"path_efficiency_{h}"] = (total(r, h).abs()/denominator).mask(denominator.eq(0), 0.)
        table = pd.DataFrame({name: f[name].to_numpy() for name in FEATURES})
        table = table.replace([np.inf, -np.inf], np.nan)
        table.insert(0, "j", np.arange(len(calendar)))
        table.insert(0, "date", calendar)
        table.insert(0, "symbol", str(symbol))
        # Legacy completeness is a diagnostic, not the missing-policy admission gate.
        table["history_complete"] = valid.rolling(121, min_periods=121).sum().eq(121).to_numpy()
        positions = np.flatnonzero(valid.to_numpy())
        first = positions[0] if len(positions) else len(calendar)
        table["history_span"] = np.maximum(0, np.arange(len(calendar))-first+1)
        table["row_eligible"] = valid.to_numpy() & table.history_span.ge(121)
        table["feature_valid"] = np.isfinite(table[FEATURES]).all(axis=1) & table.row_eligible
        support = _supports(O, H, L, C, V, r, u, du, clv)
        support_frame = pd.DataFrame({"support__"+name: support[name].fillna(0).to_numpy(dtype=np.int16) for name in FEATURES})
        table = pd.concat([table, support_frame], axis=1)
        table["bar_valid"] = valid.to_numpy()
        table["y_raw"] = (O.shift(-6)/O.shift(-1)-1.).to_numpy()
        table["target_valid"] = np.isfinite(table.y_raw)
        table["label_start_j"], table["label_end_j"] = table.j+1, table.j+6
        tables.append(table)
    return pd.concat(tables, ignore_index=True)

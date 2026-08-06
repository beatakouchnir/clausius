"""Calibration metrics — ECE, MCE, AURC, and recalibration.

VENDORED — copied in rather than imported, so this repository is
self-contained.
Origin: the author's earlier calibration study (Apache-2.0)
Same authorship and license as the rest of this repo (Apache-2.0); see NOTICE.

Do not edit in place — this is a snapshot of the code that produced the numbers
in FINDINGS.md, and it should not drift away from them.
"""

import math

import numpy as np


# ── candidate confidence signals from a vote distribution ────────────────
def vote_signals(votes):
    """Derive candidate confidence signals from a vote count dict {answer: n}.
    The pilot showed raw `share` is badly overconfident on MCQ; these are the
    alternatives Phase 2 compares per route:
      share      — winner's fraction of valid votes (the current signal)
      margin     — (top1 − top2) / total; decisiveness, not just plurality
      neg_entropy— 1 − H/H_max over the vote distribution; spread-aware
    Returns a dict; empty votes → all zero (maximally unsure)."""
    counts = sorted(votes.values(), reverse=True)
    total = sum(counts)
    if not total:
        return {'share': 0.0, 'margin': 0.0, 'neg_entropy': 0.0}
    top1 = counts[0]
    top2 = counts[1] if len(counts) > 1 else 0
    ps = [c / total for c in counts]
    H = -sum(p * math.log(p) for p in ps if p > 0)
    H_max = math.log(len(counts)) if len(counts) > 1 else 1.0
    return {'share': top1 / total,
            'margin': (top1 - top2) / total,
            'neg_entropy': (1 - H / H_max) if H_max > 0 else 1.0}


def _arrays(confidences, corrects):
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(corrects, dtype=float)
    if c.shape != y.shape or c.ndim != 1:
        raise ValueError("confidences and corrects must be 1-D arrays of equal length")
    if len(c) == 0:
        raise ValueError("no samples")
    return c, y


# ── calibration ────────────────────────────────────────────────────────
def reliability_bins(confidences, corrects, n_bins=10, equal_mass=False):
    """Per-bin (mean_confidence, accuracy, count) for a reliability diagram.
    equal_mass=True uses quantile bins (each ~equal count) instead of the
    equal-width [0,1] bins — steadier when confidence clusters near 1.0."""
    c, y = _arrays(confidences, corrects)
    if equal_mass:
        edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c < hi)
        if m.any():
            out.append({'lo': float(lo), 'hi': float(min(hi, 1.0)),
                        'conf': float(c[m].mean()), 'acc': float(y[m].mean()),
                        'count': int(m.sum())})
    return out


def ece(confidences, corrects, n_bins=10, equal_mass=False):
    """Expected Calibration Error: sum over bins of (count/N)·|acc − conf|."""
    c, _ = _arrays(confidences, corrects)
    n = len(c)
    return sum(b['count'] / n * abs(b['acc'] - b['conf'])
               for b in reliability_bins(confidences, corrects, n_bins, equal_mass))


def mce(confidences, corrects, n_bins=10, equal_mass=False):
    """Maximum Calibration Error: worst per-bin |acc − conf|."""
    bins = reliability_bins(confidences, corrects, n_bins, equal_mass)
    return max((abs(b['acc'] - b['conf']) for b in bins), default=0.0)


def brier(confidences, corrects):
    """Mean squared error of confidence vs. outcome (lower is better)."""
    c, y = _arrays(confidences, corrects)
    return float(np.mean((c - y) ** 2))


# ── selective prediction ────────────────────────────────────────────────
def risk_coverage(confidences, corrects):
    """Sort by confidence desc; for each coverage k/N return (coverage, risk),
    risk = error rate on the top-k most-confident. Ties broken by original order.
    Returns dict with the curve arrays plus AURC and the optimal-ranking AURC."""
    c, y = _arrays(confidences, corrects)
    order = np.argsort(-c, kind='stable')            # most confident first
    y_sorted = y[order]
    n = len(y)
    cum_correct = np.cumsum(y_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k                     # error rate on covered set
    aurc = float(np.mean(risk))                      # mean risk over all coverages
    # optimal: all correct answers ranked first (best any ranker could do)
    y_opt = np.sort(y)[::-1]
    risk_opt = 1.0 - np.cumsum(y_opt) / k
    aurc_opt = float(np.mean(risk_opt))
    return {'coverage': coverage.tolist(), 'risk': risk.tolist(),
            'aurc': aurc, 'aurc_optimal': aurc_opt,
            'e_aurc': aurc - aurc_opt}               # excess over optimal (Geifman)


def selective_accuracy_at(confidences, corrects, coverage):
    """Accuracy on the most-confident `coverage` fraction (0<coverage≤1)."""
    c, y = _arrays(confidences, corrects)
    k = max(1, int(round(coverage * len(c))))
    top = y[np.argsort(-c, kind='stable')[:k]]
    return float(top.mean())


def coverage_at_risk(confidences, corrects, max_risk):
    """Largest coverage whose selective risk stays ≤ max_risk (the 'how much can
    I answer if I tolerate X% error?' question — the abstain-threshold setter)."""
    rc = risk_coverage(confidences, corrects)
    ok = [cov for cov, r in zip(rc['coverage'], rc['risk']) if r <= max_risk]
    return max(ok) if ok else 0.0


def threshold_for_coverage(confidences, corrects, coverage):
    """The confidence threshold that yields (approximately) the given coverage —
    i.e. what to set Budget.abstain_below to answer that fraction."""
    c, _ = _arrays(confidences, corrects)
    k = max(1, int(round(coverage * len(c))))
    return float(np.sort(c)[::-1][k - 1])


def threshold_at_risk(confidences, corrects, max_risk=0.15):
    """The confidence to answer at for a chosen error budget: the largest coverage
    whose selective risk stays ≤ max_risk, expressed as the confidence of the
    least-confident answered item. Abstain BELOW the returned value. This is the
    operating-point setter — where coverage_at_risk asks 'how much can I answer at
    X% error?', this returns the actual threshold that realizes it, per route.

    Returns None if no threshold clears the target at any coverage (the route
    can't hit that risk → caller falls back).

    Evaluated at each DISTINCT confidence level (group risk over {conf ≥ t}), not
    a top-k prefix: vote-share confidence is discrete and heavily tied, so a
    prefix cut would break ties arbitrarily and a single wrong item at the top
    could spuriously reject the whole route. The lowest qualifying threshold
    (maximum coverage) is chosen."""
    c, y = _arrays(confidences, corrects)
    best_t, best_cov = None, -1.0
    for t in np.unique(c):                            # ascending distinct levels
        mask = c >= t
        risk = 1.0 - float(y[mask].mean())            # error on everything answered at t
        cov = float(mask.mean())
        if risk <= max_risk and cov > best_cov:       # keep the widest-coverage cut
            best_cov, best_t = cov, float(t)
    return best_t


def min_risk_threshold(confidences, corrects, min_coverage=0.25):
    """Best-effort operating point for a route that CANNOT meet its target risk at
    any coverage (e.g. the conceptual-MCQ vote route, whose best-calibrated bucket
    tops out below the target accuracy): the threshold with the lowest achievable
    selective risk among cuts that still answer ≥ min_coverage of items. Returns
    (threshold, achieved_risk, coverage), or None if even min_coverage is unreachable.
    Keeps a route usable and HONEST — it operates at the risk it can actually hit,
    and the caller discloses that the desired target was infeasible."""
    c, y = _arrays(confidences, corrects)
    best = None
    for t in np.unique(c):
        mask = c >= t
        cov = float(mask.mean())
        if cov < min_coverage:
            continue
        risk = 1.0 - float(y[mask].mean())
        if best is None or risk < best[1] or (risk == best[1] and cov > best[2]):
            best = (float(t), risk, cov)
    return best


def summary(confidences, corrects, n_bins=10):
    """One dict with the headline numbers for a report row."""
    c, y = _arrays(confidences, corrects)
    rc = risk_coverage(c, y)
    return {'n': int(len(c)), 'accuracy': float(y.mean()),
            'mean_conf': float(c.mean()),
            'ece': ece(c, y, n_bins), 'ece_equalmass': ece(c, y, n_bins, True),
            'mce': mce(c, y, n_bins), 'brier': brier(c, y),
            'aurc': rc['aurc'], 'e_aurc': rc['e_aurc'],
            'overconfidence': float(c.mean() - y.mean()),  # >0 = overconfident
            'sel_acc@50': selective_accuracy_at(c, y, 0.5),
            'sel_acc@80': selective_accuracy_at(c, y, 0.8),
            'cov@risk10': coverage_at_risk(c, y, 0.10)}


# ── recalibration (post-hoc) ─────────────────────────────────────────────
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def platt_scale(confidences, corrects, a_grid=None, b_grid=None):
    """Logistic (Platt) recalibration: fit p' = sigmoid(a·logit(p) + b) by
    minimizing log-loss. Returns (params, recal_fn). The 2-parameter form is the
    honest choice for vote-share confidence — a single temperature (b=0) only
    sharpens symmetrically around 0.5 and can't correct a one-directional bias
    (mean-conf ≠ mean-acc), which is exactly how vote share fails. Grid-searched,
    so no optimizer dependency. FIT on a calibration split, APPLY to a test split
    (fitting and scoring on the same data understates ECE)."""
    c, y = _arrays(confidences, corrects)
    lg = _logit(c)
    # a_grid floor near 0 so a near-uninformative signal can be flattened to the
    # base rate (a→0) rather than clipped — that flattening is itself a finding.
    a_grid = a_grid if a_grid is not None else np.concatenate(
        [np.linspace(0.0, 0.1, 6)[1:], np.linspace(0.15, 3.0, 30)])
    b_grid = b_grid if b_grid is not None else np.linspace(-3.0, 3.0, 61)

    def nll(a, b):
        p = 1.0 / (1.0 + np.exp(-(a * lg + b)))
        p = np.clip(p, _EPS, 1 - _EPS)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best, best_loss = (1.0, 0.0), 1e18
    for a in a_grid:
        for b in b_grid:
            loss = nll(a, b)
            if loss < best_loss:
                best_loss, best = loss, (float(a), float(b))
    a, b = best
    return best, (lambda p, a=a, b=b: 1.0 / (1.0 + np.exp(-(a * _logit(p) + b))))


# ── bootstrap confidence intervals ───────────────────────────────────────
def bootstrap_ci(metric, confidences, corrects, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric(conf, corr) -> float. ECE and AURC on
    small local-model samples are noisy, so the point estimate alone overstates
    precision — report the interval. Returns {point, lo, hi, se}.

    Note: ECE is a positively-biased statistic (a sum of |acc−conf| over bins), so
    its bootstrap distribution skews above the point estimate and the CI can sit
    entirely above `point` (Kumar et al. 2019). That is expected, not a bug — the
    interval reflects resampling variability, and the upward bias is disclosed in
    the writeup. AURC and accuracy are mean-like and bracket their point normally."""
    c, y = _arrays(confidences, corrects)
    n = len(c)
    point = float(metric(c, y))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots[i] = metric(c[idx], y[idx])
        except Exception:
            boots[i] = np.nan
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'point': point, 'lo': float(lo), 'hi': float(hi), 'se': float(boots.std())}


def summary_ci(confidences, corrects, n_boot=2000, seed=0):
    """Headline metrics with bootstrap CIs — the publication-grade report row."""
    c, y = _arrays(confidences, corrects)
    acc = np.mean(y)
    return {
        'n': int(len(c)),
        'accuracy': {'point': float(acc),
                     **{k: v for k, v in bootstrap_ci(
                         lambda a, b: float(np.mean(b)), c, y, n_boot, seed).items()
                        if k in ('lo', 'hi')}},
        'ece': bootstrap_ci(lambda a, b: ece(a, b), c, y, n_boot, seed),
        'aurc': bootstrap_ci(lambda a, b: risk_coverage(a, b)['aurc'], c, y, n_boot, seed),
        'mean_conf': float(np.mean(c)),
    }


# ── cross-validated recalibration selection ──────────────────────────────
def _kfold_indices(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]           # k roughly-equal disjoint folds


def cross_val_ece(x, y, k=5, seed=0, recal=True):
    """Mean held-out ECE for a signal x against outcomes y, optionally with a
    Platt recalibration fit per fold. This is how signals are compared honestly —
    fitting and scoring on the same data understates error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < k or len(np.unique(y)) < 2:
        return float('nan')
    folds = _kfold_indices(n, k, seed)
    eces = []
    for f in folds:
        test = f
        train = np.setdiff1d(np.arange(n), test)
        if not len(test) or not len(train):
            continue
        if recal:
            _, fn = platt_scale(x[train], y[train])
            p = fn(x[test])
        else:
            p = x[test]
        eces.append(ece(p, y[test], n_bins=min(10, max(2, len(test) // 3))))
    return float(np.mean(eces)) if eces else float('nan')


def oof_recalibrated(x, y, k=5, seed=0):
    """Out-of-fold recalibrated predictions: each point is scored by a Platt fit
    on the OTHER folds. The honest way to draw an 'after' reliability diagram —
    no point is recalibrated using itself."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    out = np.array(x, dtype=float)
    if n < k or len(np.unique(y)) < 2:
        return out
    for test in _kfold_indices(n, k, seed):
        train = np.setdiff1d(np.arange(n), test)
        if len(train) and len(test):
            _, fn = platt_scale(x[train], y[train])
            out[test] = fn(x[test])
    return out


class Recalibrator:
    """Per-route confidence recalibration: for each route, a chosen vote signal
    (share / margin / neg_entropy) plus Platt (a, b) mapping it to a calibrated
    probability-correct. Fit from labelled (route, vote_stats, correct) records —
    the pilot showed the routes miscalibrate differently, so one global map is
    wrong. Ships as JSON so a fitted recalibrator travels with the model.

    routes = {key: {'signal': str, 'a': float, 'b': float}}
    thresholds = {key: float}   # per-key abstain-below operating point (§ below)

    A `key` is a route ('math') or a route+sub-route ('mcq/knowledge',
    'mcq/stem') when that route calibrates its content classes separately — the
    n=740 study showed one MCQ map hides opposite biases that cancel in aggregate
    (under-crediting knowledge, overconfident on near-chance STEM). apply() and
    abstain_threshold() take an optional `subject` and resolve the compound key
    first, falling back to the flat route key, then to raw."""

    def __init__(self, routes, meta=None, thresholds=None):
        self.routes = routes
        self.meta = meta or {}
        # per-route abstain thresholds, on the SAME confidence scale each route
        # emits at inference (recalibrated prob for mapped routes, raw share for
        # kept-raw routes) — so the harness can compare a route's confidence to
        # its own threshold. See fit(target_risk=...).
        self.thresholds = thresholds or {}

    @staticmethod
    def _resolve(table, route, subject):
        """Compound key (route/subject) if present, else the flat route, else None."""
        if subject and f"{route}/{subject}" in table:
            return f"{route}/{subject}"
        return route if route in table else None

    def apply(self, route, vote_stats, subject=None):
        key = self._resolve(self.routes, route, subject)
        if key is None or not vote_stats:
            return None
        cfg = self.routes[key]
        p = vote_stats.get(cfg['signal'])
        if p is None:
            return None
        z = cfg['a'] * float(_logit(p)) + cfg['b']
        return float(1.0 / (1.0 + math.exp(-z)))

    def abstain_threshold(self, route, subject=None):
        """Per-route (or per-sub-route) confidence threshold to answer at (abstain
        below it), or None if there is no fitted operating point. Comparable to the
        confidence the route emits: recalibrated prob for mapped keys, raw share
        otherwise."""
        key = self._resolve(self.thresholds, route, subject)
        return self.thresholds.get(key) if key else None

    @classmethod
    def fit(cls, records, signals=('share', 'margin', 'neg_entropy'),
            min_n=12, k=5, seed=0, min_gain=0.01, target_risk=0.15,
            split_routes=()):
        """Per key: pick the signal with the lowest cross-validated recalibrated
        ECE, then fit Platt on all of that key's data for the shipped params.
        `records`: list of {route, vote_stats: {...}, correct: bool[, subject]}.

        `split_routes`: routes that calibrate their content classes separately —
        a record's `subject` field then extends its key to 'route/subject'
        (e.g. 'mcq/knowledge'). A split sub-route that lacks enough data falls
        through the min_n gate on its own, so the flat map is the safe default.

        DO-NO-HARM gate: a key gets a recalibration map only if it beats the
        status quo (raw `share`, ungated) by `min_gain` ECE in cross-validation —
        otherwise the key keeps its raw confidence (apply() → None). This mirrors
        the grounded route's gate: don't 'fix' a route that is already calibrated,
        where small-n Platt would only add noise (the pilot showed math is
        already well-calibrated; MCQ is what needs the fix)."""
        split_routes = set(split_routes)

        def _key(r):
            route = r['route']
            if route in split_routes and r.get('subject'):
                return f"{route}/{r['subject']}"
            return route

        by_route = {}
        for r in records:
            if r.get('vote_stats') and 'correct' in r:
                by_route.setdefault(_key(r), []).append(r)
        routes, meta, thresholds = {}, {}, {}

        def _fit_threshold(conf, y, info):
            """Widest-coverage cut at target_risk; if infeasible, the route's best
            achievable risk (disclosed in info). Records into thresholds/info."""
            t = threshold_at_risk(conf, y, target_risk)
            if t is None:
                best = min_risk_threshold(conf, y)
                if best is None:
                    return
                t, achieved, cov = best
                info['abstain_target_infeasible'] = {
                    'target_risk': target_risk, 'achieved_risk': round(achieved, 3),
                    'coverage': round(cov, 3)}
            thresholds[route] = t
            info['abstain_at'] = t

        for route, recs in by_route.items():
            y = np.array([float(r['correct']) for r in recs])
            if len(recs) < min_n or len(np.unique(y)) < 2:
                meta[route] = {'skipped': f'n={len(recs)} (<{min_n}) or single-class'}
                continue
            scored = {}
            for sig in signals:
                if all(sig in r['vote_stats'] for r in recs):
                    x = np.array([r['vote_stats'][sig] for r in recs])
                    scored[sig] = cross_val_ece(x, y, k=k, seed=seed)
            scored = {s: v for s, v in scored.items() if not math.isnan(v)}
            if not scored:
                continue
            best_sig = min(scored, key=scored.get)
            status_quo = cross_val_ece(
                np.array([r['vote_stats']['share'] for r in recs]), y,
                k=k, seed=seed, recal=False) if all('share' in r['vote_stats'] for r in recs) else float('nan')
            gain = (status_quo - scored[best_sig]) if not math.isnan(status_quo) else 1.0
            info = {'n': len(recs), 'cv_ece_by_signal': scored,
                    'status_quo_ece': status_quo, 'chosen': best_sig, 'cv_gain': gain}
            if gain < min_gain:                        # do-no-harm: leave raw
                info['kept_raw'] = f'CV gain {gain:.3f} < {min_gain}'
                # abstain threshold on the RAW share the route still emits
                if all('share' in r['vote_stats'] for r in recs):
                    _fit_threshold(np.array([r['vote_stats']['share'] for r in recs]), y, info)
                meta[route] = info
                continue
            x = np.array([r['vote_stats'][best_sig] for r in recs])
            (a, b), _ = platt_scale(x, y)
            routes[route] = {'signal': best_sig, 'a': a, 'b': b}
            # abstain threshold on the OUT-OF-FOLD recalibrated confidence — the
            # scale this route emits at inference, and honest (no point uses its
            # own fold to set the operating point it will be judged against).
            _fit_threshold(oof_recalibrated(x, y, k=k, seed=seed), y, info)
            meta[route] = info
        meta['_target_risk'] = target_risk
        return cls(routes, meta, thresholds)

    def to_dict(self):
        return {'routes': self.routes, 'meta': self.meta, 'thresholds': self.thresholds}

    def save(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        return cls(d['routes'], d.get('meta', {}), d.get('thresholds', {}))


# ── plotting (optional — needs the `calibration` extra) ──────────────────
def _require_mpl():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise SystemExit("plotting needs matplotlib — `uv sync --extra calibration`")

AMBER, DIM, ALERT = '#e8a13d', '#9a9484', '#d26a4b'


def plot_reliability(confidences, corrects, path, title='reliability', n_bins=10):
    plt = _require_mpl()
    bins = reliability_bins(confidences, corrects, n_bins)
    e = ece(confidences, corrects, n_bins)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([0, 1], [0, 1], '--', color=DIM, lw=1, label='perfect')
    xs = [b['conf'] for b in bins]
    ax.bar(xs, [b['acc'] for b in bins], width=1.0 / n_bins * 0.9,
           color=AMBER, alpha=0.85, edgecolor='#5a4a2a')
    for b in bins:                                    # gap = miscalibration
        ax.plot([b['conf'], b['conf']], [b['acc'], b['conf']], color=ALERT, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='confidence', ylabel='accuracy',
           title=f"{title}  (ECE={e:.3f}, n={len(confidences)})")
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_risk_coverage(confidences, corrects, path, title='risk–coverage',
                       threshold=None):
    plt = _require_mpl()
    rc = risk_coverage(confidences, corrects)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot(rc['coverage'], rc['risk'], color=AMBER, lw=2, label='study')
    y_opt = np.sort(np.asarray(corrects, float))[::-1]
    k = np.arange(1, len(y_opt) + 1)
    ax.plot(k / len(y_opt), 1 - np.cumsum(y_opt) / k, '--', color=DIM, lw=1,
            label='optimal ranking')
    if threshold is not None:                         # where abstain_below sits
        cov = float(np.mean(np.asarray(confidences) >= threshold))
        ax.axvline(cov, color=ALERT, lw=1, ls=':',
                   label=f'abstain@{threshold:g} → cov {cov:.2f}')
    ax.set(xlim=(0, 1), ylim=(0, max(0.05, max(rc['risk']) * 1.1)),
           xlabel='coverage (fraction answered)', ylabel='risk (error rate)',
           title=f"{title}  (AURC={rc['aurc']:.3f}, E-AURC={rc['e_aurc']:.3f})")
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == '__main__':
    # self-test on synthetic data: a well-calibrated signal vs an overconfident one
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y_cal = (rng.uniform(0, 1, 4000) < p).astype(float)          # perfectly calibrated
    y_over = (rng.uniform(0, 1, 4000) < p ** 2).astype(float)    # p too high
    print("calibrated  :", {k: round(v, 3) for k, v in summary(p, y_cal).items()})
    print("overconfident:", {k: round(v, 3) for k, v in summary(p, y_over).items()})
    # fit on first half, apply to second (honest held-out recalibration)
    (a, b), recal = platt_scale(p[:2000], y_over[:2000])
    print(f"platt a={a:.2f} b={b:.2f}  held-out ECE "
          f"{ece(p[2000:], y_over[2000:]):.3f} → {ece(recal(p[2000:]), y_over[2000:]):.3f}")

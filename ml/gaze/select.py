"""Per-axis feature choice for the gaze aim, by leave-one-dot-out over the 12
fixed dots (plus the 6 check dots as a second opinion).

    python ml/gaze/select.py [session.json]
"""
import sys, itertools
import numpy as np
sys.argv = sys.argv[:2]
exec(open('ml/gaze/analyze.py').read().split('# 1. How strongly')[0])   # loads rows, feats, W, H

def pursuit_at(t): return (.5 + .42 * np.sin(2 * np.pi * t / 11), .5 + .37 * np.sin(2 * np.pi * t / 7.3 + np.pi / 2))
pur = [r for r in d['frames'] if r['phase'] == 'pursuit']
off = min(np.arange(-3000, 3000, 10), key=lambda o: np.mean([(pursuit_at((r['t'] - pur[0]['t'] - o) / 1000)[0] - r['target']['x']) ** 2 for r in pur[::5]]))
t0 = pur[0]['t'] + off

def data(lag=.2):
    out = []
    for r, f in rows:
        t = r['target']
        if not t or not t['settled']: continue
        if r['phase'] == 'pursuit': out.append((f, pursuit_at((r['t'] - t0) / 1000 - lag), 'pur'))
        else: out.append((f, (t['x'], t['y']), f"{r['phase']}{r['index']}"))
    return out
D = data()

def fit(X, y, lam, w=None):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; w = np.ones(len(Z)) if w is None else w
    A = (Z * w[:, None]).T @ Z + lam * w.sum() * np.eye(Z.shape[1]); ym = np.average(y, weights=w)
    c = np.linalg.solve(A, (Z * w[:, None]).T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ c + ym
def feats_of(fs, keys, poly):
    X = np.array([[f[k] for k in keys] for f in fs])
    if poly: X = np.hstack([X, X ** 2])
    return X

def score(axis, keys, lam, poly, pursuit_w):
    """Median error (px) on held-out dots: each fixed dot left out in turn, then the check dots."""
    A = 0 if axis == 'x' else 1; S = W if axis == 'x' else H
    groups = sorted(set(g for *_, g in D if g.startswith('fix')))
    errs = []
    for held in groups + ['check']:
        tr = [(f, t, g) for f, t, g in D if not g.startswith('check') and g != held]
        te = [(f, t, g) for f, t, g in D if (g.startswith('check') if held == 'check' else g == held)]
        w = np.array([pursuit_w if g == 'pur' else 1 for *_, g in tr])
        m = fit(feats_of([f for f, *_ in tr], keys, poly), np.array([t[A] for _, t, _ in tr]), lam, w)
        for g in sorted(set(g for *_, g in te)):
            sel = [(f, t) for f, t, gg in te if gg == g]
            p = m(feats_of([f for f, _ in sel], keys, poly))
            errs.append(abs(np.median(p) - sel[0][1][A]) * S)
    return np.median(errs), np.mean(errs)

X_CAND = {'ix bx': ['ix', 'bx'], 'bs in/out': ['bs_inL', 'bs_outL', 'bs_inR', 'bs_outR'], 'bs in/out + ix': ['bs_inL', 'bs_outL', 'bs_inR', 'bs_outR', 'ix'],
          'irxL irxR bx': ['irxL', 'irxR', 'bx'], 'ix bx hx': ['ix', 'bx', 'hx']}
Y_CAND = {'by': ['by'], 'bs up/down': ['bs_upL', 'bs_downL', 'bs_upR', 'bs_downR'], 'bs up/down + iy': ['bs_upL', 'bs_downL', 'bs_upR', 'bs_downR', 'iy'],
          'iy by': ['iy', 'by'], 'bs up/down + irisTop': ['bs_upL', 'bs_downL', 'bs_upR', 'bs_downR', 'irisTopL', 'irisTopR'],
          'bs up/down + iy + blink': ['bs_upL', 'bs_downL', 'bs_upR', 'bs_downR', 'iy', 'bs_blinkL', 'bs_blinkR'], 'iy by lo hy': ['iy', 'by', 'lo', 'hy']}
for axis, cands in (('x', X_CAND), ('y', Y_CAND)):
    print(f'\naxis {axis}: median / mean error px over held-out dots')
    out = []
    for (name, keys), lam, poly, pw in itertools.product(cands.items(), (1e-3, 1e-2, 5e-2), (False, True), (0, .3, 1)):
        med, mean = score(axis, keys, lam, poly, pw)
        out.append((med, mean, name, lam, poly, pw))
    for med, mean, name, lam, poly, pw in sorted(out)[:10]:
        print(f'  {med:5.0f} {mean:5.0f}  {name:28s} λ={lam:<5} {"sq " if poly else "lin"} glide×{pw}')

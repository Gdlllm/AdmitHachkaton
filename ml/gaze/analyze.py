"""Offline look at a gaze calibration recording (ml/gaze/data/session-*.json):
which signals follow the eyes, and which model aims best on the held-out check dots.

    python ml/gaze/analyze.py [session.json]
"""
import json, sys, glob
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob('ml/gaze/data/session-*.json'))[-1]
d = json.load(open(path))
W, H = d['screen']['w'], d['screen']['h']
FR = [r for r in d['frames'] if r['f'] and r['raw'] and r['bs']]
SHAPES = ['inL', 'outL', 'upL', 'downL', 'inR', 'outR', 'upR', 'downR', 'blinkL', 'blinkR', 'squintL', 'squintR', 'wideL', 'wideR']
RAW = [33, 133, 159, 145, 160, 144, 158, 153, 468, 469, 470, 471, 472, 263, 362, 386, 374, 387, 373, 385, 380, 473, 474, 475, 476, 477, 1, 234, 454, 13, 14, 10, 152]
ix = {v: i for i, v in enumerate(RAW)}

def feats(r):
    """A wide set of candidate signals, all in the mirrored screen sense (x: towards screen right)."""
    p = np.array(r['raw']); P = lambda i: p[ix[i]]
    out = dict(r['f'])
    for name, (o, i_, top, bot, iris) in {'L': (33, 133, 159, 145, 468), 'R': (263, 362, 386, 374, 473)}.items():
        w = np.linalg.norm(P(o)[:2] - P(i_)[:2]); mid = (P(o) + P(i_)) / 2
        out['irx' + name] = -(P(iris)[0] - mid[0]) / w
        out['iry' + name] = (P(iris)[1] - mid[1]) / w
        out['lid' + name] = (P(bot)[1] - P(top)[1]) / w
        out['irisTop' + name] = (P(iris)[1] - P(top)[1]) / w     # iris centre below the upper lid
    b = dict(zip(SHAPES, r['bs']))
    for k, v in b.items(): out['bs_' + k] = v
    m = np.array(r['m']).reshape(4, 4)
    for i in range(3):
        for j in range(3): out[f'm{i}{j}'] = m[i, j]
    return out

rows = [(r, feats(r)) for r in FR]
KEYS = list(rows[0][1].keys())

def split(lag=0.12):
    tr, te = [], []
    for r, f in rows:
        t = r['target']
        if not t or not t['settled']: continue
        if r['phase'] == 'fix': tr.append((f, (t['x'], t['y']), 'fix%d' % r['index']))
        elif r['phase'] == 'pursuit': tr.append((f, (t['lagged']['x'], t['lagged']['y']) if lag == .12 else None, 'pur'))
        elif r['phase'] == 'check': te.append((f, (t['x'], t['y']), 'chk%d' % r['index']))
    return tr, te

# 1. How strongly each signal follows the dot, and how noisy it is on one dot.
tr, te = split()
fix = [(f, t, g) for f, t, g in tr if g.startswith('fix')]
print(f'{path}\nscreen {W}x{H}, frames {len(FR)}, fix {len(fix)}, pursuit {sum(g == "pur" for *_, g in tr)}, check {len(te)}')
print('\nsignal          corr_x  corr_y   noise   range/noise(best axis)')
res = []
for k in KEYS:
    v = np.array([f[k] for f, *_ in fix]); tx = np.array([t[0] for _, t, _ in fix]); ty = np.array([t[1] for _, t, _ in fix])
    if v.std() < 1e-9: continue
    cx, cy = np.corrcoef(v, tx)[0, 1], np.corrcoef(v, ty)[0, 1]
    groups = {}
    for (f, t, g) in fix: groups.setdefault(g, []).append(f[k])
    noise = np.sqrt(np.mean([np.var(a) for a in groups.values()]))
    means = [np.mean(a) for a in groups.values()]
    res.append((max(abs(cx), abs(cy)), k, cx, cy, noise, (max(means) - min(means)) / max(noise, 1e-9)))
for s, k, cx, cy, n, snr in sorted(res, reverse=True)[:22]:
    print(f'{k:14s} {cx:+.2f}   {cy:+.2f}   {n:.4f}   {snr:5.1f}')

# 2. Models: train on dots + glide, test on the check dots (never seen), error in pixels.
def design(fs, keys, poly=False):
    X = np.array([[f[k] for k in keys] for f in fs])
    if poly:
        n = X.shape[1]; X = np.hstack([X] + [X[:, [i]] * X[:, [j]] for i in range(n) for j in range(i, n)])
    return X
def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]); w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) / sd) @ w + y.mean()
def evaluate(keys_x, keys_y, lam=1e-3, poly=False, lag=.12, use_pursuit=True, verbose=False):
    trn = [(r, f) for r, f in rows if r['target'] and r['target']['settled'] and (r['phase'] == 'fix' or (use_pursuit and r['phase'] == 'pursuit'))]
    def tgt(r):
        if r['phase'] == 'pursuit':
            tt = (r['t'] - trn_t0) / 1000.0 - lag
            return pursuit_at(tt)
        return (r['target']['x'], r['target']['y'])
    fs = [f for _, f in trn]; T = np.array([tgt(r) for r, _ in trn])
    tst = [(r, f) for r, f in rows if r['phase'] == 'check' and r['target'] and r['target']['settled']]
    Tt = np.array([(r['target']['x'], r['target']['y']) for r, _ in tst])
    fx = ridge_fit(design(fs, keys_x, poly), T[:, 0], lam); fy = ridge_fit(design(fs, keys_y, poly), T[:, 1], lam)
    px, py = fx(design([f for _, f in tst], keys_x, poly)), fy(design([f for _, f in tst], keys_y, poly))
    # Per check dot: the median aim (what the smoothed reticle shows), and the frame-to-frame jitter.
    errs, jit = [], []
    for i in sorted(set(r['index'] for r, _ in tst)):
        sel = [k for k, (r, _) in enumerate(tst) if r['index'] == i]
        ex, ey = (np.median(px[sel]) - Tt[sel[0], 0]) * W, (np.median(py[sel]) - Tt[sel[0], 1]) * H
        errs.append((abs(ex), abs(ey))); jit.append((np.std(px[sel]) * W, np.std(py[sel]) * H))
    e = np.array(errs); j = np.array(jit)
    return np.median(np.hypot(e[:, 0], e[:, 1])), np.median(e[:, 0]), np.median(e[:, 1]), np.median(j[:, 0]), np.median(j[:, 1])

pur = [r for r in d['frames'] if r['phase'] == 'pursuit']
trn_t0 = pur[0]['t'] - 1000 * 0  # pursuit clock: the page's phase start is not stored; recover it from the lagged target
# Recover the pursuit start from the recorded (unlagged) target: x = .5 + .42 sin(2πt/11).
def pursuit_at(t): return (.5 + .42 * np.sin(2 * np.pi * t / 11), .5 + .37 * np.sin(2 * np.pi * t / 7.3 + np.pi / 2))
best = None
for off in np.arange(-3000, 3000, 10):
    t = np.array([(r['t'] - (pur[0]['t'] + off)) / 1000 for r in pur])
    err = np.mean([(pursuit_at(tt)[0] - r['target']['x']) ** 2 + (pursuit_at(tt)[1] - r['target']['y']) ** 2 for tt, r in zip(t, pur)])
    if best is None or err < best[0]: best = (err, off)
trn_t0 = pur[0]['t'] + best[1]
print(f'\npursuit clock recovered (fit error {best[0]:.2e})')

print('\nmodel                                         err_px  err_x  err_y  jitter_x jitter_y')
sets = {
    'current (hx ix bx | hy iy by lo)': (['hx', 'ix', 'bx'], ['hy', 'iy', 'by', 'lo']),
    'eyes only (ix bx | iy by)': (['ix', 'bx'], ['iy', 'by']),
    'blendshapes only': (['bs_inL', 'bs_outL', 'bs_inR', 'bs_outR'], ['bs_upL', 'bs_downL', 'bs_upR', 'bs_downR']),
    'per-eye irises + blendshapes': (['irxL', 'irxR', 'bs_inL', 'bs_outL', 'bs_inR', 'bs_outR'], ['iryL', 'iryR', 'irisTopL', 'irisTopR', 'lidL', 'lidR', 'bs_upL', 'bs_downL', 'bs_upR', 'bs_downR']),
    'everything eye + head matrix': (['irxL', 'irxR', 'bs_inL', 'bs_outL', 'bs_inR', 'bs_outR', 'hx', 'm02', 'm20'], ['iryL', 'iryR', 'irisTopL', 'irisTopR', 'lidL', 'lidR', 'bs_upL', 'bs_downL', 'bs_upR', 'bs_downR', 'hy', 'm12', 'm21', 'bs_squintL', 'bs_squintR', 'bs_wideL', 'bs_wideR']),
}
for name, (kx, ky) in sets.items():
    for lam in (1e-3, 1e-2, 5e-2):
        for poly in (False, True):
            if poly and len(kx) + len(ky) > 12: continue
            e = evaluate(kx, ky, lam, poly)
            print(f'{name[:34]:34s} λ={lam:<5} {"poly" if poly else "lin ":4s} {e[0]:6.0f} {e[1]:6.0f} {e[2]:6.0f} {e[3]:8.1f} {e[4]:8.1f}')
print('\neye lag for the glide (current feature set, λ=1e-2):')
for lag in (0, .08, .12, .18, .25, .35):
    print(f'  lag {lag:.2f}s -> err {evaluate(*sets["per-eye irises + blendshapes"], 1e-2, False, lag)[0]:.0f}px')
print('without the glide:', f'{evaluate(*sets["per-eye irises + blendshapes"], 1e-2, False, .12, use_pursuit=False)[0]:.0f}px')

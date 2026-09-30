"""Score the fitted mesh facing against labels by eye (real_labels.json) and
count front/back flips of the mesh, for one or more eval_app.mjs output dirs.
  python score_real.py DIR [DIR ...]"""
import json, os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from summarize_eval import facing_track
from facing import mesh_facing


def hard_flips(path, band=.4):
    """Changes between a clearly front-facing and a clearly back-facing mesh
    (|forward.z| > band); wobble while side-on does not count."""
    frames = json.load(open(path))['frames']; state, count = None, 0
    for f in frames:
        if not f['mesh']: continue
        z = mesh_facing(np.array(f['mesh']['kp']))[1][2]
        now = 'F' if z < -band else 'B' if z > band else None
        if now and state and now != state: count += 1
        if now: state = now
    return count
labels = json.load(open(os.path.join(os.path.dirname(__file__), 'real_labels.json')))
for d in sys.argv[1:]:
    right = total = 0; per = []; flips = frames = mesh_frames = 0; duration = 0; hard = 0
    for path in sorted(glob.glob(os.path.join(d, '*.json'))):
        name = os.path.basename(path)[:-5]
        tr = facing_track(path)
        frames += len(tr); mesh_frames += sum(1 for x in tr if x[2])
        flips += sum(1 for a, b in zip(tr, tr[1:]) if a[2] and b[2] and a[2] != b[2]); duration += tr[-1][0] - tr[0][0] if tr else 0
        hard += hard_flips(path)
        if name not in labels: continue
        l = labels[name]; r = t = 0
        for i, ch in enumerate(l['labels']):
            if ch == 'S': continue
            at = l['start'] + i
            near = [x[2] for x in tr if abs(x[0] - at) <= .25 and x[2]]
            if not near: continue
            guess = 'F' if sum(near) > 0 else 'B'
            r += guess == ch; t += 1
        per.append(f'{name} {r}/{t}'); right += r; total += t
    print(f'{os.path.basename(d)}: facing right {right}/{total} ({right / max(total, 1):.0%}) | sign flips {flips / max(duration, 1) * 60:.0f}/min, hard front/back flips {hard} ({hard / max(duration, 1) * 60:.1f}/min) in {duration:.0f}s | mesh on {mesh_frames}/{frames} frames')
    print('   ', ', '.join(per))

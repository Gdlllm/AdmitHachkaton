"""Per-video agreement between MediaPipe's left/right labels and the fitted
mesh facing, from eval_app.mjs output.  python summarize_eval.py DIR [names]"""
import json, sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from facing import label_facing, mesh_facing
def facing_track(path):
    d = json.load(open(path)); out = []
    for f in d['frames']:
        pose = np.array(f['pose']) if len(f['pose']) == 33 else None
        torso = pose is not None and min(pose[[11, 12, 23, 24], 2]) > 0.5
        lf = float(label_facing(pose)) if torso else 0.0
        mf = float(mesh_facing(np.array(f['mesh']['kp']))[0]) if f['mesh'] else 0.0
        out.append((f['t'], lf, mf, f['face'] is not None, f['body'], f['fitError']))
    return out
if __name__ == '__main__':
    d = sys.argv[1]; names = sys.argv[2:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, '*.json')))
    for name in names:
        tr = facing_track(os.path.join(d, name + '.json'))
        both = [(lf, mf) for _, lf, mf, *_ in tr if lf and mf]
        agree = np.mean([lf == mf for lf, mf in both]) if both else float('nan')
        flips = sum(1 for a, b in zip(tr, tr[1:]) if a[2] and b[2] and a[2] != b[2])
        lflips = sum(1 for a, b in zip(tr, tr[1:]) if a[1] and b[1] and a[1] != b[1])
        print(f'{name:14s} frames {len(tr):4d} mesh {sum(1 for x in tr if x[2]):4d} label=mesh {agree:.2f} mesh flips {flips:3d} label flips {lflips:3d} '
              f'mesh back {np.mean([x[2] < 0 for x in tr if x[2]]) if any(x[2] for x in tr) else 0:.2f} face {np.mean([x[3] for x in tr]):.2f}')

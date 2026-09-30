"""Export a trained body-sense model for the browser.

  python export.py model.pt cmu37.npz

Writes public/dense/body-sense.{bin,json} (fp16 weights) and
tests/data/body-sense-parity.json: simulated frames with the features and
last-frame outputs PyTorch computes from the same fp16 weights, so the JS
runtime can be checked end to end.
"""
import hashlib, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import simulate as sim
from train import BodySense

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ckpt = torch.load(sys.argv[1], map_location='cpu')
model = BodySense(**{k: v for k, v in ckpt['config'].items() if k in ('features', 'channels', 'dilations')}, lift=ckpt['config'].get('lift', False))
sim.FACE_POINTS = ckpt['config'].get('facePoints', True)          # parity features as this model was trained
model.load_state_dict(ckpt['state'])
tensors, blobs, offset = [], [], 0
for name, value in model.state_dict().items():
    half = value.detach().float().numpy().astype('<f2')
    value.copy_(torch.from_numpy(half.astype(np.float32)))          # reference uses the shipped precision
    tensors.append(dict(name=name, shape=list(value.shape), offset=offset, length=int(half.size)))
    blobs.append(half.tobytes()); offset += half.size
data = b''.join(blobs)
os.makedirs(os.path.join(ROOT, 'public', 'dense'), exist_ok=True)
NAME = 'body-lift' if ckpt['config'].get('lift') else 'body-sense'
sim.NET_JOINTS = bool(ckpt['config'].get('netJoints')); sim.FEATURES = ckpt['config']['features']
open(os.path.join(ROOT, 'public', 'dense', NAME + '.bin'), 'wb').write(data)
manifest = dict(format='body-sense-v1', config=ckpt['config'], tensors=tensors, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                parameters=int(sum(t['length'] for t in tensors)), trainedSteps=ckpt['step'],
                validation={k: v for k, v in ckpt['val'].items()},
                source='Trained in ml/body-sense on CMU Graphics Lab motion capture (mocap.cs.cmu.edu), synthetic cameras and simulated MediaPipe errors.')
json.dump(manifest, open(os.path.join(ROOT, 'public', 'dense', NAME + '.json'), 'w'), indent=1)
print('weights', len(data), 'bytes;', manifest['parameters'], 'parameters')

# Parity vectors: 6 simulated windows without missing frames.
model.eval()
bank = sim.MotionBank(sys.argv[2], torch.device('cpu'), holdout_subjects=range(140, 150))
cases = []
b = sim.simulate(bank, 64, 'val', seed=4242, raw=True)
with torch.no_grad(): out = model(b['features'])
r = b['raw']
for i in [int(k) for k in torch.nonzero((b['weight'] > 0) & ~r['missing'].any(1)).flatten()[:6]]:
    W, H = float(r['width'][i]), float(r['height'][i])
    frames = []
    for t in range(sim.T):
        pose = [dict(x=float(r['x'][i, t, j]), y=float(r['y'][i, t, j]), visibility=float(r['visibility'][i, t, j]), presence=float(r['presence'][i, t, j])) for j in range(33)]
        face = r['face'][i, t].tolist(); net = r['network'][i, t].tolist()
        frames.append(dict(pose=pose, dt=float(r['dt'][i, t]), face=dict(forward=face[1:]) if face[0] else None,
                           network=dict(forward=net[1:], **({'joints': r['network_joints'][i, t].flatten().tolist()} if r.get('network_joints') is not None else {})) if net[0] else None,
                           features=b['features'][i, t].tolist()))
    o = {k: v[i].tolist() for k, v in out.items()}
    expected = dict(forward=o['forward'], back=float(torch.sigmoid(out['back'][i])), swap=torch.sigmoid(out['swap'][i]).tolist(), joints=o['joints'])
    if 'joints3d' in out: expected.update(joints3d=o['joints3d'], sigma3d=torch.exp(out['logsig'][i]).tolist())
    cases.append(dict(width=W, height=H, frames=frames, expected=expected))
json.dump(dict(cases=cases), open(os.path.join(ROOT, 'tests', 'data', NAME + '-parity.json'), 'w'))
print('parity cases', len(cases))

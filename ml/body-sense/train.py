"""Train the body-sense model on synthetic windows (simulate.py).

  python train.py cmu37.npz OUT_DIR [--steps 24000] [--channels 128]

Held-out CMU subjects 140-149 are only used for validation. Writes
OUT_DIR/model.pt (best validation loss) and OUT_DIR/log.jsonl.
"""
import argparse, json, math, os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import simulate as sim

HOLDOUT = range(140, 150)


class Block(nn.Module):
    def __init__(self, channels, dilation, kernel=3):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.proj = nn.Linear(channels, channels)
        self.pad = (kernel - 1) * dilation

    def forward(self, h):                              # (B,T,C)
        n = F.pad(self.norm(h).transpose(1, 2), (self.pad, 0))
        return h + self.proj(F.gelu(self.conv(n), approximate='tanh').transpose(1, 2))


class BodySense(nn.Module):
    """Causal temporal convolution; the output for the last frame sees 31 frames."""
    def __init__(self, features=sim.FEATURES, channels=128, dilations=(1, 2, 4, 8), lift=sim.NET_JOINTS):
        super().__init__()
        self.lift = lift
        self.config = dict(features=features, channels=channels, dilations=list(dilations), kernel=3, facePoints=sim.FACE_POINTS,
                           netJoints=lift, lift=lift)
        self.inp = nn.Linear(features, channels)
        self.blocks = nn.ModuleList(Block(channels, d) for d in dilations)
        self.norm = nn.LayerNorm(channels)
        # forward, back logit, swap logits, 2D joints [, 3D joints, log-sigma per joint]
        self.head = nn.Linear(channels, 3 + 1 + 16 + 66 + (99 + 33 if lift else 0))

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks: h = block(h)
        out = self.head(self.norm(h[:, -2:]))           # last two frames (the earlier one for the velocity loss)
        now = out[:, -1]
        pred = dict(forward=F.normalize(now[:, :3], dim=-1), back=now[:, 3], swap=now[:, 4:20], joints=now[:, 20:86].view(-1, 33, 2))
        if self.lift:
            pred['joints3d'] = now[:, 86:185].view(-1, 33, 3)
            pred['logsig'] = now[:, 185:218].clamp(-5, 1)
            pred['joints3d_prev'] = out[:, -2, 86:185].view(-1, 33, 3)
        return pred


def losses(pred, batch):
    w = batch['weight']; norm = w.sum().clamp_min(1)
    forward = ((1 - (pred['forward'] * batch['forward']).sum(-1)) * w).sum() / norm
    back = (F.binary_cross_entropy_with_logits(pred['back'], batch['back'], reduction='none') * w).sum() / norm
    swap = (F.binary_cross_entropy_with_logits(pred['swap'], batch['swap'], reduction='none').mean(-1) * w).sum() / norm
    joints = ((pred['joints'] - batch['joints']).abs().sum(-1).mean(-1) * w).sum() / norm
    out = dict(forward=forward, back=back, swap=swap, joints=joints, total=forward + back + swap + .3 * joints)
    if 'joints3d' in pred:
        # Gaussian NLL per joint: the model says how sure it is (hidden joints get a wider sigma).
        d2 = ((pred['joints3d'] - batch['joints3d']) ** 2).sum(-1)
        sig2 = torch.exp(2 * pred['logsig'])
        nll = (d2 / (2 * sig2) + 3 * pred['logsig']).mean(-1)
        l1 = (pred['joints3d'] - batch['joints3d']).norm(dim=-1).mean(-1)
        # No jitter: the frame-to-frame change should match the true motion.
        vel = ((pred['joints3d'] - pred['joints3d_prev']) - (batch['joints3d'] - batch['joints3d_prev'])).norm(dim=-1).mean(-1)
        out.update(lift_nll=(nll * w).sum() / norm, lift_l1=(l1 * w).sum() / norm, lift_vel=(vel * w).sum() / norm)
        out['total'] = out['total'] + .1 * out['lift_nll'] + 2 * out['lift_l1'] + 2 * out['lift_vel']
    return out


@torch.no_grad()
def evaluate(model, bank, batches=8, size=2048):
    model.eval()
    acc = {k: [] for k in ('back_acc', 'back_front', 'back_side', 'back_back', 'labels_back_acc', 'net_back_acc',
                           'shoulder_swap_acc', 'all_pairs_acc', 'no_fix_all_pairs_acc', 'angle_deg', 'joint_err_hidden', 'joint_err_visible', 'loss',
                           'mpjpe_mm', 'mpjpe_vis_mm', 'mpjpe_hid_mm', 'mpjpe_tight_mm', 'net_mpjpe_mm', 'accel_mm', 'lift_back_acc')}
    for i in range(batches):
        b = sim.simulate(bank, size, 'val', seed=10_000 + i)
        p = model(b['features'])
        m = b['weight'] > 0
        truth = b['back'] > .5; guess = p['back'] > 0
        cos = b['cos_face']
        acc['loss'].append(losses(p, b)['total'].item())
        acc['back_acc'].append((guess == truth)[m].float().mean().item())
        for name, sel in (('back_front', cos > .5), ('back_side', cos.abs() <= .5), ('back_back', cos < -.5)):
            acc[name].append((guess == truth)[m & sel].float().mean().item())
        # Baselines: MediaPipe labels as given; the surface network alone.
        pose = b['features'][:, -1, :132].view(-1, 33, 4)
        lr = (pose[:, 11, :2] + pose[:, 23, :2]) / 2 - (pose[:, 12, :2] + pose[:, 24, :2]) / 2
        up = (pose[:, 11, :2] + pose[:, 12, :2]) / 2 - (pose[:, 23, :2] + pose[:, 24, :2]) / 2
        labels_back = (lr[:, 0] * up[:, 1] - lr[:, 1] * up[:, 0]) > 0
        acc['labels_back_acc'].append((labels_back == truth)[m].float().mean().item())
        net_back = b['net_forward'][:, 2] > 0
        acc['net_back_acc'].append((net_back == truth)[m & b['net_on']].float().mean().item())
        swap_true = b['swap'] > .5; swap_guess = p['swap'] > 0
        acc['shoulder_swap_acc'].append((swap_guess[:, 5] == swap_true[:, 5])[m].float().mean().item())
        acc['all_pairs_acc'].append((swap_guess == swap_true).all(-1)[m].float().mean().item())
        acc['no_fix_all_pairs_acc'].append((~swap_true).all(-1)[m].float().mean().item())
        acc['angle_deg'].append(torch.rad2deg(torch.acos((p['forward'] * b['forward']).sum(-1).clamp(-1, 1)))[m].mean().item())
        err = (p['joints'] - b['joints']).norm(dim=-1)
        vis = b['visible'] & m[:, None]; hid = ~b['visible'] & m[:, None]
        acc['joint_err_hidden'].append(err[hid].mean().item()); acc['joint_err_visible'].append(err[vis].mean().item())
        if 'joints3d' in p:
            mm = 500.0                                   # torso length ~0.5 m
            e3 = (p['joints3d'] - b['joints3d']).norm(dim=-1) * mm
            acc['mpjpe_mm'].append(e3[m].mean().item())
            acc['mpjpe_vis_mm'].append(e3[vis].mean().item()); acc['mpjpe_hid_mm'].append(e3[hid].mean().item())
            acc['mpjpe_tight_mm'].append(e3[m & b['tight']].mean().item() if (m & b['tight']).any() else float('nan'))
            v = ((p['joints3d'] - p['joints3d_prev']) - (b['joints3d'] - b['joints3d_prev'])).norm(dim=-1) * mm
            acc['accel_mm'].append(v[m].mean().item())
            # Baseline: the network's own (delayed, sometimes mirrored) joints as given.
            nj = b['features'][:, -1, sim.FEATURES_V8:].view(-1, len(sim.NET_JOINT_IDS), 3)
            ids = torch.tensor(sim.NET_JOINT_IDS, device=nj.device)
            en = (nj - b['joints3d'][:, ids]).norm(dim=-1).mean(-1) * mm
            acc['net_mpjpe_mm'].append(en[m & b['net_on']].mean().item())
            ep = (p['joints3d'][:, ids] - b['joints3d'][:, ids]).norm(dim=-1).mean(-1) * mm
            acc.setdefault('lift_mpjpe_net_joints_mm', []).append(ep[m & b['net_on']].mean().item())
            tf = sim._torso_forward(p['joints3d'][:, None])[:, 0]
            acc['lift_back_acc'].append(((tf[:, 2] > 0) == truth)[m].float().mean().item())
    model.train()
    return {k: round(sum(v) / len(v), 4) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=24000); ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--channels', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--eval-every', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(0)
    bank = sim.MotionBank(args.data, dev, holdout_subjects=HOLDOUT)
    model = BodySense(channels=args.channels).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=.05)
    log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    print(f'{params} parameters on {dev}', flush=True)
    best, began, running = float('inf'), time.time(), {}
    for step in range(1, args.steps + 1):
        batch = sim.simulate(bank, args.batch, 'train', seed=step)
        l = losses(model(batch['features']), batch)
        opt.zero_grad(set_to_none=True); l['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for k, v in l.items(): running[k] = running.get(k, 0) + v.item()
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, bank)
            row = dict(step=step, minutes=round((time.time() - began) / 60, 1), lr=sched.get_last_lr()[0],
                       train={k: round(v / args.eval_every, 4) for k, v in running.items()}, val=val)
            running = {}
            log.write(json.dumps(row) + '\n'); log.flush()
            print(json.dumps(row), flush=True)
            if val['loss'] < best:
                best = val['loss']
                torch.save(dict(state=model.state_dict(), config=model.config, step=step, val=val), os.path.join(args.out, 'model.pt'))


if __name__ == '__main__':
    main()

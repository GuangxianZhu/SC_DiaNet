"""Gain-compensated Btanh: per-layer state-count multipliers fitted so that the SC activation
magnitude matches the float model layer by layer (validation data only). Zero hardware cost."""
import os, sys, json, torch, sc
from train import get_data
from dianet import PatchDiaNet, align_jump
torch.set_num_threads(int(os.environ.get('NT', 4)))

def float_hid(net, x):
    sp = net.spec; seg = net.segments(x); hid = []
    for i in range(len(sp.shapes)):
        if i == 0: z = seg[0]
        elif i < sp.red_dep: z = x.clone(); z[..., sp.inserts[i]] += seg[i]
        else: z = x
        pre = torch.einsum('bgi,goi->bgo', z, net.W(i))
        if i >= 2:
            j = hid[i-2]; a, b = align_jump(pre, j); pre = pre.clone(); pre[..., a] += j[..., b]
        x = torch.tanh(net.gain * pre); hid.append(x)
    return hid

def fit(model, x, L, iters=4):
    with torch.no_grad():
        fs = float_hid(model.sub, x); fm = float_hid(model.main, fs[-1][..., 0::2].reshape(len(x), 1, -1))
    fl = [f[..., 0::2].flatten() for f in fs + fm]
    ns = len(model.sub.spec.shapes)
    rs = [1.0] * len(fl)
    orig = sc.btanh
    for it in range(iters):
        model.sub.sc_rs, model.main.sc_rs = rs[:ns], rs[ns:]
        rec = []
        sc.btanh = lambda c, m, r: (lambda o: (rec.append(o.mean(-1) * 2 - 1), o)[1])(orig(c, m, r))
        g = torch.Generator().manual_seed(it)
        with torch.no_grad(): sc.sc_patchdianet(model, x, L, g, 1.0)
        sc.btanh = orig
        slopes = []
        for k, (f, s_) in enumerate(zip(fl, rec)):
            s_ = s_[..., 0::2].flatten()
            slope = ((f * s_).sum() / (s_ * s_).sum()).item()  # how much SC output must be scaled to match float
            slopes.append(slope)
            rs[k] = float(min(max(rs[k] * slope ** 0.7, 0.25), 8.0))
        print('iter', it, 'slopes', [round(v, 2) for v in slopes], flush=True)
    return rs

if __name__ == '__main__':
    path, L = sys.argv[1], int(sys.argv[2])
    sc.DECOR = os.environ.get('DECOR') or None
    _, (xv, yv), _ = get_data('dianet')
    m = PatchDiaNet(); m.load_state_dict(torch.load(path)); m.eval()
    rs = fit(m, xv[:100], L)
    json.dump(rs, open(f'gaincal_{os.path.basename(path)}_{sc.DECOR}_L{L}.json', 'w'))
    print([round(v, 2) for v in rs])

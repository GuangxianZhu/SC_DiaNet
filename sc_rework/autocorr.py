"""Per-layer lag-1 autocorrelation of SC activations and variance inflation vs i.i.d. bits."""
import os, sys, json, torch, sc
from train import get_data
from dianet import DeepMLP
torch.set_num_threads(4)
sc.RN = True
d = int(sys.argv[1]); L = int(sys.argv[2])
_, (xv, yv), _ = get_data('deep')
m = DeepMLP(d); m.load_state_dict(torch.load(f'deep{d}.pt')); m.eval()
x = xv[:100]
out = {}
for dec in [None, 'perm']:
    sc.DECOR = dec
    rec = []; orig = sc.btanh
    sc.btanh = lambda c, mm, r: (lambda o: (rec.append(o), o)[1])(orig(c, mm, r))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad(): sc.sc_deepmlp(m, x, L, g, 1.0)
    sc.btanh = orig
    rows = []
    for k, o in enumerate(rec):
        # o: bits at the Btanh output (before decorrelation is applied) -> what the next layer sees
        # is post-decorrelation; recompute what the next layer sees:
        b = o
        if dec == 'perm':
            b = torch.gather(o, -1, torch.rand(o.shape).argsort(-1))
        p = b.mean(-1, keepdim=True)
        c = b - p
        var = (c * c).mean(-1)
        lag1 = ((c[..., 1:] * c[..., :-1]).mean(-1) / var.clamp(min=1e-6))
        ok = (var > 0.05)
        # variance inflation of block means (block = 32 bits) vs iid
        blk = b[..., : L // 32 * 32].reshape(*b.shape[:-1], -1, 32).mean(-1)
        vin = blk.var(-1) / (p[..., 0] * (1 - p[..., 0]) / 32).clamp(min=1e-6)
        rows.append((round(lag1[ok].mean().item(), 3), round(vin[ok].mean().item(), 2)))
    out[str(dec)] = rows
    print(dec, 'layer: (lag1 autocorr, block-variance inflation)', rows, flush=True)
json.dump(out, open(f'autocorr_deep{d}_L{L}.json', 'w'))

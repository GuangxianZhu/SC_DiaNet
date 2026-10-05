"""SC accuracy vs bitstream length for PatchDiaNet and FCNN.

Usage: python eval_sc.py {dianet|fcnn} model.pt N_TEST [L,L,...]
r_scale (Btanh state count multiplier) is picked on 500 validation images.
"""
import os, sys, time, json
import torch
from train import get_data, accuracy
from dianet import PatchDiaNet, FCNN, DeepMLP
import sc
from sc import sc_patchdianet, sc_fcnn, sc_deepmlp
sc.DECOR = os.environ.get('DECOR') or None

torch.set_num_threads(int(os.environ.get("NT", 4)))


def sc_acc(fn, model, x, y, L, seed, r_scale, bs):
    g = torch.Generator().manual_seed(seed)
    correct = 0
    for i in range(0, len(x), bs):
        correct += (fn(model, x[i:i + bs], L, g, r_scale).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def main():
    kind, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    Ls = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [32, 64, 128, 256, 512, 1024]
    _, (xva, yva), (xte, yte) = get_data(kind)
    model = DeepMLP(int(os.environ['DEPTH']), int(os.environ.get('WIDTH', 32))) if kind == 'deep' else PatchDiaNet() if kind == 'dianet' else FCNN(tuple(int(d) for d in os.environ.get('DIMS', '784,100,200,10').split(',')))
    model.load_state_dict(torch.load(path)); model.eval()
    fn = sc_patchdianet if kind == 'dianet' else sc_deepmlp if kind == 'deep' else sc_fcnn
    bs = 50 if kind == 'dianet' else 25
    res = {'float_test': accuracy(model, xte, yte), 'float_test_subset': accuracy(model, xte[:n], yte[:n])}
    print(res, flush=True)
    cal = {}
    for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
        cal[s] = sc_acc(fn, model, xva[:500], yva[:500], 256, 123, s, bs)
        print('calib r_scale', s, cal[s], flush=True)
    s = max(cal, key=cal.get)
    res['r_scale'] = s
    for L in Ls:
        t = time.time()
        res[f'L{L}'] = sc_acc(fn, model, xte[:n], yte[:n], L, 7, s, bs)
        print(f'L={L} acc={res[f"L{L}"]:.4f} ({time.time()-t:.0f}s)', flush=True)
    json.dump(res, open(f'res_{os.path.basename(path)}_{sc.DECOR}_rn{int(sc.RN)}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

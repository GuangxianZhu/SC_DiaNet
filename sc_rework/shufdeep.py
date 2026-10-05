import sys, torch, sc
from train import get_data
from dianet import DeepMLP
torch.set_num_threads(4); sc.RN = True
d = int(sys.argv[1])
_, _, (xt, yt) = get_data('deep')
m = DeepMLP(d); m.load_state_dict(torch.load(f'deep{d}.pt')); m.eval()
x, y = xt[:500], yt[:500]
for mode in sys.argv[2].split(','):
    sc.DECOR = None if mode == 'none' else mode
    D = int(mode[4:]) if mode.startswith('shuf') else 0
    sc.WARM = d * D
    for L in [256, 1024]:
        g = torch.Generator().manual_seed(0)
        acc = sum((sc.sc_deepmlp(m, x[i:i+100], L + sc.WARM, g, 1.0).argmax(1) == y[i:i+100]).sum().item() for i in range(0, 500, 100)) / 500
        print(d, mode, L, 'warm', sc.WARM, acc, flush=True)

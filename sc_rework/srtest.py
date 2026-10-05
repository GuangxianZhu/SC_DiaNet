import sys, torch, sc
from train import get_data
from dianet import DeepMLP
torch.set_num_threads(4); sc.RN = True
d = int(sys.argv[1])
_, (xv, yv), (xt, yt) = get_data('deep')
m = DeepMLP(d); m.load_state_dict(torch.load(f'deep{d}.pt')); m.eval()
for sr in [False, True]:
    sc.SR = sr
    for rs in [0.5, 1, 2, 4]:
        g = torch.Generator().manual_seed(0)
        acc = (sc.sc_deepmlp(m, xv[:300], 256, g, rs).argmax(1) == yv[:300]).float().mean().item()
        print(d, 'SR', sr, 'r_scale', rs, 'val@256', round(acc, 3), flush=True)

"""Train PatchDiaNet / FCNN on MNIST with a held-out validation split.

Model selection uses the last 5k training images, never the test set.
Usage: python train.py {dianet|fcnn} EPOCHS OUT.pt [seed] [gain] [noise_L]
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from dianet import PatchDiaNet, FCNN

torch.set_num_threads(int(os.environ.get("NT", 4)))


INPUT = os.environ.get('INPUT', 'gray')  # gray: [0,1]; bin: {0,1}; bip: {-1,1}


def load(train):
    ds = torchvision.datasets.MNIST('data', train=train, download=True)
    x = ds.data.float() / 255.0
    if INPUT == 'raw':
        x = x * 255
    if INPUT in ('bin', 'bip'):
        x = (x > 0.3).float()
        if INPUT == 'bip':
            x = 2 * x - 1
    return x, ds.targets


def to_patches(img, k=4, s=2):
    """25x25 center crop (rows/cols 2..26), k x k patches with stride s -> (N, P, k*k)."""
    x = img[:, 2:27, 2:27].unsqueeze(1)
    return F.unfold(x, kernel_size=k, stride=s).transpose(1, 2).contiguous()


def get_data(kind):
    xtr, ytr = load(True)
    xte, yte = load(False)
    if kind == 'dianet':
        xtr, xte = to_patches(xtr), to_patches(xte)
    return (xtr[:55000], ytr[:55000]), (xtr[55000:], ytr[55000:]), (xte, yte)


def accuracy(model, x, y, bs=1000):
    model.eval()
    with torch.no_grad():
        return sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
                   for i in range(0, len(x), bs)) / len(x)


def main():
    kind, epochs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    gain = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    noise_L = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    torch.manual_seed(seed)
    (xtr, ytr), (xva, yva), (xte, yte) = get_data(kind)
    model = PatchDiaNet() if kind == 'dianet' else FCNN(tuple(int(d) for d in os.environ.get('DIMS', '784,100,200,10').split(',')))
    if kind == 'dianet':
        model.set_sc(gain, noise_L)
    else:
        model.noise_L = noise_L
    if os.environ.get('INIT'):
        model.load_state_dict(torch.load(os.environ['INIT']))
    print(f'gain {gain} noise_L {noise_L}')
    print('params(nonzero):', sum(int((p != 0).sum()) for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = 0
    for ep in range(epochs):
        model.train(); t = time.time()
        perm = torch.randperm(len(xtr))
        tot = 0
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); model.clamp_()
            tot += loss.item() * len(idx)
        sched.step()
        va = accuracy(model, xva, yva)
        if va > best:
            best = va
            torch.save(model.state_dict(), out)
        print(f'ep {ep} loss {tot/len(xtr):.4f} val {va:.4f} best {best:.4f} {time.time()-t:.0f}s', flush=True)
    model.load_state_dict(torch.load(out))
    print(f'FINAL val {best:.4f} test {accuracy(model, xte, yte):.4f}')


if __name__ == '__main__':
    main()

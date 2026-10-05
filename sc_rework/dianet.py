"""Batched (grouped) re-implementation of Script_DiaNet + float/SC forward.

Semantics follow DiaNet_SC_II/patch dianet mnist/11x11x1_4x4_wider.ipynb exactly
(masks, input-segment insertion into odd slots, jump connection from layer i-2,
tanh activation, weights clamped to [-1, 1], output = even slots of last layer).
"""
import os
import torch
import torch.nn as nn
from orig_dianet import Script_DiaNet


class Spec:
    """Topology extracted from the original generator."""

    def __init__(self, inp, out):
        ref = Script_DiaNet(inp, out)
        self.inp, self.out = inp, out
        self.red_dep, self.red_full = ref.red_dep, ref.red_full
        self.red_triangle = ref.red_triangle
        self.shapes = [tuple(l.weight.shape) for l in ref.layers]  # (out, in)
        self.masks = [m.clone() for m in ref.masks]
        self.select_insert = ref.select_insert
        # segment slices of the padded input
        self.segs, s = [], 0
        for i in range(self.red_dep, 0, -1):
            n = i * 2 + 1 if i == self.red_dep else i
            self.segs.append((s, s + n))
            s += n
        self.inserts = [None]
        for i in range(1, self.red_dep):
            self.inserts.append(self.select_insert(self.shapes[i][1], self.red_triangle[i]))


def align_jump(pre, jump):
    """Return (slice_of_pre, slice_of_jump) used by the original jump add."""
    if pre.shape[-1] > jump.shape[-1]:
        return (slice(2, -2), slice(None))
    if pre.shape[-1] == jump.shape[-1]:
        return (slice(None), slice(None))
    return (slice(None), slice(2, -2))


def sc_noise(a, L, training):
    """Add the sampling noise of decoding an L-bit bipolar stream of value a: var = (1-a^2)/L."""
    if not training or not L:
        return a
    return a + torch.randn_like(a) * ((1 - a.detach() ** 2).clamp(min=0) / L).sqrt()


class GroupDiaNet(nn.Module):
    """G independent DiaNets with identical topology, evaluated in parallel."""

    def __init__(self, spec: Spec, groups: int):
        super().__init__()
        self.spec, self.G = spec, groups
        self.gain = 1.0      # tanh(gain * pre); realised in SC by Btanh state count r = 2*gain*m
        self.noise_L = 0     # >0: SC-aware training, inject bitstream noise of an L-bit stream
        self.weights = nn.ParameterList()
        for (o, i), m in zip(spec.shapes, spec.masks):
            w = torch.empty(groups, o, i)
            bound = 1 / i ** 0.5  # nn.Linear default init range
            nn.init.uniform_(w, -bound, bound)
            self.weights.append(nn.Parameter(w * m))
        for k, m in enumerate(spec.masks):
            self.register_buffer(f'mask{k}', m)

    def mask(self, k):
        return getattr(self, f'mask{k}')

    def W(self, k):
        return self.weights[k] * self.mask(k)

    @torch.no_grad()
    def clamp_(self):
        for p in self.weights:
            p.clamp_(-1, 1)

    def segments(self, x):
        sp = self.spec
        cont = x.new_zeros(*x.shape[:-1], sp.red_full)
        cont[..., : x.shape[-1]] = x
        return [cont[..., a:b] for a, b in sp.segs]

    def forward(self, x):  # x: (B, G, inp)
        sp = self.spec
        seg = self.segments(x)
        hid = []
        for i in range(len(sp.shapes)):
            if i == 0:
                z = seg[0]
            elif i < sp.red_dep:
                z = x.clone()
                z[..., sp.inserts[i]] = z[..., sp.inserts[i]] + seg[i]
            else:
                z = x
            pre = torch.einsum('bgi,goi->bgo', z, self.W(i))
            if i >= 2:
                jump = hid[i - 2]
                a, b = align_jump(pre, jump)
                pre = pre.clone()
                pre[..., a] = pre[..., a] + jump[..., b]
            x = sc_noise(torch.tanh(self.gain * pre), self.noise_L, self.training)
            hid.append(x)
        return x[..., 0::2]


class PatchDiaNet(nn.Module):
    """121 patch sub-DiaNets (16 -> 1) feeding a main DiaNet (121 -> 10)."""

    def __init__(self, patch_num=121, patch_size=16, patch_out=1, classes=10):
        super().__init__()
        self.sub = GroupDiaNet(Spec(patch_size, patch_out), patch_num)
        self.main = GroupDiaNet(Spec(patch_num * patch_out, classes), 1)

    def clamp_(self):
        self.sub.clamp_(); self.main.clamp_()

    def set_sc(self, gain, noise_L):
        for n in (self.sub, self.main):
            n.gain, n.noise_L = gain, noise_L

    def forward(self, x):  # (B, P, S)
        x = sc_noise(x, self.sub.noise_L, self.training)
        h = self.sub(x).reshape(x.shape[0], 1, -1)
        return self.main(h)[:, 0]


class FCNN(nn.Module):
    """Baseline from SC inference/inf_1_clp0.5.ipynb (no bias, tanh, clamped w)."""

    def __init__(self, dims=(784, 100, 200, 10)):
        super().__init__()
        self.ls = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.noise_L = 0
        self.wmax = float(os.environ.get('CLAMP', 1.0))  # original recipe: clamp 0.5, SC uses 2*w

    @torch.no_grad()
    def clamp_(self):
        for l in self.ls:
            l.weight.clamp_(-self.wmax, self.wmax)

    def forward(self, x):
        x = sc_noise(x.flatten(1), self.noise_L, self.training)
        for l in self.ls:
            x = sc_noise(torch.tanh(l(x)), self.noise_L, self.training)
        return x

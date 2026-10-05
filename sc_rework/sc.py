"""Bit-level stochastic-computing (SC) inference for PatchDiaNet and FCNN.

Bipolar encoding (v -> P(1) = (v+1)/2), XNOR multipliers, an accumulative
parallel counter (APC) per neuron, and the APC-driven saturating-counter
tanh (Btanh, Kim et al., DAC 2016) as activation. Random numbers come from
torch's RNG (ideal independent SNGs); LFSR effects are not modelled here.
"""
import torch
from dianet import align_jump

import os
RN = bool(int(os.environ.get('RN', '0')))  # per-neuron weight range normalisation
WARM = 0  # cycles discarded at the output decoder (pipeline fill of decorrelation buffers)
DECOR = None  # None | 'perm' (ideal time permutation) | 'shufD' (D-entry shuffle buffer) | 'regen' (decode + re-SNG)


def shuffle_buffer(bits, D):
    """Hardware-realistic decorrelator: a D-entry buffer per stream; each cycle a random
    slot is read out and overwritten by the incoming bit (index from a shared RNG)."""
    shp, L = bits.shape[:-1], bits.shape[-1]
    buf = (torch.rand(*shp, D) < 0.5).float()
    out = torch.empty_like(bits)
    for t in range(L):
        k = torch.randint(D, (*shp, 1))
        out[..., t] = torch.gather(buf, -1, k)[..., 0]
        buf.scatter_(-1, k, bits[..., t:t + 1])
    return out


def sng(v, L, gen):
    """Bipolar stochastic number generator: (...,) values in [-1,1] -> (..., L) 0/1 float."""
    p = (v.clamp(-1, 1) + 1) / 2
    return (torch.rand(*v.shape, L, generator=gen) < p.unsqueeze(-1)).float()


def btanh(count, m, r):
    """count: (..., O, L) APC outputs, m: (O,) operand count, r: (O,) state count (even).

    State moves by 2*count - m each cycle, saturates in [0, r-1], output 1 in the upper half.
    """
    L = count.shape[-1]
    step = 2 * count - m.unsqueeze(-1)
    state = (r // 2).expand(count.shape[:-1]).clone().float()
    out = torch.empty_like(count)
    rmax = (r - 1).float()
    half = (r // 2).float()
    for t in range(L):
        state = torch.minimum(torch.clamp(state + step[..., t], min=0), rmax)
        out[..., t] = (state >= half).float()
    return out


def sc_layer(z, Wbits, M, jump_bits=None, jump_map=None, r_scale=1.0, wscale=None):
    """One SC neuron layer.

    z: (B, G, J, L) input bits; Wbits: (G, O, J, L) weight bits; M: (G or 1, O, J) operand mask.
    jump_bits: (B, G, O, L) bits added as an extra unit-weight operand where jump_map[o] is True.
    """
    Mf = M.float()
    # XNOR(w, z) = 1 - w - z + 2wz, summed over active operands
    cnt = torch.einsum('goj,gojt->got', Mf, 1 - Wbits).unsqueeze(0)
    cnt = cnt - torch.einsum('goj,bgjt->bgot', Mf, z)
    cnt = cnt + 2 * torch.einsum('bgjt,gojt->bgot', z, Wbits * Mf.unsqueeze(-1))
    m = Mf.sum(-1)  # (G, O)
    if jump_bits is not None:
        cnt = cnt + jump_bits * jump_map.float().view(1, 1, -1, 1)
        m = m + jump_map.float()
    m = m[0] if m.shape[0] == 1 else m
    gain = r_scale if wscale is None else r_scale * (wscale[0] if wscale.shape[0] == 1 else wscale)
    r = 2 * torch.clamp(torch.round(gain * m), min=1)
    out = btanh(cnt, m, r)
    if DECOR == 'perm':
        idx = torch.rand(out.shape).argsort(-1)
        out = torch.gather(out, -1, idx)
    elif DECOR and DECOR.startswith('shuf'):
        out = shuffle_buffer(out, int(DECOR[4:]))
    elif DECOR == 'regen':
        out = (torch.rand(out.shape) < out.mean(-1, keepdim=True)).float()
    return out


@torch.no_grad()
def sc_groupdianet(net, xbits, L, gen, r_scale=1.0):
    """xbits: (B, G, inp, L) bits of the inputs; returns bits of even outputs."""
    sp = net.spec
    B, G = xbits.shape[:2]
    # padded container: padding is a structural zero, never an operand
    cont = torch.zeros(B, G, sp.red_full, L)
    cont[:, :, : sp.inp] = xbits
    real = torch.zeros(sp.red_full, dtype=torch.bool)
    real[: sp.inp] = True
    seg = [cont[:, :, a:b] for a, b in sp.segs]
    seg_real = [real[a:b] for a, b in sp.segs]
    hid, x, x_real = [], None, None
    for i in range(len(sp.shapes)):
        O, J = sp.shapes[i]
        if i == 0:
            z, z_real = seg[0], seg_real[0]
        else:
            z, z_real = x.clone(), x_real.clone()
            if i < sp.red_dep:
                ins = sp.inserts[i]
                z[:, :, ins] = seg[i]
                z_real[ins] = seg_real[i]
        W = net.W(i).detach()  # (G, O, J)
        M = (net.mask(i).bool() & z_real.view(1, -1)).unsqueeze(0).expand(G, O, J)
        jb = jm = None
        if i >= 2:
            a, b = align_jump(torch.zeros(O), torch.zeros(hid[i - 2].shape[2]))
            jb = torch.zeros(B, G, O, L)
            jb[:, :, a] = hid[i - 2][:, :, b]
            jm = torch.zeros(O, dtype=torch.bool)
            idx = torch.arange(O)[a]
            jm[idx] = True
            jm &= (torch.arange(O) % 2 == 0)  # odd jump sources are structural zeros
        ws = None
        if RN:  # range normalisation: scale each neuron's weights to full range, Btanh r compensates
            ws = (W.abs() * M).amax(-1).clamp(min=1e-3)  # (G, O)
            if jm is not None:
                ws = torch.where(jm.view(1, -1), torch.ones_like(ws), ws)
            W = W / ws.unsqueeze(-1)
        rs_i = r_scale * (net.sc_rs[i] if getattr(net, 'sc_rs', None) is not None else 1.0)
        x = sc_layer(z, sng(W, L, gen), M, jb, jm, rs_i, ws)
        x_real = torch.arange(O) % 2 == 0  # odd neurons have no inputs (always 0)
        hid.append(x)
    return x[:, :, 0::2]


@torch.no_grad()
def sc_patchdianet(model, x, L, gen, r_scale=1.0):
    """x: (B, P, S) pixel values in [0,1]. Returns (B, 10) decoded values."""
    xb = sng(x, L, gen)
    h = sc_groupdianet(model.sub, xb, L, gen, r_scale)  # (B, P, 1, L)
    h = h.reshape(x.shape[0], 1, -1, L)
    out = sc_groupdianet(model.main, h, L, gen, r_scale)[:, 0]
    return out[..., WARM:].mean(-1) * 2 - 1


@torch.no_grad()
def sc_fcnn(model, x, L, gen, r_scale=1.0):
    z = sng(x.flatten(1), L, gen).unsqueeze(1)  # (B, 1, J, L)
    for l in model.ls:
        W = l.weight.detach().unsqueeze(0)
        M = torch.ones_like(W, dtype=torch.bool)
        ws = None
        if RN:
            ws = W.abs().amax(-1).clamp(min=1e-3)
            W = W / ws.unsqueeze(-1)
        z = sc_layer(z, sng(W, L, gen), M, r_scale=r_scale, wscale=ws)
    return z[:, 0].mean(-1) * 2 - 1


@torch.no_grad()
def sc_deepmlp(model, x, L, gen, r_scale=1.0):
    z = sng(x.flatten(1), L, gen).unsqueeze(1)
    n = len(model.ls)
    for k, l in enumerate(model.ls):
        W = l.weight.detach().unsqueeze(0)
        O = W.shape[1]
        M = torch.ones_like(W, dtype=torch.bool)
        jb = jm = None
        if model.skip and 0 < k < n - 1:
            jb, jm = z, torch.ones(O, dtype=torch.bool)
        ws = None
        if RN:
            ws = W.abs().amax(-1).clamp(min=1e-3)
            if jm is not None:
                ws = torch.ones_like(ws)
            W = W / ws.unsqueeze(-1)
        z = sc_layer(z, sng(W, L, gen), M, jb, jm, r_scale, ws)
    return z[:, 0].mean(-1) * 2 - 1

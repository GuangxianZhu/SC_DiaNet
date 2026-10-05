import torch, sys
import sc
from train import get_data
from dianet import PatchDiaNet, align_jump
torch.set_num_threads(2)
_,(xv,yv),_=get_data('dianet')
m=PatchDiaNet(); m.load_state_dict(torch.load(sys.argv[1])); m.eval()
x=xv[:20]; L=int(sys.argv[2]) if len(sys.argv)>2 else 1024
def fl_hid(net,x):
    sp=net.spec; seg=net.segments(x); hid=[]
    for i in range(len(sp.shapes)):
        if i==0: z=seg[0]
        elif i<sp.red_dep: z=x.clone(); z[...,sp.inserts[i]]+=seg[i]
        else: z=x
        pre=torch.einsum('bgi,goi->bgo',z,net.W(i))
        if i>=2:
            j=hid[i-2]; a,b=align_jump(pre,j); pre=pre.clone(); pre[...,a]+=j[...,b]
        x=torch.tanh(net.gain*pre); hid.append(x)
    return hid
with torch.no_grad():
    fs=fl_hid(m.sub,x); h=fs[-1][...,0::2].reshape(20,1,-1); fm=fl_hid(m.main,h)
rec=[]; orig=sc.btanh
def bt(c,mm,r):
    o=orig(c,mm,r); rec.append(o.mean(-1)*2-1); return o
sc.btanh=bt
g=torch.Generator().manual_seed(0)
with torch.no_grad(): out=sc.sc_patchdianet(m,x,L,g,1.0)
for i,(f,s) in enumerate(zip(fs+fm,rec)):
    f=f[...,0::2].flatten(); s=s[...,0::2].flatten()
    print(i,f'corr {torch.corrcoef(torch.stack([f,s]))[0,1].item():.3f} |f| {f.abs().mean().item():.3f} |s| {s.abs().mean().item():.3f}')

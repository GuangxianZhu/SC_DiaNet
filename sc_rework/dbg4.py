import torch, sys, sc
from train import get_data
from dianet import PatchDiaNet
torch.set_num_threads(int(__import__('os').environ.get('NT',2)))
_,_,(xt,yt)=get_data('dianet')
m=PatchDiaNet(); m.load_state_dict(torch.load(sys.argv[1])); m.eval()
n=int(sys.argv[2]); x,y=xt[:n],yt[:n]
import json, os
if os.environ.get('GAINCAL'):
    rs=json.load(open(os.environ['GAINCAL'])); ns=len(m.sub.spec.shapes)
    m.sub.sc_rs, m.main.sc_rs = rs[:ns], rs[ns:]
for mode in sys.argv[3].split(','):
    sc.DECOR=None if mode=='none' else mode
    D=int(mode[4:]) if mode.startswith('shuf') else 0
    sc.WARM=29*D
    for L in [int(v) for v in sys.argv[4].split(',')]:
        g=torch.Generator().manual_seed(0)
        acc=sum((sc.sc_patchdianet(m,x[i:i+50],L+sc.WARM,g,1.0).argmax(1)==y[i:i+50]).sum().item() for i in range(0,n,50))/n
        print(sys.argv[1],mode,L,'warm',sc.WARM,acc,flush=True)

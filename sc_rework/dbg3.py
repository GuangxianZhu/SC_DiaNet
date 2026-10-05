import torch, sys, sc
from train import get_data
from dianet import PatchDiaNet
torch.set_num_threads(2)
_,_,(xt,yt)=get_data('dianet')
m=PatchDiaNet(); m.load_state_dict(torch.load(sys.argv[1])); m.eval()
n=200; x,y=xt[:n],yt[:n]
for mode in [None,'perm','regen']:
    sc.DECOR=mode
    for L in [256,1024]:
        g=torch.Generator().manual_seed(0)
        acc=sum((sc.sc_patchdianet(m,x[i:i+50],L,g,1.0).argmax(1)==y[i:i+50]).sum().item() for i in range(0,n,50))/n
        print(mode,L,acc,flush=True)

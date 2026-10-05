import subprocess, re, json, sys
GE = {'$_AND_':1.33,'$_OR_':1.33,'$_NOT_':0.67,'$_XOR_':2.33,'$_XNOR_':2.33,'$_MUX_':2.33,'$_NAND_':1,'$_NOR_':1,'$_ANDNOT_':1.33,'$_ORNOT_':1.33}
res={}
for m in [int(a) for a in sys.argv[1:]]:
    v=subprocess.run(['python3','gen_neuron.py',str(m),str(2*m)],capture_output=True,text=True).stdout
    open(f'n{m}.v','w').write(v)
    subprocess.run(['yowasp-yosys','-q','-p',f'read_verilog n{m}.v; synth -flatten -noabc -top sc_neuron_m{m}; opt_clean; tee -o st{m}.txt stat'],capture_output=True)
    cells=dict((k,int(n)) for n,k in re.findall(r'^\s+(\d+)\s+(\$_\w+_)',open(f'st{m}.txt').read(),re.M))
    ge=sum(GE.get(k,5.33 if 'DFF' in k else 1.5)*n for k,n in cells.items())
    res[m]={'cells':cells,'GE':round(ge,1)}
    print(m, round(ge,1), cells, flush=True)
json.dump(res,open('neuron_ge.json','w'),indent=1)

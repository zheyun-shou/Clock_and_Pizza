import os, sys
import json
import random, math
import itertools

import numpy as np
import torch
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warning
import warnings
warnings.filterwarnings("ignore")

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from dataset import MyAddDataSet
from model import MyModelA, MyModelB, MyModelC, MyModelD, MyModelX, Transformer

# constants
C=59
DEVICE='cpu'

# model type
model_type='A'

# read all the filename in the specified directory, and extract the runid from the filename
def read_from_json(directory, model_type=None):
    run_ids = []
    for filename in os.listdir(directory):
        if filename.startswith('config_') and filename.endswith('.json'):
            config_path = os.path.join(directory, filename)
            model_path = os.path.join(directory, f'model_{runid}.pt')
            
            if os.path.exists(model_path):
                with open(config_path, 'r') as config_file:
                    config = json.load(config_file)
                    if 'attn_coeff' in config:
                        if abs(config['attn_coeff']) < 1e-6 and model_type == 'A':
                            print(f"Loading model {runid}, Transformer with constant attention")
                        elif abs(config['attn_coeff']) >= 1e-6 and model_type == 'B':
                            print(f"Loading model {runid}, Transformer")
                        run_ids.append(filename[len('config_'):-len('.json')])
                    elif 'model_type' in config:
                        if config['model_type'] in linear_models.keys() and config['model_type'] == model_type.upper():
                            print(f"Loading model {runid}, Model {config['model_type']}")
                            run_ids.append(filename[len('config_'):-len('.json')])

def gradient_symmetricity(model, xs, C=59):
    tt=0
    for abc in xs:
        a,b,c=abc
        x=torch.tensor([[a,b]],device=DEVICE)
        t,o0=model.forward_h(x)
        model.zero_grad()
        #print(a,b,c)
        model.remove_all_hooks()
        o=o0[0,-1,:]
        t.retain_grad()
        o[c].backward(retain_graph=True)
        tg=t.grad[0].detach().cpu().numpy() # target gradient
        #tt+=tg[0][0]
        dp=np.sum(tg[0]*tg[1])/np.sqrt(np.sum(tg[0]**2))/np.sqrt(np.sum(tg[1]**2)) # 
        tt+=dp
    symm = tt/len(xs)
    return symm

def circularity(model, first_k=4):
    we=model.embed.W_E.T
    pca = PCA(n_components=20)
    we2=pca.fit_transform(we.detach().cpu().numpy())
    def ang(x):
        return math.cos(x)+math.sin(x)*1j
    rst=0
    for ix in range(first_k):
        vs=we2[:,ix]*1
        vs=vs/np.sqrt(np.sum(vs*vs))/math.sqrt(59)
        tt=[]
        for i in range(1,59):
            vv=[vs[t*i%59] for t in range(59)]
            sa=sum(vv[t]*ang(2*math.pi*t/59) for t in range(59))
            tt.append((-abs(sa)**2*2,i))
        tt.sort()
        i=tt[0][1]
        rst+=max(min(-tt[0][0],1.),0.)
        i=min(i,59-i)
        v=[vs[t*i%59] for t in range(59)]
    rst/=first_k
    return rst
    
def distance_irrelevance(model, dataloader, show_plot=False):
    oo=[[0]*C for _ in range(C)]
    for x,y in dataloader:
        with torch.inference_mode():
            model.eval()
            model.remove_all_hooks()
            o=model(x)[:,-1,:] # model output
            o0=o[list(range(len(x))),y] # extracts the logit corresponding to the true label y for each sample?
            o0=o0.cpu()
            x=x.cpu()
            for p,q in zip(o0,x):
                A,B=int(q[0].item()),int(q[1].item())
                oo[(A+B)%C][(A-B)%C]=p.item()
    oo=np.array(oo)
    dd=np.mean(np.std(oo,axis=0))/np.std(oo.flatten())
    if show_plot:
        plt.figure(dpi=300)
        sns.heatmap(np.array(oo))
        plt.xlabel(f'(a-b) mod {C}')
        plt.ylabel(f'(a+b) mod {C}')
        plt.title('Correct Logits')
    return dd


if __name__ == '__main__':
    # load the config file
    run_ids = read_from_json(os.path.join(root_path, 'code/save'), model_type='A')
    # runid='A_pretrained'
    runid = run_ids[0]
    with open(os.path.join(root_path, f'code/save/config_{runid}.json'),'r') as f:
        config=json.load(f)

    # load the dataset
    dataset = MyAddDataSet(func=eval(config['funcs']),
                        C=config['C'],
                        diff_vocab=config['diff_vocab'],
                        eqn_sign=config['eqn_sign'],
                        device=DEVICE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)

    # initialize the model
    linear_models={'A':MyModelA,'B':MyModelB,'C':MyModelC,'D':MyModelD,'X':MyModelX}
    if model_type in linear_models:
        model=linear_models[model_type]()
    else:
        model=Transformer(num_layers=config.get('n_layers',1),
                num_heads=config['n_heads'],
                d_model=config['d_model'],
                d_head=config.get('d_head',config['d_model']//config['n_heads']),
                attn_coeff=config['attn_coeff'],
                d_vocab=dataset.vocab,
                act_type=config.get('act_fn','relu'),
                n_ctx=dataset.dim,)
    model.to(DEVICE)

    # pre-generated sequence for gradient symmetricity test
    xs = random.Random(42).sample(list(itertools.product(range(C), repeat=3)), 100)

    for id in run_ids:
        model_file=os.path.join(root_path, f'code/save/model_{id}.pt')
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        print('Gradient Symmetricity', gradient_symmetricity(model, xs))
        print('Circularity', circularity(model))
        print('Distance Irrelevance', distance_irrelevance(model, dataloader, show_plot=False))

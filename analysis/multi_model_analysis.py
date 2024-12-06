import os, sys
import json
import random, math
import itertools

import numpy as np
import torch
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ignore warning
import warnings
warnings.filterwarnings("ignore")

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.utils import extract_embeddings
from analysis.dataset import MyAddDataSet
from analysis.model import MyModelA, MyModelB, MyModelC, MyModelD, MyModelX, Transformer

# constants
C=59
DEVICE='cpu'

# model type
model_type='B'
experiment_dir='_d'
varying_width=True
verbose=True

assert model_type in ['A', 'B', 'alpha', 'beta', 'gama', 'delta', 'x']
type_mapping = {'alpha': 'A', 'beta': 'B', 'gama': 'C', 'delta': 'D', 'x': 'X'}

# read all the filename in the specified directory, and extract the runid from the filename
def read_from_json(directory, model_type=None):
    run_ids, run_types = [], []
    for filename in os.listdir(directory):
        if filename.startswith('config_'+model_type) and filename.endswith('.json'):
            config_path = os.path.join(directory, filename)
            runid = filename[len('config_'):-len('.json')]
            model_path = os.path.join(directory, f'model_{runid}.pt')
            
            if os.path.exists(model_path):
                with open(config_path, 'r') as config_file:
                    config = json.load(config_file)
                    if 'attn_coeff' in config:
                        if abs(config['attn_coeff']) < 1e-6 and model_type == 'A':
                            run_type = "Transformer with constant attention"
                        elif abs(config['attn_coeff']) >= 1e-6 and model_type == 'B':
                            run_type = "Transformer"
                        run_types.append(run_type)
                        run_ids.append(filename[len('config_'):-len('.json')])
                    elif 'model_type' in config:
                        if config['model_type'] == type_mapping[model_type]:
                            run_type = f"Model {config['model_type']}"
                            run_types.append(run_type)
                            run_ids.append(filename[len('config_'):-len('.json')])
    return run_ids, run_types

def gradient_symmetricity(model, xs=None):
    tt=0
    if xs is None:
        xs = random.sample(list(itertools.product(range(C), repeat=3)), 100)
    for abc in xs:
        a,b,c=abc
        x=torch.tensor([[a,b]],device=DEVICE)
        t,o0=model.forward_h(x)
        model.zero_grad()
        #print(a,b,c)
        try: # for transformer
            model.remove_all_hooks()
            o=o0[0,-1,:]
        except: # for linear models
            o=o0[0,:]
        
        t.retain_grad()
        o[c].backward(retain_graph=True)
        tg=t.grad[0].detach().cpu().numpy() # target gradient
        #tt+=tg[0][0]
        dp=np.sum(tg[0]*tg[1])/np.sqrt(np.sum(tg[0]**2))/np.sqrt(np.sum(tg[1]**2)) # 
        tt+=dp
    symm = tt/len(xs)
    return symm

def circularity(model, first_k=4):
    embedding = extract_embeddings(model)
    if model_type in ['A', 'B']:
        we=embedding['embed'].T
    elif model_type in ['alpha', 'beta', 'gama', 'delta']:
        we=embedding['embed']
    pca = PCA(n_components=20)
    we2=pca.fit_transform(we)
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
    
def distance_irrelevance(model, dataloader, show_plot=False, get_logits=False):
    oo=[[0]*C for _ in range(C)]
    for x,y in dataloader:
        with torch.inference_mode():
            model.eval()
            try:
                model.remove_all_hooks()
                o=model(x)[:,-1,:] # model output
            except:
                o=model(x)[:, :]
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
        sns.heatmap(np.array(oo).T)
        plt.xlabel(f'(a+b) mod {C}')
        plt.ylabel(f'(a-b) mod {C}')
        plt.title('Correct Logits')
        plt.savefig(os.path.join(root_path, f"result/rep_logits.png"))
    if get_logits:
        return oo, dd
    else:
        return dd


if __name__ == '__main__':
    experiment_dir = os.path.join(root_path, f'code/save/{experiment_dir}')
    run_ids, run_types = read_from_json(experiment_dir, model_type)
    runid = run_ids[0]
    with open(os.path.join(experiment_dir, f'config_{runid}.json'),'r') as f:
        config=json.load(f)

    # load the dataset
    dataset = MyAddDataSet(
                        func=lambda x: (x[0]+x[1])%C,
                        C=C,
                        diff_vocab=False,
                        eqn_sign=False,
                        device=DEVICE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)

    linear_models={'alpha':MyModelA,'beta':MyModelB,'gama':MyModelC,'delta':MyModelD,'x':MyModelX}

    # pre-generated sequence for gradient symmetricity test
    xs = random.Random(42).sample(list(itertools.product(range(C), repeat=3)), 100)

    attn_coeffs = []
    gradient_symmetricities = []
    circularities = []
    distance_irrelevances = []
    d_models = []

    for id, type in zip(run_ids, run_types):

        if verbose:
            print(f"\nLoading model {id}, {type}")
        
        # initialize the model
        model_file = os.path.join(experiment_dir, f'model_{id}.pt')
        json_file = os.path.join(experiment_dir, f'config_{id}.json')
        with open(json_file, 'r') as f:
            config = json.load(f)
            attn_coeffs.append(config['attn_coeff'] if 'attn_coeff' in config else 0)
            d_models.append(config['d_model'] if 'd_model' in config else 0)
        
            if model_type not in ['A', 'B']:
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
            model.load_state_dict(torch.load(model_file, map_location=DEVICE))

            grad_sym = gradient_symmetricity(model, xs=xs)
            circ = circularity(model, first_k=4)
            dist_irr = distance_irrelevance(model, dataloader, show_plot=False)
            if verbose:
                print('Gradient Symmetricity', grad_sym)
                print('Circularity', circ)
                print('Distance Irrelevance', dist_irr)
            
            gradient_symmetricities.append(grad_sym)
            circularities.append(circ)
            distance_irrelevances.append(dist_irr)

    # Create a DataFrame
    df = pd.DataFrame({
        'Model Width': d_models,
        'Attn Coeff': attn_coeffs,
        'Gradient Symmetricity': gradient_symmetricities,
        'Circularity': circularities,
        'Distance Irrelevance': distance_irrelevances
    })

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    if not varying_width:
        # Plot the first scatter plot
        sc1 = axes[0].scatter(df['Attn Coeff'], df['Distance Irrelevance'], 
                            c=df['Gradient Symmetricity'], cmap='Oranges', edgecolor='k', s=40)
        axes[0].set_xlabel("Attention Rate")
        axes[0].set_ylabel("Distance Irrelevance")
        # axes[0].set_title("Gradient Symmetricity")
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(0, 1)
        cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
        cbar1.set_label("Gradient Symmetricity")

        # Plot the second scatter plot
        sc2 = axes[1].scatter(df['Attn Coeff'], df['Gradient Symmetricity'], 
                            c=df['Distance Irrelevance'], cmap='Oranges', edgecolor='k', s=40)
        axes[1].set_xlabel("Attention Rate")
        axes[1].set_ylabel("Gradient Symmetry")
        # axes[1].set_title("Distance Irrelevance")
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(0, 1)
        cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
        cbar2.set_label("Distance Irrelevance")

        # Show plot
        plt.savefig(os.path.join(root_path, f"result/rep_attn_coeff.png"))
    else:
        # Plot the first scatter plot
        sc1 = axes[0].scatter(df['Attn Coeff'], df['Model Width'], 
                            c=df['Gradient Symmetricity'], cmap='Oranges', edgecolor='k', s=40)
        axes[0].set_xlabel("Attention Rate")
        axes[0].set_ylabel("Model Width")
        # axes[0].set_title("Gradient Symmetricity")
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(24, 768)
        axes[0].set_yscale('log')
        axes[0].set_yticklabels(['32', '64', '128', '256', '512'])
        cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
        cbar1.set_label("Gradient Symmetricity")

        # Plot the second scatter plot
        sc2 = axes[1].scatter(df['Attn Coeff'], df['Model Width'], 
                            c=df['Distance Irrelevance'], cmap='Oranges', edgecolor='k', s=40)
        axes[1].set_xlabel("Attention Rate")
        axes[1].set_ylabel("Model Width")
        # axes[1].set_title("Distance Irrelevance")
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(24, 768)
        axes[1].set_yscale('log')
        axes[1].set_yticklabels(['32', '64', '128', '256', '512'])
        cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
        cbar2.set_label("Distance Irrelevance")

        # Show plot
        plt.savefig(os.path.join(root_path, f"result/rep_varying_width.png"))


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
from tqdm import tqdm
from copy import deepcopy

# ignore warning
import warnings
warnings.filterwarnings("ignore")

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.utils import extract_embeddings
from analysis.datasets import MyAddDataSet, MyDataset
from analysis.models import MyModelA, MyModelB, MyModelC, MyModelD, MyModelX, Transformer

# constants
C=59
DEVICE='cpu'
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

def embedding_gradient(model, model_type, xs=None):
    tt=0
    _model = deepcopy(model)
    device = next(_model.parameters()).device

    if xs is None:
        xs = random.sample(list(itertools.product(range(C), repeat=3)), 100)
    
    # now use scikit PCA to reduce the dimensionality of the embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    pts = []
    for ii in range(6):
        we=model.embed.W_E.T.detach().cpu().numpy()
        we2=pca.fit_transform(we)
        # set numbers in we2 with index is not equal to ii to 0

        we2[:, np.arange(we2.shape[1]) != ii] = 0

        we3=pca.inverse_transform(we2)
        model.embed.W_E.data=torch.tensor(we3.T).to(model.embed.W_E.device)

        
        for abc in xs:
            a,b,c=abc
            x=torch.tensor([[a,b]],device=device)
            t,o0=_model.forward_h(x)
            _model.zero_grad()
            #print(a,b,c)
            try: # for transformer
                _model.remove_all_hooks()
                o=o0[0,-1,:]
            except: # for linear models
                o=o0[0,:]
            
            t.retain_grad()
            o[c].backward(retain_graph=True)
            tg=t.grad[0].detach().cpu().numpy() # target gradient
            #tt+=tg[0][0]
            pts.append((tg[0].sum(), tg[1].sum()))
    pts = np.array(pts)
    # plot and save
    plt.figure(dpi=300)
    plt.scatter(pts[:,0], pts[:,1], c='r' if model_type == 'A' else 'b')
    plt.xlabel(r"Gradients on $E_a$")
    plt.ylabel(r"Gradients on $E_b$")
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.title('Embedding Gradient')
    plt.savefig(os.path.join(root_path, f"result/rep_model_{model_type}_embedding_gradient.png"))
    return pts

def gradient_symmetricity(model, xs=None):
    tt=0
    _model = deepcopy(model)
    device = next(_model.parameters()).device

    if xs is None:
        xs = random.sample(list(itertools.product(range(C), repeat=3)), 100)
    for abc in xs:
        a,b,c=abc
        x=torch.tensor([[a,b]],device=device)
        t,o0=_model.forward_h(x)
        _model.zero_grad()
        #print(a,b,c)
        try: # for transformer
            _model.remove_all_hooks()
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
    
def distance_irrelevance(model, dataloader, show_plot=False, get_logits=False, top_k_logits=False):
    oo=[[0]*C for _ in range(C)]
    oc=[[0]*C for _ in range(C)]
    for x,y in dataloader:
        with torch.inference_mode():
            model.eval()
            try:
                model.remove_all_hooks()
                o=model(x)[:,-1,:] # model output
            except:
                o=model(x)[:, :]
            
            if not top_k_logits:
                o0=o[list(range(len(x))),y] # extracts the logit corresponding to the true label y for each sample?
                o0=o0.cpu()
                x=x.cpu()
                for p,q in zip(o0,x):
                    A,B=int(q[0].item()),int(q[1].item())
                    oo[(A+B)%C][(A-B)%C]=p.item()
            else:
                o[list(range(len(x))),y]=float("-inf")
                o1=o.topk(dim=-1,k=2).values.cpu()
                print(o1.numpy())
                for p,q in zip(o1,x):
                    A,B=int(q[0].item()),int(q[1].item())
                    oc[(A+B)%C][(A-B)%C]=p[0].item()
    
    oo=np.array(oo) if not top_k_logits else np.array(oc)
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

def run_model_analysis(experiment_name, model_type, save_dataframe=True, save_config=False, check_embed_gradient=False, verbose=False):
    
    print("Running model analysis for", experiment_name, "and model type", model_type, "with: \n save_dataframe:", save_dataframe, "\n save_config:", save_config, "\n check_embed_gradient:", check_embed_gradient, "\n verbose:", verbose)

    assert model_type in ['A', 'B', 'alpha', 'beta', 'gama', 'delta', 'x'], "Invalid model type!"
    experiment_dir = os.path.join(root_path, f'code/save/{experiment_name}')
    run_ids, run_types = read_from_json(experiment_dir, model_type)
    runid = run_ids[0]
    with open(os.path.join(experiment_dir, f'config_{runid}.json'),'r') as f:
        config=json.load(f)

    # load the dataset
    if model_type in ['A', 'B']:
        dataset = MyAddDataSet(C=C,
                            func=lambda x: (x[0]+x[1])%C,
                            diff_vocab=False,
                            eqn_sign=False,
                            device=DEVICE)
    else:
        dataset = MyDataset(n_vocab=C, device=DEVICE)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)

    linear_models={'alpha':MyModelA,'beta':MyModelB,'gama':MyModelC,'delta':MyModelD,'x':MyModelX}

    # pre-generated sequence for gradient symmetricity test
    xs = random.Random(42).sample(list(itertools.product(range(C), repeat=3)), 100)

    configs = []

    for run_id, run_type in tqdm(zip(run_ids, run_types), total=len(run_ids)):

        if verbose:
            print(f"\nLoading model {run_id}, {run_type}")
        
        # initialize the model
        model_file = os.path.join(experiment_dir, f'model_{run_id}.pt')
        json_file = os.path.join(experiment_dir, f'config_{run_id}.json')
        with open(json_file, 'r') as f:
            config = json.load(f)
            if model_type in ['A', 'B']:
                model=Transformer(num_layers=config.get('n_layers',1),
                        num_heads=config['n_heads'],
                        d_model=config['d_model'],
                        d_head=config.get('d_head',config['d_model']//config['n_heads']),
                        attn_coeff=config['attn_coeff'],
                        d_vocab=dataset.vocab,
                        act_type=config.get('act_fn','relu'),
                        n_ctx=dataset.dim,)
            else:
                model=linear_models[model_type]()
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_file, map_location=DEVICE))

            grad_sym = gradient_symmetricity(model, xs=xs)
            circ = circularity(model, first_k=4)
            dist_irr = distance_irrelevance(model, dataloader, show_plot=False)
            # only do this with the first model and check_embed_gradient=True
            if check_embed_gradient and model_type in ['A', 'B'] and run_id == run_ids[0]:
                pts = embedding_gradient(model, model_type, xs=xs)
            if verbose:
                print('Gradient Symmetricity', grad_sym)
                print('Circularity', circ)
                print('Distance Irrelevance', dist_irr)
            config['dist_irr'] = dist_irr
            config['grad_sym'] = grad_sym
            config['circ'] = circ
            configs.append(config)

    # Create a DataFrame
    df = pd.DataFrame(configs)
    try:
        columns_to_remove = ['C', 'func', 'epoch']
        df.drop(columns=columns_to_remove, inplace=True)
    except:
        pass

    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if save_dataframe:
        df.to_csv(os.path.join(result_dir, f"results_{experiment_name}.csv"))

    if save_config:
        for run_id, config in zip(run_ids, configs):
            with open(os.path.join(experiment_dir, f'config_{run_id}.json'),'w') as f:
                json.dump(config, f, separators=(',\n', ': '))


if __name__ == '__main__':
    model_type='B'
    experiment_name='model_B_embeddings'
    run_model_analysis(experiment_name, model_type, 
                       save_dataframe=False, 
                       save_config=True, 
                       check_embed_gradient=True, 
                       verbose=False)
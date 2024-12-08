#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import os, sys
import json
import pandas as pd
import seaborn as sns

torch.set_default_tensor_type(torch.DoubleTensor)

# ignore warning
import warnings
warnings.filterwarnings("ignore")

C=59
DEVICE='cpu'

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.models import MyModelA, MyModelB, MyModelC, MyModelD, MyModelX, Transformer
from analysis.utils import extract_embeddings, format_subplot, get_final_circle_freqs

type_mapping = {'alpha': 'A', 'beta': 'B', 'gama': 'C', 'delta': 'D', 'x': 'X'}
linear_models={'alpha':MyModelA,'beta':MyModelB,'gama':MyModelC,'delta':MyModelD,'x':MyModelX}
# %%

model_type = 'beta'
embedding_types = ['unembed', 'embed']
experiment_name = f'model_{model_type}_embeddings'
data_path = "./result"

def read_from_model(directory, filter=''):
    infos=[]
    for filename in os.listdir(directory):
        if filename.startswith('model_'+filter) and filename.endswith('.pt'):
            runid = filename[len('model_'):-len('.pt')]
            config_path = os.path.join(directory, f'config_{runid}.json')
            model_path = os.path.join(directory, f'model_{runid}.pt')
            embedding_path = os.path.join(directory, f'embeddings_{runid}.npz')
            info = {'run_id': runid, 'model_path': model_path}
            if os.path.exists(config_path):
               info['config_path'] = config_path
            if os.path.exists(embedding_path):
               info['embedding_path'] = embedding_path
            infos.append(info)
    return infos

def plot_evolution(model_type, embedding_type, info):

    assert model_type in ['A', 'B', 'alpha', 'beta', 'gama', 'delta', 'x']
    assert embedding_type in ['embed', 'embed1', 'embed2', 'pos_embed', 'unembed']

    result_dir = os.path.join(root_path, f"result/{experiment_name}", embedding_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    embeddings = np.load(info['embedding_path'], allow_pickle=True)['arr_0'] # list of model embedding for each step
    if not embedding_type in embeddings[0].keys():
        print(f"Embedding type {embedding_type} not found in embeddings")
        return False
    
    # concat all the arrays from embedding_type in embeddings
    embeddings = np.array([embedding[embedding_type] for embedding in embeddings])
    if embeddings.shape[1] != C:
        print('Taking transpose of embeddings with shape: ', embeddings.shape)
        embeddings = embeddings.transpose(0, 2, 1)
    
    # %%
    spectrums = np.fft.fft(embeddings, axis=1)
    signals = np.linalg.norm(spectrums, axis=-1)
    # print(signals.shape)
    # %%
    circle_freqs = get_final_circle_freqs(embeddings)
    num_circles = len(circle_freqs)
    assert num_circles > 0, "No circle frequencies found"
    circle_freqs_i = list(zip(*circle_freqs))[0]

    # %%
    signals_df = pd.DataFrame(signals[:, 1:30], columns=[f'k={i}' for i in range(1, 30)])
    circle_df = signals_df[[f'k={i}' for i in circle_freqs_i]]
    noncircle_df = signals_df[[f'k={i}' for i in range(1, 30) if not i in circle_freqs_i]]

    circle_df.head()
    # %%
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(dpi=300)

    sns.lineplot(data=noncircle_df, legend=None, dashes=False, palette=['grey'], alpha=0.2)
    sns.lineplot(data=circle_df, legend=True, dashes=False)

    grid_x=True
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax.set_xscale('log')
    plt.xlabel('step')
    plt.ylabel('signal')
    plt.title("Evolution of Frequency Signal With Time", fontsize=14)
    plt.savefig(os.path.join(result_dir, f"signal_evolution_{info['run_id']}.png"))
    plt.show()

    # %%
    steps = np.linspace(0, embeddings.shape[0]-1, num=5)
    freqs = circle_freqs_i
    multiplier = 20000 / embeddings.shape[0]

    fig, axes = plt.subplots(len(freqs), len(steps), figsize=(3*len(steps), 3*len(freqs)), dpi=300)

    for i, freq in enumerate(freqs):
        for j, step in enumerate(steps):
            real = spectrums[-1, freq].real
            imag = spectrums[-1, freq].imag
            real /= np.linalg.norm(real)
            imag /= np.linalg.norm(imag)
            embed = np.stack([embeddings[step] @ real, embeddings[step] @ imag, np.arange(59)],axis=0)
            
            embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
            sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i, j], palette="viridis", legend=False)

            axes[i, j].set(xlabel=None)
            axes[i, j].set(ylabel=None)
            axes[i, j].set_xlim(-1.5, 1.5)
            axes[i, j].set_ylim(-1.5, 1.5)
            axes[i, j].text(1, -2, "k = {}".format(freq))

            if i == 0:
                axes[i, j].set_title(f"{step * multiplier} steps", fontsize=12)
            format_subplot(axes[i, j])

    fig.suptitle("Evolution of Embedding on FFT Plane", fontsize=17)
    plt.savefig(os.path.join(result_dir, f"embedding_evolution_{info['run_id']}.png"))
    plt.show()

    #%%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    final_embedding = embeddings[-1]

    # do PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2*num_circles)
    pca.fit(final_embedding)
    components = pca.components_

    # print(pca.singular_values_)

    for i in range(num_circles):
        x = components[i * 2]
        y = components[i * 2 + 1]
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        embed = np.stack([final_embedding @ x, final_embedding @ y, np.arange(59)],axis=0)

        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(circle_freqs[i][0], circle_freqs[i][1]), fontsize=14)
        format_subplot(axes[i])

    fig.suptitle("Embeddings of Frequencies on PCA Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, f"pca_embedding_{info['run_id']}.png"))
    plt.show()

    # %%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    final_embedding = embeddings[-1]

    for i, (freq, sig) in enumerate(circle_freqs):
        real = spectrums[-1, freq].real
        imag = spectrums[-1, freq].imag
        real /= np.linalg.norm(real)
        imag /= np.linalg.norm(imag)
        embed = np.stack([final_embedding @ real, final_embedding @ imag, np.arange(59)],axis=0)
        
        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(freq, sig), fontsize=14)
        format_subplot(axes[i])

    fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, f"fft_embedding_{info['run_id']}.png"))
    plt.show()

    return True

def plot_final_embedding(model_type, embedding_type, info):
    result_dir = os.path.join(root_path, f"result/{experiment_name}", embedding_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if model_type in ['A', 'B']:
        model=Transformer(num_layers=config.get('n_layers',1),
                num_heads=config['n_heads'],
                d_model=config['d_model'],
                d_head=config.get('d_head',config['d_model']//config['n_heads']),
                attn_coeff=config['attn_coeff'],
                d_vocab=C,
                act_type=config.get('act_fn','relu'),
                n_ctx=2)
    else:
        model=linear_models[model_type]()
    model.to(DEVICE)
    model.load_state_dict(torch.load(info['model_path'], map_location=DEVICE))
    embedding = extract_embeddings(model)[embedding_type]

    spectrums = np.fft.fft(embedding, axis=0)
    signals = np.linalg.norm(spectrums, axis=1)
    sorted_freq = np.argsort(signals)[::-1]
    threshold = np.mean(signals) * 2 
    num_circles = (signals > threshold).sum() // 2
    cur_freqs = [min(sorted_freq[i * 2], sorted_freq[i * 2 + 1]) for i in range(num_circles)]

    circle_freqs = list(zip(cur_freqs, signals[cur_freqs]))
    num_circles = len(circle_freqs)
    assert num_circles > 0, "No circle frequencies found"
    circle_freqs_i = list(zip(*circle_freqs))[0]

    # %%
    signals_df = pd.DataFrame(signals[:, 1:30], columns=[f'k={i}' for i in range(1, 30)])
    circle_df = signals_df[[f'k={i}' for i in circle_freqs_i]]
    noncircle_df = signals_df[[f'k={i}' for i in range(1, 30) if not i in circle_freqs_i]]

    #%%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    # do PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2*num_circles)
    pca.fit(embedding)
    components = pca.components_

    for i in range(num_circles):
        x = components[i * 2]
        y = components[i * 2 + 1]
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        embed = np.stack([embedding @ x, embedding @ y, np.arange(59)],axis=0)

        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(circle_freqs[i][0], circle_freqs[i][1]), fontsize=14)
        format_subplot(axes[i])

    fig.suptitle("Embeddings of Frequencies on PCA Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, f"pca_embedding_{info['run_id']}.png"))
    plt.show()

    # %%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    for i, (freq, sig) in enumerate(circle_freqs):
        real = spectrums[-1, freq].real
        imag = spectrums[-1, freq].imag
        real /= np.linalg.norm(real)
        imag /= np.linalg.norm(imag)
        embed = np.stack([embedding @ real, embedding @ imag, np.arange(59)],axis=0)
        
        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(freq, sig), fontsize=14)
        format_subplot(axes[i])

        
    fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, f"fft_embedding_{info['run_id']}.png"))
    plt.show()

    return True

for embedding_type in embedding_types:
    experiment_dir = os.path.join(root_path, f"code/save/{experiment_name}")
    infos = read_from_model(experiment_dir, model_type)
    for info in infos:
        print(f"\nLoading {embedding_type} of model {info['run_id']}")
        # if info contains config_path
        if 'embedding_path' and 'config_path' in info:
            plot_evolution(model_type, embedding_type, info)
        else:
            plot_final_embedding(model_type, embedding_type, info)

        if 'config_path' in info:
            with open(info['config_path'], 'r') as f:
                config = json.load(f)
                try:
                    print('Gradient Symmetricity', config['grad_sym'])
                    print('Circularity', config['circ'])
                    print('Distance Irrelevance', config['dist_irr'])
                except:
                    pass

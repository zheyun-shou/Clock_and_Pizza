#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import os
import sys
from tqdm import tqdm
import glob
import os
import pandas as pd
import seaborn as sns
from utils import format_subplot, get_final_circle_freqs
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
trial_embeddings = [] 
data_path = "./"
save_path = "./spectrum_figs"

if not os.path.exists(save_path):
    os.makedirs(save_path)

embeddings = np.load(os.path.join(data_path, "pos_embeddings.npy")) # list of model embedding for each step
embeddings = embeddings.transpose(0, 2, 1)
embeddings.shape
# %%
spectrums = np.fft.fft(embeddings, axis=1)
signals = np.linalg.norm(spectrums, axis=-1)
signals.shape
# %%
circle_freqs = get_final_circle_freqs(embeddings)
circle_freqs_i = list(zip(*circle_freqs))[0]
num_circles = len(circle_freqs)
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
plt.savefig(os.path.join(save_path, "signal_evolution.png"))
plt.show()
#%%
fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

final_embedding = embeddings[-1]

# do PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2*num_circles)
pca.fit(final_embedding)
components = pca.components_

print(pca.singular_values_)

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
    axes[i].text(-0.5, 0, r"k = {}, signal norm $\Delta$ = {:03}".format(circle_freqs[i][0], circle_freqs[i][1]), fontsize=14)
    format_subplot(axes[i])

fig.suptitle("Embeddings of Frequencies on PCA Planes", fontsize=20)
plt.savefig(os.path.join(save_path, "pca_embedding.png"))
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
    axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {}".format(freq, sig), fontsize=14)
    format_subplot(axes[i])

    
fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
plt.savefig(os.path.join(save_path, "fft_embedding.png"))
plt.show()
# %%
steps = range(0, 19999, 10000)
freqs = range(1, 31)

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
            axes[i, j].set_title(f"{step} steps", fontsize=12)
        format_subplot(axes[i, j])

fig.suptitle("Evolution of Embedding on FFT Plane", fontsize=17)
plt.savefig(os.path.join(save_path, "embedding_evolution.png"))
plt.show()
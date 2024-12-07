import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.model_analysis import run_model_analysis

overwrite = False # set to True if you want to overwrite the existing results

experiment_name = 'attn_fixedwidth'
model_type = 'B' # A, B, alpha, beta, gamma, delta, x
circ_threshold = 0.85 # ignore the data with circularity less than threshold

# check if the file already exists
result_dir = os.path.join(root_path, f'result/{experiment_name}')
if not os.path.exists(os.path.join(result_dir, f'results_{experiment_name}.csv')) or overwrite:
    print(f"Running model analysis for {experiment_name} and model type {model_type}")
    run_model_analysis(experiment_name, model_type, verbose=False)

df = pd.read_csv(os.path.join(result_dir, f'results_{experiment_name}.csv'))
# print(df.columns)
df = df[df['circ'] >= circ_threshold]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

if experiment_name == 'attn_fixedwidth':
    # Plot the first scatter plot
    sc1 = axes[0].scatter(df['attn_coeff'], df['dist_irr'], 
                        c=df['grad_sym'], cmap='Oranges', edgecolor='k', s=40)
    axes[0].set_xlabel("Attention Rate")
    axes[0].set_ylabel("Distance Irrelevance")
    # axes[0].set_title("Gradient Symmetricity")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(0, 1)
    cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
    cbar1.set_label("Gradient Symmetricity")

    # Plot the second scatter plot
    sc2 = axes[1].scatter(df['attn_coeff'], df['grad_sym'], 
                        c=df['dist_irr'], cmap='Oranges', edgecolor='k', s=40)
    axes[1].set_xlabel("Attention Rate")
    axes[1].set_ylabel("Gradient Symmetry")
    # axes[1].set_title("Distance Irrelevance")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(0, 1)
    cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
    cbar2.set_label("Distance Irrelevance")

    # Show plot
    plt.savefig(os.path.join(result_dir, f"rep_attn_coeff.png"))
elif experiment_name == 'attn_varywidth':
    # Plot the first scatter plot
    sc1 = axes[0].scatter(df['attn_coeff'], df['d_model'], 
                        c=df['grad_sym'], cmap='Oranges', edgecolor='k', s=40)
    axes[0].set_xlabel("Attention Rate")
    axes[0].set_ylabel("Model Width")
    # axes[0].set_title("Gradient Symmetricity")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(32, 512)
    axes[0].set_yscale('log')
    axes[0].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[0].set_yticklabels(['32', '64', '128', '256', '512'])
    axes[0].set_yticks([32, 64, 128, 256, 512])
    cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
    cbar1.set_label("Gradient Symmetricity")

    # Plot the second scatter plot
    sc2 = axes[1].scatter(df['attn_coeff'], df['d_model'], 
                        c=df['dist_irr'], cmap='Oranges', edgecolor='k', s=40)
    axes[1].set_xlabel("Attention Rate")
    axes[1].set_ylabel("Model Width")
    # axes[1].set_title("Distance Irrelevance")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(32, 512)
    axes[1].set_yscale('log')
    axes[1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axes[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[1].set_yticklabels(['32', '64', '128', '256', '512'])
    axes[1].set_yticks([32, 64, 128, 256, 512])
    cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
    cbar2.set_label("Distance Irrelevance")

    # Show plot
    plt.savefig(os.path.join(result_dir, f"rep_varying_width.png"))
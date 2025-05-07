from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import euclidean, cosine
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import os

def stable_cosine_similarity(a, b):
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)

def compute_2d_projection(embeddings, method='tsne', perplexity=30, n_iter=1000, n_neighbors=15, min_dist=0.1):
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'tsne', 'umap', or 'pca'")
    return embeddings_2d

def plot_embeddings(embeddings_2d, model_name, tokens=None, method='pca', use_altair=False):
    if use_altair:
        if tokens is None:
            tokens = [f"tok_{i}" for i in range(len(embeddings_2d))]
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'token': tokens
        })
        
        # Configure Altair to use more space
        alt.data_transformers.disable_max_rows()
        
        # Create base chart
        base = alt.Chart(df).encode(
            x=alt.X('x:Q', title='Dimension 1'),
            y=alt.Y('y:Q', title='Dimension 2'),
            tooltip=['token', 'x', 'y']
        )
        
        # Create points
        points = base.mark_circle(size=60).properties(
            width=800,
            height=600,
            title=f'{method.upper()} of {model_name} Embeddings'
        )
        
        # Save as HTML for viewing outside Jupyter
        filename = f'{method.lower()}_{model_name.replace("/", "_")}.html'
        points.save(filename)
        print(f"Saved Altair chart to {filename}")
        return points
    else:
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=1)
        plt.title(f'{method.upper()} Visualization of {model_name} Vocabulary Embeddings')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        filename = f'{method.lower()}_{model_name.replace("/", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {method.upper()} plot to {filename}")

def compute_distances(embeddings, num_pairs=1000, use_mean_embeddings=False, tokens_per_mean=1):
    cosine_sims, l2_dists = [], []
    for _ in range(num_pairs):
        if use_mean_embeddings:
            tokens_a = random.sample(range(len(embeddings)), tokens_per_mean)
            tokens_b = random.sample(range(len(embeddings)), tokens_per_mean)
            a = np.mean(embeddings[tokens_a], axis=0)
            b = np.mean(embeddings[tokens_b], axis=0)
        else:
            i, j = random.sample(range(len(embeddings)), 2)
            a, b = embeddings[i], embeddings[j]
        cosine_sims.append(1 - cosine(a, b))
        l2_dists.append(euclidean(a, b))
    return np.array(cosine_sims), np.array(l2_dists)

def plot_cosine_histogram(cosine_sims, model_name, use_mean_embeddings=False):
    plt.figure(figsize=(10, 6))
    hist, bins = np.histogram(cosine_sims, bins='auto')
    plt.hist(cosine_sims, bins=bins, alpha=0.7, edgecolor='black')
    title = f'Cosine Similarity Distribution for {model_name}'
    if use_mean_embeddings:
        title += ' (Mean Embeddings)'
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    filename = f'cosine_hist_{model_name.replace("/", "_")}'
    if use_mean_embeddings:
        filename += '_mean'
    plt.savefig(f'{filename}.png')
    plt.close()

def get_token_embeddings(model_name, max_vocab=50000):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    norms = np.linalg.norm(emb, axis=1)
    print(f"mean norm: {norms.mean():.4f}, std: {norms.std():.4f}, min: {norms.min():.4f}, max: {norms.max():.4f}")
    
    # Get token strings
    token_strings = []
    for i in range(min(max_vocab, len(tokenizer))):
        token = tokenizer.decode([i])
        token_strings.append(token)
    
    return emb[:max_vocab], token_strings

model_name = "lmsys/vicuna-7b-v1.5"
USE_MEAN_EMBEDDINGS = False
TOKENS_PER_MEAN = 50
VISUALIZATION_METHOD = 'umap'  # 'tsne', 'pca', or 'umap'
USE_ALTAIR = True

emb, token_strings = get_token_embeddings(model_name)
embeddings_2d = compute_2d_projection(emb, method=VISUALIZATION_METHOD)
plot = plot_embeddings(embeddings_2d, model_name, tokens=token_strings, method=VISUALIZATION_METHOD, use_altair=USE_ALTAIR)
# Altair plot already saved as HTML above.

cos_sims, _ = compute_distances(emb, num_pairs=1000, use_mean_embeddings=USE_MEAN_EMBEDDINGS, tokens_per_mean=TOKENS_PER_MEAN)
print(f"mean: {cos_sims.mean():.4f}, std: {cos_sims.std():.4f}, min: {cos_sims.min():.4f}, max: {cos_sims.max():.4f}")
plot_cosine_histogram(cos_sims, model_name, use_mean_embeddings=USE_MEAN_EMBEDDINGS)

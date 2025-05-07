from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import euclidean, cosine
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

def stable_cosine_similarity(a, b):
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)

def plot_embedding_space(embeddings, model_name, method='tsne', perplexity=30, n_iter=1000, n_neighbors=15, min_dist=0.1):
    print(f"Computing {method.upper()} for {model_name}...")
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        # Print explained variance ratio
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'tsne', 'umap', or 'pca'")
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=1)
    plt.title(f'{method.upper()} Visualization of {model_name} Vocabulary Embeddings')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    
    # Save the plot
    filename = f'{method.lower()}_{model_name.replace("/", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {method.upper()} plot to {filename}")

def compute_distances(embeddings, num_pairs=1000, use_mean_embeddings=False, tokens_per_mean=1):
    cosine_sims, l2_dists = [], []
    
    for _ in range(num_pairs):
        if use_mean_embeddings:
            # Sample multiple tokens and average them
            tokens_a = random.sample(range(len(embeddings)), tokens_per_mean)
            tokens_b = random.sample(range(len(embeddings)), tokens_per_mean)
            a = np.mean(embeddings[tokens_a], axis=0)
            b = np.mean(embeddings[tokens_b], axis=0)
        else:
            # Sample single tokens as before
            i, j = random.sample(range(len(embeddings)), 2)
            a, b = embeddings[i], embeddings[j]
            
        cosine_sims.append(1-cosine(a, b))
        l2_dists.append(euclidean(a, b))
    return np.array(cosine_sims), np.array(l2_dists)

def plot_cosine_histogram(cosine_sims, model_name, use_mean_embeddings=False):
    plt.figure(figsize=(10, 6))
    
    # Use numpy's histogram with automatic bin selection
    hist, bins = np.histogram(cosine_sims, bins='auto')
    
    # Plot the histogram
    plt.hist(cosine_sims, bins=bins, alpha=0.7, edgecolor='black')
    
    title = f'Cosine Similarity Distribution for {model_name}'
    if use_mean_embeddings:
        title += ' (Mean Embeddings)'
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    filename = f'cosine_hist_{model_name.replace("/", "_")}'
    if use_mean_embeddings:
        filename += '_mean'
    plt.savefig(f'{filename}.png')
    plt.close()

def get_token_embeddings(model_name, max_vocab=50000):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    emb_layer = model.get_input_embeddings()
    
    # Handle different embedding layer structures
    if hasattr(emb_layer, 'weight'):
        emb = emb_layer.weight.detach().cpu().numpy()
    elif hasattr(emb_layer, 'embedding'):
        emb = emb_layer.embedding.detach().cpu().numpy()
    else:
        # Try to get embeddings directly from the layer
        emb = emb_layer.detach().cpu().numpy()
    
    norms = np.linalg.norm(emb, axis=1)
    print(f"mean norm: {norms.mean():.4f}, std: {norms.std():.4f}, min: {norms.min():.4f}, max: {norms.max():.4f}")
    return emb[:max_vocab]  # clip to 50k if huge

# Example usage
model_names = [
    # "EleutherAI/gpt-j-6B",
    # "Qwen/Qwen2-7B",
    # "tiiuae/falcon-7b",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "lmsys/vicuna-7b-v1.5"
    # "google/gemma-2-9b-it",
    # "allenai/Molmo-7B-D-0924",
    "allenai/OLMo-1B"
]

USE_MEAN_EMBEDDINGS = False  # Set to True to use mean embeddings, False for single token comparisons
TOKENS_PER_MEAN = 50  # Number of tokens to average together
VISUALIZATION_METHOD = 'pca'  # Choose from 'tsne', 'umap', or 'pca'

for name in model_names:
    print(f"\n{name}")
    emb = get_token_embeddings(name)
    
    # Plot embedding space visualization
    plot_embedding_space(emb, name, method=VISUALIZATION_METHOD)
    
    # Compute and plot cosine similarities
    cos_sims, _ = compute_distances(emb, num_pairs=1000, use_mean_embeddings=USE_MEAN_EMBEDDINGS, tokens_per_mean=TOKENS_PER_MEAN)
    print(f"mean: {cos_sims.mean():.4f}, std: {cos_sims.std():.4f}, min: {cos_sims.min():.4f}, max: {cos_sims.max():.4f}")
    plot_cosine_histogram(cos_sims, name, use_mean_embeddings=USE_MEAN_EMBEDDINGS)

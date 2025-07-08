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
from datasets import load_dataset
import random

# Constants
NUM_SENTENCES = 100
USE_MEAN_EMBEDDINGS = False
VISUALIZATION_METHOD = 'pca'
BATCH_SIZE = 32  # Default batch size for models with padding token

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

def load_diverse_sentences(num_sentences=NUM_SENTENCES):
    """Load a diverse set of sentences from DBPedia-14 dataset."""
    # Load DBPedia dataset
    dataset = load_dataset('dbpedia_14', split='train')
    
    # Get the content and shuffle
    sentences = dataset['content']
    sentences = list(sentences)
    random.shuffle(sentences)
    
    # Select the desired number of sentences
    selected_sentences = sentences[:num_sentences]
    
    print(f"Loaded {len(selected_sentences)} diverse sentences from DBPedia-14")
    return selected_sentences

def get_layer_embeddings(model, tokenizer, sentences, num_sentences=NUM_SENTENCES, batch_size=8):
    """Get embeddings for one random token from each sentence at each layer."""
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Ensure we don't process more sentences than available
    sentences = sentences[:num_sentences]
    total_batches = (len(sentences) + batch_size - 1) // batch_size
    print(f"Processing {len(sentences)} sentences in {total_batches} batches of size {batch_size}")
    
    layer_embeddings = []  # Will store [num_layers, num_sentences, hidden_size]
    num_layers = None
    
    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        current_batch = i // batch_size + 1
        
        # Tokenize batch - only use padding if batch_size > 1
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "truncation": True
        }
        if batch_size > 1:
            tokenizer_kwargs["padding"] = True
            
        encodings = tokenizer(batch_sentences, **tokenizer_kwargs)
        # Move encodings to GPU
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Get model outputs with all hidden states
        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)
        
        # Get all hidden states (tuple of tensors, one per layer)
        all_hidden_states = outputs.hidden_states  # [num_layers, batch_size, seq_len, hidden_size]
        
        if num_layers is None:
            num_layers = len(all_hidden_states)
            # Initialize layer_embeddings with the correct number of layers
            layer_embeddings = [[] for _ in range(num_layers)]
        
        # For each sentence in the batch, randomly select one token position
        for layer_idx in range(num_layers):
            layer_hidden = all_hidden_states[layer_idx]  # [batch_size, seq_len, hidden_size]
            
            for sent_idx in range(len(batch_sentences)):
                # Get non-padding token positions
                non_pad_mask = encodings['attention_mask'][sent_idx].bool()
                valid_positions = torch.where(non_pad_mask)[0]
                
                # Randomly select one position
                if len(valid_positions) > 0:
                    random_pos = random.choice(valid_positions)
                    layer_embeddings[layer_idx].append(layer_hidden[sent_idx, random_pos].cpu().numpy())
        
        print(f"Processed batch {current_batch}/{total_batches}")
    
    # Convert lists to numpy arrays
    return np.array([np.array(embeddings) for embeddings in layer_embeddings])

def analyze_layer_similarities(layer_embeddings, model_name, static_embeddings):
    """Analyze cosine similarities between tokens at each layer."""
    num_layers = len(layer_embeddings)
    layer_stats = []
    all_cos_sims = []  # Store all cosine similarities for plotting
    
    # Save layer-wise embeddings to disk for cross-modal analysis
    embeddings_dir = Path("analysis_results/cached_text_embeddings") / model_name.replace("/", "_")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving layer-wise text embeddings to {embeddings_dir}")
    
    # Save static vocabulary embeddings as layer 0
    np.save(embeddings_dir / "layer_0_static_vocab.npy", static_embeddings)
    print("Saved static vocabulary embeddings")
    
    # Save each transformer layer's embeddings
    for layer_idx in range(num_layers):
        layer_file = embeddings_dir / f"layer_{layer_idx + 1}_text_embeddings.npy"
        np.save(layer_file, layer_embeddings[layer_idx])
        print(f"Saved layer {layer_idx + 1} text embeddings: {layer_embeddings[layer_idx].shape}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "num_layers": num_layers,
        "num_sentences": len(layer_embeddings[0]),
        "embedding_dim": layer_embeddings[0].shape[1],
        "static_vocab_size": len(static_embeddings),
        "static_vocab_dim": static_embeddings.shape[1]
    }
    
    import json
    with open(embeddings_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata}")
    
    # First analyze static vocabulary matrix (Layer 0)
    print("\nComputing static vocabulary matrix similarities...")
    # Use compute_distances to get 5000 random pairs from the vocabulary
    static_cos_sims, _ = compute_distances(static_embeddings, num_pairs=5000)
    print("Finished static vocabulary matrix similarities")
    
    static_stats = {
        'mean': static_cos_sims.mean(),
        'std': static_cos_sims.std(),
        'min': static_cos_sims.min(),
        'max': static_cos_sims.max()
    }
    layer_stats.append(static_stats)
    all_cos_sims.append(static_cos_sims)
    
    # Then analyze each transformer layer
    print("\nComputing transformer layer similarities...")
    for layer_idx in range(num_layers):
        print(f"\nProcessing layer {layer_idx + 1}/{num_layers}")
        embeddings = layer_embeddings[layer_idx]
        # Compute pairwise cosine similarities
        cos_sims = []
        total_pairs = len(embeddings) * (len(embeddings) - 1) // 2
        pair_count = 0
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = stable_cosine_similarity(embeddings[i], embeddings[j])
                cos_sims.append(sim)
                pair_count += 1
                if pair_count % 100 == 0:  # Print progress every 100 pairs
                    print(f"Processed {pair_count}/{total_pairs} pairs ({(pair_count/total_pairs)*100:.1f}%)")
        
        print(f"Finished layer {layer_idx + 1}")
        cos_sims = np.array(cos_sims)
        stats = {
            'mean': cos_sims.mean(),
            'std': cos_sims.std(),
            'min': cos_sims.min(),
            'max': cos_sims.max()
        }
        layer_stats.append(stats)
        all_cos_sims.append(cos_sims)
    
    print("\nCreating visualization...")
    # Create a single plot showing all distributions
    plt.figure(figsize=(15, 8))
    
    # Plot each layer's distribution
    layer_numbers = [0, 0.5] + list(range(1, num_layers + 1))  # 0, 0.5, 1, 2, 3, ...
    for layer_idx, cos_sims in enumerate(all_cos_sims):
        # Create a violin plot for each layer
        plt.violinplot(cos_sims, positions=[layer_numbers[layer_idx]], 
                      showmeans=True, showextrema=True)
    
    plt.title(f'Cosine Similarity Distribution Across Layers for {model_name}')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.xticks(layer_numbers)
    
    # Add mean values as text above each violin
    for layer_idx, stats in enumerate(layer_stats):
        plt.text(layer_numbers[layer_idx], stats['mean'], 
                f'{stats["mean"]:.3f}', 
                ha='center', va='bottom')
    
    plt.savefig(f'layer_similarities_combined_{model_name.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization saved")
    
    return layer_stats


model_names = [
    # "EleutherAI/gpt-j-6B",
    "Qwen/Qwen2-7B",
    # "tiiuae/falcon-7b",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "lmsys/vicuna-7b-v1.5"
    # "google/gemma-2-9b-it",
    # "allenai/Molmo-7B-D-0924",
    # "allenai/OLMo-1B"
]

for name in model_names:
    print(f"\n{name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    
    # Adjust batch size if no padding token
    current_batch_size = 1 if tokenizer.pad_token is None else BATCH_SIZE
    print(f"Using batch size: {current_batch_size} (padding token {'not' if tokenizer.pad_token is None else ''} available)")
    
    # Get initial token embeddings
    emb = get_token_embeddings(name)
    
    # Plot embedding space visualization
    plot_embedding_space(emb, name, method=VISUALIZATION_METHOD)
    
    # Load sentences and get layer-wise embeddings
    sentences = load_diverse_sentences(num_sentences=NUM_SENTENCES)
    layer_embeddings = get_layer_embeddings(model, tokenizer, sentences, batch_size=current_batch_size)
    
    # Analyze layer-wise similarities including static vocabulary matrix
    layer_stats = analyze_layer_similarities(layer_embeddings, name, emb)
    
    # Print layer-wise statistics
    print("\nLayer-wise statistics:")
    print(f"Static Vocabulary Matrix (Layer 0): mean={layer_stats[0]['mean']:.4f}, std={layer_stats[0]['std']:.4f}, min={layer_stats[0]['min']:.4f}, max={layer_stats[0]['max']:.4f}")
    print(f"First Hidden State (Layer 0.5): mean={layer_stats[1]['mean']:.4f}, std={layer_stats[1]['std']:.4f}, min={layer_stats[1]['min']:.4f}, max={layer_stats[1]['max']:.4f}")
    for layer_idx, stats in enumerate(layer_stats[2:], start=1):
        print(f"Layer {layer_idx}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

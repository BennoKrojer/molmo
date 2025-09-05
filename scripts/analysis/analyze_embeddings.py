import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import euclidean
import random

def stable_cosine_similarity(a, b):
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)

def compute_distances(embeddings, num_pairs=1000):
    cosine_sims, l2_dists = [], []
    for _ in range(num_pairs):
        i, j = random.sample(range(len(embeddings)), 2)
        a, b = embeddings[i], embeddings[j]
        cosine_sims.append(stable_cosine_similarity(a, b))
        l2_dists.append(euclidean(a, b))
    return np.array(cosine_sims), np.array(l2_dists)

def main():
    vocab_path = Path("/mnt/research/scratch/bkroje/molmo_data/saved_embeddings/vocab_embeddings.npy")
    vocab_embeddings = np.load(vocab_path).astype(np.float32)

    cos_sims, l2_dists = compute_distances(vocab_embeddings)

    plt.hist(cos_sims, bins=50, alpha=0.7)
    plt.title("Intra-Modality Cosine Similarity")
    plt.savefig("simple_cosine_hist.png")
    plt.close()

    plt.hist(l2_dists, bins=50, alpha=0.7)
    plt.title("Intra-Modality L2 Distance")
    plt.savefig("simple_l2_hist.png")
    plt.close()

if __name__ == "__main__":
    main()

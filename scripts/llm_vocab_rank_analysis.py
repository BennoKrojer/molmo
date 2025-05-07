"""Analyze the rank of the LLM vocabulary space using SVD and visualize the spectrum."""
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python llm_vocab_rank_analysis.py <vocab_embeddings_path>")
        sys.exit(1)
    
    vocab_embeddings_path = Path(sys.argv[1])
    
    # Load vocabulary embeddings
    log.info(f"Loading vocabulary embeddings from {vocab_embeddings_path}")
    vocab_embeddings = np.load(vocab_embeddings_path)
    log.info(f"Vocabulary embeddings shape: {vocab_embeddings.shape}")
    
    # Center the embeddings
    vocab_embeddings = vocab_embeddings - np.mean(vocab_embeddings, axis=0)
    
    # Perform SVD
    log.info("Performing SVD...")
    U, s, Vh = svd(vocab_embeddings, full_matrices=False)
    
    # Calculate explained variance
    explained_variance = s**2 / np.sum(s**2)
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot the singular value spectrum
    plt.figure(figsize=(12, 6))
    
    # Plot singular values
    plt.subplot(1, 2, 1)
    plt.plot(s, 'b-', linewidth=2)
    plt.title('Singular Value Spectrum')
    plt.xlabel('Component')
    plt.ylabel('Singular Value')
    plt.grid(True)
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_variance, 'r-', linewidth=2)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    
    # Save the plot
    output_path = vocab_embeddings_path.parent / 'vocab_rank_analysis.png'
    plt.tight_layout()
    plt.savefig(output_path)
    log.info(f"Saved analysis plot to {output_path}")
    
    # Print some statistics
    log.info(f"Total number of components: {len(s)}")
    log.info(f"Number of components explaining 90% variance: {np.argmax(cumulative_variance >= 0.9) + 1}")
    log.info(f"Number of components explaining 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
    log.info(f"Number of components explaining 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")

if __name__ == "__main__":
    main()

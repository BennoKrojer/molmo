"""Extract vocabulary and visual embeddings from Molmo for analysis."""
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from olmo.config import ModelConfig
from olmo.model import Molmo
from olmo.data.pixmo_datasets import PixMoCap
from olmo.util import resource_path
from olmo.data import build_mm_preprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def stable_cosine_similarity(a, b):
    """Compute cosine similarity between two vectors with numerical stability."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    norm_a = np.sqrt(np.sum(a * a) + 1e-8)
    norm_b = np.sqrt(np.sum(b * b) + 1e-8)
    
    a_norm = a / norm_a
    b_norm = b / norm_b
    
    dot_product = np.sum(a_norm * b_norm)
    return np.clip(dot_product, -1.0, 1.0)

def compute_and_plot_similarities(embeddings, title, output_path, num_pairs=10000):
    """Compute and plot cosine similarities for a set of embeddings."""
    similarities = []
    
    # Convert to torch tensor for efficient computation
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    # Normalize embeddings for cosine similarity
    embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, dim=-1)
    
    for _ in range(num_pairs):
        i, j = np.random.choice(len(embeddings), 2, replace=False)
        emb1_norm = embeddings_norm[i]
        emb2_norm = embeddings_norm[j]
        
        # Compute cosine similarity using normalized dot product
        sim = torch.dot(emb1_norm, emb2_norm).item()
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    # Log statistics
    log.info(f"{title}:")
    log.info(f"Mean similarity: {np.mean(similarities):.4f}")
    log.info(f"Std similarity: {np.std(similarities):.4f}")
    
    # Create figure
    plt.figure(figsize=(8, 5))
    plt.hist(similarities, bins=50, alpha=0.7, range=(-1, 1))
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python extract_embeddings.py <checkpoint_path> <output_dir>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log.info(f"Loading model from {checkpoint_path}")
    model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    # Create preprocessor
    model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # Override system prompt kind to avoid length conditioning
    model_config.system_prompt_kind = "style"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    # Extract vocabulary embeddings
    log.info("Extracting vocabulary embeddings...")
    vocab_embeddings = model.transformer.wte.embedding.detach().cpu().numpy()
    log.info(f"Vocabulary embeddings shape: {vocab_embeddings.shape}")
    # np.save(output_dir / "vocab_embeddings.npy", vocab_embeddings)
    
    # Compute and plot vocabulary similarities
    compute_and_plot_similarities(
        vocab_embeddings,
        "Vocabulary Embedding Similarities",
        output_dir / "vocab_similarities.png"
    )

    # Load PixMoCap dataset
    log.info("Loading PixMoCap dataset...")
    dataset = PixMoCap(split="train", mode="captions")
    
    # Extract visual embeddings for a few thousand images
    num_images = min(200, len(dataset))
    visual_embeddings = []
    
    log.info(f"Extracting visual embeddings for {num_images} images...")
    for i in tqdm(range(num_images)):
        example = dataset.get(i, np.random)
        
        # Create example with empty prompt
        example = {
            "image": example["image"],
            "messages": {
                "messages": [""],  # Empty prompt
                "style": "long_caption"
            }
        }
        
        # Preprocess example
        batch = preprocessor(example, rng=np.random)
        
        # Get visual embeddings directly from vision backbone
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Get image features from vision backbone
                image_features = model.vision_backbone(
                    torch.tensor(batch.get("images")).unsqueeze(0).cuda(),
                    torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                )
                
                # Reshape to (num_patches, d_model)
                image_features = image_features.view(-1, image_features.shape[-1])
                
                # Sample 5 random tokens from this image
                indices = np.random.choice(len(image_features), min(5, len(image_features)), replace=False)
                image_features = image_features[indices]
                
                visual_embeddings.append(image_features.detach().cpu().numpy())
    
    # Stack all visual embeddings
    visual_embeddings = np.concatenate(visual_embeddings, axis=0)
    log.info(f"Final visual embeddings shape: {visual_embeddings.shape}")
    np.save(output_dir / "visual_embeddings.npy", visual_embeddings)
    
    # Compute and plot visual similarities
    compute_and_plot_similarities(
        visual_embeddings,
        "Visual Embedding Similarities",
        output_dir / "visual_similarities.png"
    )
    
    # Compute and plot inter-modality similarities
    log.info("Computing inter-modality similarities...")
    inter_similarities = []
    
    # Convert to torch tensors and normalize for efficient computation
    vocab_embeddings_tensor = torch.tensor(vocab_embeddings, dtype=torch.float32)
    visual_embeddings_tensor = torch.tensor(visual_embeddings, dtype=torch.float32)
    vocab_embeddings_norm = torch.nn.functional.normalize(vocab_embeddings_tensor, dim=-1)
    visual_embeddings_norm = torch.nn.functional.normalize(visual_embeddings_tensor, dim=-1)
    
    for _ in range(10000):
        i = np.random.randint(0, len(vocab_embeddings))
        j = np.random.randint(0, len(visual_embeddings))
        
        # Compute cosine similarity using normalized dot product
        sim = torch.dot(vocab_embeddings_norm[i], visual_embeddings_norm[j]).item()
        inter_similarities.append(sim)
    
    inter_similarities = np.array(inter_similarities)
    
    # Log statistics
    log.info("Inter-modality similarities:")
    log.info(f"Mean similarity: {np.mean(inter_similarities):.4f}")
    log.info(f"Std similarity: {np.std(inter_similarities):.4f}")
    
    # Plot inter-modality similarities
    plt.figure(figsize=(8, 5))
    plt.hist(inter_similarities, bins=50, alpha=0.7, range=(-1, 1))
    plt.title("Inter-modality Embedding Similarities")
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / "inter_modality_similarities.png")
    plt.close()
    
    log.info("Successfully extracted and saved all embeddings and similarity plots!")

if __name__ == "__main__":
    main() 
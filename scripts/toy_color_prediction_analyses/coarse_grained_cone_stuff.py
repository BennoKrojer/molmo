"""Extract vocabulary and visual embeddings from Molmo for analysis."""
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import gc
import json

from olmo.config import ModelConfig
from olmo.model import Molmo
from olmo.data.pixmo_datasets import PixMoCap
from olmo.util import resource_path
from olmo.data import build_mm_preprocessor
import argparse
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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


def compute_layer_similarities(hidden_states, visual_positions, title_prefix, results_dir):
    """Compute and plot similarities across all layers for visual tokens."""
    
    num_layers = len(hidden_states)
    layer_similarities = []
    
    log.info(f"Computing layer similarities for {num_layers} layers...")
    
    for layer_idx in range(num_layers):
        layer_hidden_states = hidden_states[layer_idx]  # Shape: (batch_size, seq_len, d_model)
        
        # Extract visual tokens from this layer
        visual_embeddings = []
        for batch_idx in range(layer_hidden_states.shape[0]):
            if batch_idx < len(visual_positions):
                batch_visual_positions = visual_positions[batch_idx]
                for pos in batch_visual_positions:
                    if pos >= 0 and pos < layer_hidden_states.shape[1]:
                        visual_embeddings.append(layer_hidden_states[batch_idx, pos].detach().cpu().numpy())
        
        if len(visual_embeddings) == 0:
            log.warning(f"No visual embeddings found for layer {layer_idx}")
            layer_similarities.append(np.nan)
            continue
            
        visual_embeddings = np.array(visual_embeddings)
        
        # Compute similarities for this layer
        similarities = []
        embeddings_tensor = torch.tensor(visual_embeddings, dtype=torch.float32)
        embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, dim=-1)
        
        num_pairs = min(10000, len(visual_embeddings) * (len(visual_embeddings) - 1) // 2)
        for _ in range(num_pairs):
            i, j = np.random.choice(len(visual_embeddings), 2, replace=False)
            emb1_norm = embeddings_norm[i]
            emb2_norm = embeddings_norm[j]
            
            sim = torch.dot(emb1_norm, emb2_norm).item()
            similarities.append(sim)
        
        layer_similarities.append(np.mean(similarities))
        
        # Save individual layer similarity plot
        if layer_idx % 5 == 0:  # Save every 5th layer to avoid too many files
            plt.figure(figsize=(8, 5))
            plt.hist(similarities, bins=50, alpha=0.7, range=(-1, 1))
            plt.title(f"{title_prefix} Layer {layer_idx} Similarities")
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(results_dir / f"{title_prefix.lower()}_layer_{layer_idx}_similarities.png")
            plt.close()
    
    # Plot layer similarities across all layers
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_layers), layer_similarities, 'b-', linewidth=2)
    plt.title(f"{title_prefix} Similarities Across Layers")
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / f"layer_similarities_combined_{title_prefix.lower()}.png")
    plt.close()
    
    # Save the similarities as numpy array
    np.save(results_dir / f"layer_similarities_{title_prefix.lower()}.npy", np.array(layer_similarities))
    
    return layer_similarities


def extract_visual_positions_from_batch(batch, preprocessor):
    """Extract visual token positions from a preprocessed batch."""
    visual_positions = []
    
    # Get image input indices
    image_input_idx = batch.get("image_input_idx")
    if image_input_idx is not None:
        if hasattr(image_input_idx, 'tolist'):
            image_input_idx = image_input_idx.tolist()
        elif hasattr(image_input_idx, '__iter__'):
            image_input_idx = list(image_input_idx)
        else:
            image_input_idx = [image_input_idx]
        
        # Handle nested structure
        if isinstance(image_input_idx[0], (list, tuple)):
            for batch_positions in image_input_idx:
                valid_positions = [pos for pos in batch_positions if pos >= 0]
                visual_positions.append(valid_positions)
        else:
            valid_positions = [pos for pos in image_input_idx if pos >= 0]
            visual_positions.append(valid_positions)
    
    return visual_positions


def delete_vision_backbone(model):
    """Delete the vision backbone from the model to free up memory."""
    if hasattr(model, 'vision_backbone') and model.vision_backbone is not None:
        log.info("Deleting vision backbone to free memory...")
        del model.vision_backbone
        model.vision_backbone = None
        clear_gpu_memory()


def main():

    # argparse stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, default="molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336/step12000-unsharded")
    parser.add_argument("--text_model_name", type=str, required=True, default="Qwen/Qwen2-7B")
    args = parser.parse_args()

    # Hardcoded parameters
    checkpoint_path = args.checkpoint_path
    text_model_name = args.text_model_name
     
    # Setup results directory
    ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
    results_dir = Path("analysis_results/anisotropy") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Results will be saved to: {results_dir}")

    # Check if cached embeddings exist
    vocab_embeddings_path = results_dir / "vocab_embeddings.npy"
    visual_embeddings_path = results_dir / "visual_embeddings.npy"
    
    # Load cached layer-wise text embeddings
    text_embeddings_dir = Path("analysis_results/cached_text_embeddings") / text_model_name.replace("/", "_")
    if not text_embeddings_dir.exists():
        log.error(f"Text embeddings directory not found: {text_embeddings_dir}")
        log.error("Please run the LLM analysis script first to generate cached text embeddings")
        return
    
    # Load text embeddings metadata
    with open(text_embeddings_dir / "metadata.json", "r") as f:
        text_metadata = json.load(f)
    
    log.info(f"Loading cached text embeddings from: {text_embeddings_dir}")
    log.info(f"Text model: {text_metadata['model_name']}")
    log.info(f"Text layers: {text_metadata['num_layers']}")
    log.info(f"Text sentences: {text_metadata['num_sentences']}")
    
    # Load static vocabulary embeddings (for baseline comparison)
    static_vocab_embeddings = np.load(text_embeddings_dir / "layer_0_static_vocab.npy")
    log.info(f"Loaded static vocabulary embeddings: {static_vocab_embeddings.shape}")
    
    # Load layer-wise text embeddings
    layerwise_text_embeddings = {}
    for layer_idx in range(1, text_metadata['num_layers'] + 1):
        layer_file = text_embeddings_dir / f"layer_{layer_idx}_text_embeddings.npy"
        if layer_file.exists():
            layerwise_text_embeddings[layer_idx] = np.load(layer_file)
            log.info(f"Loaded layer {layer_idx} text embeddings: {layerwise_text_embeddings[layer_idx].shape}")
        else:
            log.warning(f"Text embeddings not found for layer {layer_idx}")
    
    # Step 1: Compute and cache visual embeddings if not exist
    if vocab_embeddings_path.exists() and visual_embeddings_path.exists():
        log.info("Found cached embeddings, loading from disk...")
        vocab_embeddings = np.load(vocab_embeddings_path)
        visual_embeddings = np.load(visual_embeddings_path)
        log.info(f"Loaded vocabulary embeddings shape: {vocab_embeddings.shape}")
        log.info(f"Loaded visual embeddings shape: {visual_embeddings.shape}")
    else:
        log.info("No cached embeddings found, computing from scratch...")
        
        # Load model
        log.info(f"Loading model from {checkpoint_path}")
        model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
        model.eval()

        # Create preprocessor
        if "hf:" in checkpoint_path:
            model_config = model.config
        else:
            model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
        # Override system prompt kind to avoid length conditioning
        model_config.system_prompt_kind = "style_and_length"
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
        np.save(vocab_embeddings_path, vocab_embeddings)
        
        # Load PixMoCap dataset
        log.info("Loading PixMoCap dataset...")
        dataset = PixMoCap(split="train", mode="captions")
        
        # Extract visual embeddings for a few hundred images
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
        np.save(visual_embeddings_path, visual_embeddings)
        
        # Clean up model and preprocessor
        del model, preprocessor
        clear_gpu_memory()
    
    # Step 2: Compute basic similarities (unchanged from original)
    log.info("Computing vocabulary similarities...")
    compute_and_plot_similarities(
        vocab_embeddings,
        "Vocabulary Embedding Similarities",
        results_dir / "vocab_similarities.png"
    )

    compute_and_plot_similarities(
        visual_embeddings,
        "Visual Embedding Similarities",
        results_dir / "visual_similarities.png"
    )
    
    # Compute and plot inter-modality similarities using static vocabulary
    log.info("Computing inter-modality similarities with static vocabulary...")
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
    log.info("Inter-modality similarities (static vocabulary):")
    log.info(f"Mean similarity: {np.mean(inter_similarities):.4f}")
    log.info(f"Std similarity: {np.std(inter_similarities):.4f}")
    
    # Plot inter-modality similarities
    plt.figure(figsize=(8, 5))
    plt.hist(inter_similarities, bins=50, alpha=0.7, range=(-1, 1))
    plt.title("Inter-modality Embedding Similarities (Static Vocabulary)")
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(results_dir / "inter_modality_similarities_static.png")
    plt.close()
    
    # Step 3: Compute layer-wise similarities using cached visual embeddings
    log.info("Starting layer-wise similarity analysis...")
    
    # Setup cache directory for hidden states
    hidden_states_cache_dir = results_dir / "cached_hidden_states"
    hidden_states_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if hidden states are already cached
    num_examples_to_process = 200  # Process 50 examples for layer analysis
    hidden_states_exist = all(
        (hidden_states_cache_dir / f"ex_{i}").exists() 
        for i in range(num_examples_to_process)
    )
    
    if not hidden_states_exist:
        log.info("Hidden states not cached, computing and saving them...")
        
        # Load model for hidden states extraction
        log.info(f"Loading model for hidden states extraction from {checkpoint_path}")
        model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
        model.eval()
        
        # Create preprocessor
        if "hf:" in checkpoint_path:
            model_config = model.config
        else:
            model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
        model_config.system_prompt_kind = "style_and_length"
        preprocessor = build_mm_preprocessor(
            model_config,
            for_inference=True,
            shuffle_messages=False,
            is_training=False,
            require_image_features=True
        )
        
        # Load dataset
        dataset = PixMoCap(split="train", mode="captions")
        
        # Process examples and save hidden states
        for i in tqdm(range(num_examples_to_process), desc="Processing examples for hidden states"):
            example_cache_dir = hidden_states_cache_dir / f"ex_{i}"
            example_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Process example
            example = dataset.get(i, np.random)
            example = {
                "image": example["image"],
                "messages": {
                    "messages": [""],  # Empty prompt
                    "style": "long_caption"
                }
            }
            
            batch = preprocessor(example, rng=np.random)
            
            # Extract visual positions
            visual_positions = extract_visual_positions_from_batch(batch, preprocessor)
            
            # Save visual positions for this example
            np.save(example_cache_dir / "visual_positions.npy", np.array(visual_positions, dtype=object))
            
            # Forward pass with hidden states
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    output = model(
                        input_ids=torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda(),
                        attention_mask=None,
                        attention_bias=None,
                        images=torch.tensor(batch.get("images")).unsqueeze(0).cuda(),
                        image_masks=torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None,
                        image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                        output_hidden_states=True,
                        return_visual_embeddings=False
                    )
            
            # Save hidden states for each layer
            hidden_states = output.hidden_states
            for layer_idx, layer_hidden_states in enumerate(hidden_states):
                layer_path = example_cache_dir / f"layer_{layer_idx}.npy"
                np.save(layer_path, layer_hidden_states.detach().cpu().numpy())
            
            # Save metadata
            metadata = {
                "num_layers": len(hidden_states),
                "sequence_length": hidden_states[0].shape[1],
                "hidden_size": hidden_states[0].shape[2]
            }
            with open(example_cache_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            clear_gpu_memory()
        
        # Clean up model
        del model, preprocessor
        clear_gpu_memory()
        log.info("Hidden states saved to cache")
    
    else:
        log.info("Found cached hidden states, skipping computation")
    
    # Step 4: Load hidden states and compute layer-wise similarities
    log.info("Loading cached hidden states and computing layer-wise similarities...")
    
    # Load metadata to get number of layers
    with open(hidden_states_cache_dir / "ex_0" / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_layers = metadata["num_layers"]
    
    # Initialize similarity tracking - now including layer 0 (input embeddings)
    inter_image_visual_similarities = []  # One random visual token per image
    intra_image_visual_similarities = []  # All visual tokens within each image
    visual_to_text_similarities = []      # Visual tokens vs layer-wise text embeddings
    visual_to_text_max_similarities = []  # Maximum similarity for each visual token to any text token
    
    # Layer 0: Compare input embeddings (vocabulary matrices)
    log.info("Computing layer 0 similarities (input embeddings)...")
    
    # For visual input embeddings, use the cached visual embeddings from vision backbone
    # These represent the "input" visual tokens before transformer processing
    
    # Inter-image visual similarities for layer 0 (using cached visual embeddings)
    if len(visual_embeddings) > 1:
        # Sample one embedding per "image" - since we have 5 per image, group them
        num_images_in_cache = len(visual_embeddings) // 5  # 5 embeddings per image
        layer0_inter_image_embeddings = []
        
        for img_idx in range(num_images_in_cache):
            # Get 5 embeddings for this image and pick one randomly
            start_idx = img_idx * 5
            end_idx = start_idx + 5
            img_embeddings = visual_embeddings[start_idx:end_idx]
            random_idx = np.random.randint(0, len(img_embeddings))
            layer0_inter_image_embeddings.append(img_embeddings[random_idx])
        
        # Compute similarities using matrix operations
        layer0_inter_image_embeddings = np.array(layer0_inter_image_embeddings)
        embeddings_tensor = torch.tensor(layer0_inter_image_embeddings, dtype=torch.float32)
        embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, dim=-1)
        
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        upper_triangular = torch.triu(similarity_matrix, diagonal=1)
        similarities = upper_triangular[upper_triangular != 0].cpu().numpy()
        
        if len(similarities) > 10000:
            similarities = np.random.choice(similarities, 10000, replace=False)
        
        inter_image_visual_similarities.append(np.mean(similarities))
    else:
        inter_image_visual_similarities.append(np.nan)
    
    # Intra-image visual similarities for layer 0 (within each image's 5 embeddings)
    layer0_intra_similarities = []
    for img_idx in range(num_images_in_cache):
        start_idx = img_idx * 5
        end_idx = start_idx + 5
        img_embeddings = visual_embeddings[start_idx:end_idx]
        
        if len(img_embeddings) > 1:
            embeddings_tensor = torch.tensor(img_embeddings, dtype=torch.float32)
            embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, dim=-1)
            
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            upper_triangular = torch.triu(similarity_matrix, diagonal=1)
            similarities = upper_triangular[upper_triangular != 0].cpu().numpy()
            
            if len(similarities) > 0:
                layer0_intra_similarities.extend(similarities)
    
    if layer0_intra_similarities:
        intra_image_visual_similarities.append(np.mean(layer0_intra_similarities))
    else:
        intra_image_visual_similarities.append(np.nan)
    
    # Visual-to-text similarities for layer 0 (visual input vs text input embeddings)
    # Use cached visual embeddings vs static vocabulary embeddings
    visual_embeddings_tensor = torch.tensor(visual_embeddings, dtype=torch.float32)
    visual_embeddings_norm = torch.nn.functional.normalize(visual_embeddings_tensor, dim=-1)
    
    static_vocab_tensor = torch.tensor(static_vocab_embeddings, dtype=torch.float32)
    static_vocab_norm = torch.nn.functional.normalize(static_vocab_tensor, dim=-1)
    
    # Compute cross-modal similarity matrix
    cross_modal_similarity = torch.mm(visual_embeddings_norm, static_vocab_norm.t())
    
    # Sample 10000 random pairs
    num_visual = cross_modal_similarity.shape[0]
    num_vocab = cross_modal_similarity.shape[1]
    
    visual_indices = torch.randint(0, num_visual, (10000,))
    vocab_indices = torch.randint(0, num_vocab, (10000,))
    
    text_similarities = cross_modal_similarity[visual_indices, vocab_indices].cpu().numpy()
    visual_to_text_similarities.append(np.mean(text_similarities))
    
    # Compute maximum similarities for each visual token
    max_similarities = torch.max(cross_modal_similarity, dim=1)[0].cpu().numpy()
    visual_to_text_max_similarities.append(np.mean(max_similarities))
    
    log.info(f"Layer 0 (input embeddings) - Inter-image: {inter_image_visual_similarities[0]:.4f}, Intra-image: {intra_image_visual_similarities[0]:.4f}, Visual-to-text: {visual_to_text_similarities[0]:.4f}, Visual-to-text-max: {visual_to_text_max_similarities[0]:.4f}")
    
    # Process transformer layers 1 through N
    for layer_idx in tqdm(range(num_layers), desc="Computing transformer layer similarities"):
        
        # 1. Inter-image visual similarities (one random visual token per image)
        inter_image_embeddings = []
        
        # 2. Intra-image visual similarities (all visual tokens within each image)
        intra_image_similarities_for_layer = []
        
        # 3. Visual-to-text similarities (collect all visual tokens)
        all_visual_embeddings_for_text = []
        
        # Collect embeddings from all examples for this layer
        for ex_idx in range(num_examples_to_process):
            example_cache_dir = hidden_states_cache_dir / f"ex_{ex_idx}"
            
            # Load hidden states for this layer
            layer_hidden_states = np.load(example_cache_dir / f"layer_{layer_idx}.npy")
            
            # Load visual positions
            visual_positions = np.load(example_cache_dir / "visual_positions.npy", allow_pickle=True)
            
            # Extract visual tokens from this layer for this example
            if len(visual_positions) > 0:
                batch_visual_positions = visual_positions[0]  # First batch
                example_visual_embeddings = []
                
                for pos in batch_visual_positions:
                    if pos >= 0 and pos < layer_hidden_states.shape[1]:
                        embedding = layer_hidden_states[0, pos]  # First batch, position pos
                        example_visual_embeddings.append(embedding)
                        all_visual_embeddings_for_text.append(embedding)
                
                if len(example_visual_embeddings) > 0:
                    # For inter-image: select one random visual token from this image
                    random_idx = np.random.randint(0, len(example_visual_embeddings))
                    inter_image_embeddings.append(example_visual_embeddings[random_idx])
                    
                    # For intra-image: compute similarities within this image using matrix ops
                    if len(example_visual_embeddings) > 1:  # Need at least 2 tokens
                        example_visual_embeddings = np.array(example_visual_embeddings)
                        embeddings_tensor = torch.tensor(example_visual_embeddings, dtype=torch.float32)
                        embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, dim=-1)
                        
                        # Compute similarity matrix efficiently: normalized_embeddings @ normalized_embeddings.T
                        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
                        
                        # Extract upper triangular part (excluding diagonal) for unique pairs
                        upper_triangular = torch.triu(similarity_matrix, diagonal=1)
                        similarities = upper_triangular[upper_triangular != 0].cpu().numpy()
                        
                        if len(similarities) > 0:
                            intra_image_similarities_for_layer.extend(similarities)
        
        # Compute inter-image visual similarities using efficient matrix operations
        if len(inter_image_embeddings) > 1:
            inter_image_embeddings = np.array(inter_image_embeddings)
            embeddings_tensor = torch.tensor(inter_image_embeddings, dtype=torch.float32)
            embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, dim=-1)
            
            # Compute full similarity matrix
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            
            # Extract upper triangular part (excluding diagonal) for unique pairs
            upper_triangular = torch.triu(similarity_matrix, diagonal=1)
            similarities = upper_triangular[upper_triangular != 0].cpu().numpy()
            
            # Sample from similarities if we have too many
            if len(similarities) > 10000:
                similarities = np.random.choice(similarities, 10000, replace=False)
            
            inter_image_visual_similarities.append(np.mean(similarities))
        else:
            inter_image_visual_similarities.append(np.nan)
        
        # Compute intra-image visual similarities
        if intra_image_similarities_for_layer:
            intra_image_visual_similarities.append(np.mean(intra_image_similarities_for_layer))
        else:
            intra_image_visual_similarities.append(np.nan)
        
        # Compute visual-to-text similarities using layer-wise text embeddings
        # Now use layer_idx + 1 to match transformer layer numbers (layer 1, 2, 3, ...)
        text_layer_idx = layer_idx + 1
        if all_visual_embeddings_for_text and text_layer_idx in layerwise_text_embeddings:
            all_visual_embeddings_for_text = np.array(all_visual_embeddings_for_text)
            visual_embeddings_tensor = torch.tensor(all_visual_embeddings_for_text, dtype=torch.float32)
            visual_embeddings_norm = torch.nn.functional.normalize(visual_embeddings_tensor, dim=-1)
            
            # Get corresponding layer text embeddings
            text_embeddings = layerwise_text_embeddings[text_layer_idx]
            text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
            text_embeddings_norm = torch.nn.functional.normalize(text_embeddings_tensor, dim=-1)
            
            # Compute cross-modal similarity matrix: visual_embeddings @ text_embeddings.T
            cross_modal_similarity = torch.mm(visual_embeddings_norm, text_embeddings_norm.t())
            
            # Sample 10000 random pairs efficiently
            num_visual = cross_modal_similarity.shape[0]
            num_text = cross_modal_similarity.shape[1]
            
            # Generate random indices
            visual_indices = torch.randint(0, num_visual, (10000,))
            text_indices = torch.randint(0, num_text, (10000,))
            
            # Extract similarities for these random pairs
            text_similarities = cross_modal_similarity[visual_indices, text_indices].cpu().numpy()
            
            visual_to_text_similarities.append(np.mean(text_similarities))
            
            # Compute maximum similarities for each visual token
            max_similarities = torch.max(cross_modal_similarity, dim=1)[0].cpu().numpy()
            visual_to_text_max_similarities.append(np.mean(max_similarities))
        else:
            visual_to_text_similarities.append(np.nan)
            visual_to_text_max_similarities.append(np.nan)
    
    # Create the final plots with updated x-axis labels
    log.info("Creating layer-wise similarity plots...")
    
    # Create layer labels: 0 (Input), 1, 2, 3, ...
    layer_labels = [0] + list(range(1, num_layers + 1))
    
    # Plot 1: Inter-image visual similarities (different images)
    plt.figure(figsize=(12, 6))
    plt.plot(layer_labels, inter_image_visual_similarities, 'b-', linewidth=2, label='Inter-image Visual')
    plt.title('Inter-image Visual Token Similarities Across Transformer Layers\n(Layer 0 = Input Embeddings, Layer 1+ = Transformer Layers)')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(layer_labels)
    plt.tight_layout()
    plt.savefig(results_dir / "inter_image_visual_similarities_across_layers.png")
    plt.close()
    
    # Plot 2: Intra-image visual similarities (same image)
    plt.figure(figsize=(12, 6))
    plt.plot(layer_labels, intra_image_visual_similarities, 'g-', linewidth=2, label='Intra-image Visual')
    plt.title('Intra-image Visual Token Similarities Across Transformer Layers\n(Layer 0 = Input Embeddings, Layer 1+ = Transformer Layers)')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(layer_labels)
    plt.tight_layout()
    plt.savefig(results_dir / "intra_image_visual_similarities_across_layers.png")
    plt.close()
    
    # Plot 3: Visual-to-text similarities (layer-wise)
    plt.figure(figsize=(12, 6))
    plt.plot(layer_labels, visual_to_text_similarities, 'r-', linewidth=2, label='Visual-to-Text (Layer-wise)')
    plt.title(f'Visual-to-Text Similarities Across Transformer Layers\n(Layer 0 = Input Embeddings, Compared to {text_model_name})')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(layer_labels)
    plt.tight_layout()
    plt.savefig(results_dir / "visual_to_text_similarities_across_layers.png")
    plt.close()
    
    # Plot 4: Combined plot with all three types
    plt.figure(figsize=(12, 6))
    plt.plot(layer_labels, inter_image_visual_similarities, 'b-', linewidth=2, label='Inter-image Visual')
    plt.plot(layer_labels, intra_image_visual_similarities, 'g-', linewidth=2, label='Intra-image Visual')
    plt.plot(layer_labels, visual_to_text_similarities, 'r-', linewidth=2, label='Visual-to-Text (Random)')
    plt.plot(layer_labels, visual_to_text_max_similarities, 'm-', linewidth=2, label='Visual-to-Text (Max)')
    plt.title(f'Visual Token Similarities Across Transformer Layers\n(Layer 0 = Input Embeddings, Text comparison: {text_model_name})')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(layer_labels)
    plt.tight_layout()
    plt.savefig(results_dir / "combined_all_similarities_across_layers.png")
    plt.close()
    
    # Save raw data
    np.save(results_dir / "inter_image_visual_similarities.npy", np.array(inter_image_visual_similarities))
    np.save(results_dir / "intra_image_visual_similarities.npy", np.array(intra_image_visual_similarities))
    np.save(results_dir / "visual_to_text_similarities.npy", np.array(visual_to_text_similarities))
    np.save(results_dir / "visual_to_text_max_similarities.npy", np.array(visual_to_text_max_similarities))
    
    log.info("Layer-wise similarity analysis complete!")
    
    # Print summary statistics
    log.info(f"Layer 0 (Input) - Inter-image: {inter_image_visual_similarities[0]:.4f}, Intra-image: {intra_image_visual_similarities[0]:.4f}, Visual-to-text: {visual_to_text_similarities[0]:.4f}, Visual-to-text-max: {visual_to_text_max_similarities[0]:.4f}")
    log.info(f"Inter-image visual similarities: mean={np.nanmean(inter_image_visual_similarities):.4f}, std={np.nanstd(inter_image_visual_similarities):.4f}")
    log.info(f"Intra-image visual similarities: mean={np.nanmean(intra_image_visual_similarities):.4f}, std={np.nanstd(intra_image_visual_similarities):.4f}")
    log.info(f"Visual-to-text similarities: mean={np.nanmean(visual_to_text_similarities):.4f}, std={np.nanstd(visual_to_text_similarities):.4f}")
    log.info(f"Visual-to-text max similarities: mean={np.nanmean(visual_to_text_max_similarities):.4f}, std={np.nanstd(visual_to_text_max_similarities):.4f}")
    
    # Also compute similarities for the original cached visual embeddings as baseline
    log.info("Computing baseline similarities using original cached visual embeddings...")
    compute_and_plot_similarities(
        visual_embeddings,
        "Original Cached Visual Embeddings (Random Subset)",
        results_dir / "original_cached_visual_similarities.png"
    )
    
    log.info("Successfully extracted and saved all embeddings and similarity plots!")

if __name__ == "__main__":
    main() 
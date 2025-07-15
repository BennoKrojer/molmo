#!/usr/bin/env python3
"""
Script to regenerate cached text embeddings with full vocabulary (no 50K limit).
This fixes the issue where cached embeddings were truncated to 50K tokens.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
import json
import argparse

def get_full_token_embeddings(model_name):
    """Get token embeddings for the full vocabulary (no truncation)."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    emb_layer = model.get_input_embeddings()
    
    # Handle different embedding layer structures
    if hasattr(emb_layer, 'weight'):
        emb = emb_layer.weight.detach().cpu().numpy()
    elif hasattr(emb_layer, 'embedding'):
        emb = emb_layer.embedding.detach().cpu().numpy()
    else:
        # Try to get embeddings directly from the layer
        emb = emb_layer.detach().cpu().numpy()
    
    vocab_size = len(tokenizer)
    actual_vocab_size = emb.shape[0]
    
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"Actual embedding matrix size: {actual_vocab_size}")
    print(f"Embedding dimension: {emb.shape[1]}")
    
    # Compute norms for diagnostics
    norms = np.linalg.norm(emb, axis=1)
    print(f"Embedding norms - mean: {norms.mean():.4f}, std: {norms.std():.4f}, min: {norms.min():.4f}, max: {norms.max():.4f}")
    
    # Return full embeddings (no truncation)
    return emb, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Regenerate cached text embeddings with full vocabulary")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-7B", 
                       help="Model name to regenerate embeddings for")
    parser.add_argument("--force", action="store_true", 
                       help="Force regeneration even if file exists")
    args = parser.parse_args()
    
    model_name = args.model_name
    
    # Setup output directory
    embeddings_dir = Path("analysis_results/cached_text_embeddings") / model_name.replace("/", "_")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = embeddings_dir / "layer_0_static_vocab.npy"
    metadata_file = embeddings_dir / "metadata.json"
    
    # Check if file exists and warn user
    if output_file.exists() and not args.force:
        print(f"Warning: {output_file} already exists!")
        print("Use --force to overwrite, or delete the file manually.")
        
        # Load existing file to show current size
        try:
            existing_emb = np.load(output_file)
            print(f"Current cached embeddings shape: {existing_emb.shape}")
        except Exception as e:
            print(f"Could not load existing file: {e}")
        
        response = input("Do you want to overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Get full embeddings
    embeddings, tokenizer = get_full_token_embeddings(model_name)
    
    # Save embeddings
    print(f"Saving full vocabulary embeddings to {output_file}")
    np.save(output_file, embeddings)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "tokenizer_vocab_size": len(tokenizer),
        "embedding_matrix_size": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "is_full_vocabulary": True,
        "note": "Generated with full vocabulary (no 50K truncation)"
    }
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved embeddings: {embeddings.shape}")
    print(f"Saved metadata to {metadata_file}")
    print("Done!")
    
    # Compare with previous version if it existed
    if embeddings.shape[0] > 50000:
        print(f"\nIMPORTANT: This full vocabulary has {embeddings.shape[0]} tokens")
        print("Previous cached version was truncated to 50,000 tokens")
        print(f"You now have access to {embeddings.shape[0] - 50000} additional tokens!")

if __name__ == "__main__":
    main() 
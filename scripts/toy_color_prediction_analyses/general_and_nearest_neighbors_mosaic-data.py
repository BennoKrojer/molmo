"""This script takes the checkpoint we trained on the simple color prediction tasks and runs inference on the train and validation splits.
For each image, and each of the 144 visual tokens in such an image:
a) we log the top5 nearest neighbor vocabularies from the LLM tokenizer
b) we check if the generated response is the actual color word
"""
import logging
import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import ColorMosaicDataset

log = logging.getLogger(__name__)

def decode_token(tokenizer, idx):
    """Decode a token and ensure it's a proper Unicode string."""
    token = tokenizer.decode([int(idx)])
    # Convert to actual characters by encoding and decoding through utf-8
    return token.encode('utf-8').decode('utf-8')

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_split(model, preprocessor, dataset, num_images, prompt, use_n_token_only):
    """Process a dataset split and return results."""
    split_results = []
    
    # Process each image
    for i in tqdm(range(num_images)):
        example_data = dataset.get(i, np.random)
        color_sequence = example_data["metadata"]["color_sequence"]
        
        # Create example with the provided prompt
        example = {
            "image": example_data["image"],
            "messages": {
                "messages": [prompt],
                "style": "none"
            }
        }

        # Preprocess example
        batch = preprocessor(example, rng=np.random)

        # Initialize image results
        image_results = {
            "image_idx": i,
            "true_color_sequence": color_sequence,
            "chunks": []
        }

        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                
                image_features, tokens_before_MLP = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                if use_n_token_only:
                    image_features = image_features[:, :, use_n_token_only, :]
                image_results["feature_shape"] = list(image_features.shape)
                
                # Get token embeddings from the model
                token_embeddings = model.transformer.wte.embedding
                
                # Reshape image features to combine batch and chunks dimensions
                batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)
                
                # Normalize the embeddings for cosine similarity
                image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
                token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)
                
                # Compute cosine similarity for each patch
                similarity = torch.matmul(image_features_norm, token_embeddings_norm.T)
                
                # Get top-5 most similar tokens for each patch
                top_k = 5
                top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)
                
                # Store results for each chunk
                for chunk_idx in range(num_chunks):
                    chunk_results = {
                        "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                        "patches": []
                    }
                    
                    for patch_idx in range(patches_per_chunk):
                        patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]
                        
                        patch_results = {
                            "patch_idx": patch_idx,
                            "nearest_neighbors": [
                                {"token": token, "similarity": float(sim)}
                                for token, sim in zip(patch_tokens, patch_values)
                            ]
                        }
                        chunk_results["patches"].append(patch_results)
                    
                    image_results["chunks"].append(chunk_results)

                # Clear intermediate tensors
                del similarity, image_features_norm, token_embeddings_norm
                clear_gpu_memory()

                # generated output - increase max_steps for multiple tokens
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()
                output = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                    max_steps=150,  # Increased for multiple color tokens
                    is_distributed=False
                )
                token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                decoded = preprocessor.tokenizer.decode(token_ids[0])
                image_results["generated_response"] = decoded

                # Clear GPU tensors
                del images_tensor, image_masks_tensor, image_features, input_ids, output
                clear_gpu_memory()

        split_results.append(image_results)
        
        # Periodically save results to avoid losing progress
        if (i + 1) % 50 == 0:
            temp_results = {
                "partial_results": True,
                "processed_images": i + 1,
                "images": split_results
            }
            temp_file = f"temp_results_{i + 1}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(temp_results, f, indent=2, ensure_ascii=False)
    
    return split_results

def main():
    # Hardcoded parameters
    checkpoint_path = "molmo_data/checkpoints/caption-prompt_mosaic-image/step3000-unsharded"
    # checkpoint_path = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize/step3000-unsharded"
    prompt = "What is the sequence of colors in this grid of colors, read from left to right like a page?"
    print(f"Prompt: {prompt}")

    # Setup results directory
    ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
    results_dir = Path("analysis_results/nearest_neighbors") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    # Create preprocessor
    if "hf:" in checkpoint_path:
        model_config = model.config
    else:
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

    use_n_token_only = model_config.vision_backbone.use_n_token_only

    # Initialize results dictionary
    all_results = {
        "checkpoint": checkpoint_path,
        "prompt": prompt,
        "dataset": "ColorMosaicDataset",
        "splits": {}
    }

    # Process train split
    log.info("Processing train split...")
    train_dataset = ColorMosaicDataset(split="train")
    num_train_images = min(200, len(train_dataset))
    train_results = process_split(model, preprocessor, train_dataset, num_train_images, prompt, use_n_token_only)
    all_results["splits"]["train"] = {
        "num_images": num_train_images,
        "images": train_results
    }
    
    # Clear memory before processing validation split
    clear_gpu_memory()

    # Process validation split
    log.info("Processing validation split...")
    val_dataset = ColorMosaicDataset(split="validation")
    num_val_images = min(200, len(val_dataset))
    val_results = process_split(model, preprocessor, val_dataset, num_val_images, prompt, use_n_token_only)
    all_results["splits"]["validation"] = {
        "num_images": num_val_images,
        "images": val_results
    }



    # Save final results
    output_file = results_dir / "nearest_neighbors_analysis_color_names_mosaic.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    log.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

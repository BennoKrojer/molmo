#!/usr/bin/env python3
"""
Patchscopes with Descriptive Few-Shot Prompt for Visual Tokens

Instead of identity prompt (X->X), use a descriptive prompt that teaches
the model to explain/describe what something is:

    Dog: Dog is a type of animal, domesticated and a common pet, with four legs and fur.
    Mathematics: Mathematics is a field of study concerned with numbers, quantities, and shapes.
    Paris: Paris is the capital city of France, known for the Eiffel Tower and art museums.
    X:

Then patch a vision token's hidden state into "X" and let the model generate
a free-form description. This might reveal what the visual content encodes.

Output format matches logitlens/NN for unified viewer integration:
- One JSON per layer: patchscopes_layer{N}.json
- Structure: {checkpoint, split, num_images, layer_idx, results: [{image_idx, chunks: [{patches}]}]}
- Each patch has: patch_idx, patch_row, patch_col, description (single string)

Usage:
    # Standard run (10 random patches per image, matches other analysis)
    python scripts/analysis/patchscopes/patchscopes_descriptive.py \
        --ckpt-path <path> --num-images 10 --layers 0,2,4,8,16 --num-patches 10

    # Legacy mode (center patches)
    python scripts/analysis/patchscopes/patchscopes_descriptive.py \
        --ckpt-path <path> --num-images 10 --layers 0,2,4,8,16 --sample-center 0.2
"""

import argparse
import gc
import json
import math
import os
import random
import time
import torch
from pathlib import Path

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap


# Descriptive few-shot prompt
DESCRIPTIVE_PROMPT = """Dog: Dog is a type of animal, domesticated and a common pet, with four legs and fur.
Mathematics: Mathematics is a field of study concerned with numbers, quantities, and shapes.
Paris: Paris is the capital city of France, known for the Eiffel Tower and art museums.
Blue: Blue is a color often associated with the sky and ocean, calming and serene.
X: """


def get_transformer_blocks(model):
    """Get transformer blocks from model."""
    if hasattr(model, 'module'):
        return model.module.transformer.blocks
    else:
        return model.transformer.blocks


class PatchAndGenerateHook:
    """
    Hook to patch hidden states and allow generation to continue.
    """
    def __init__(self, patch_position, patch_hidden_state):
        self.patch_position = patch_position
        self.patch_hidden_state = patch_hidden_state
        self.handle = None
        self.call_count = 0

    def hook_fn(self, module, args):
        """Patch on EVERY forward pass to replace X with visual hidden state."""
        if isinstance(args, tuple) and len(args) > 0:
            hidden_states = args[0].clone()
            # Only patch if we have the full sequence (not single token generation)
            if hidden_states.shape[1] > self.patch_position:
                hidden_states[:, self.patch_position, :] = self.patch_hidden_state
                self.call_count += 1
                return (hidden_states,) + args[1:]
        return None

    def register(self, module):
        self.handle = module.register_forward_pre_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def generate_with_patch(model, tokenizer, prompt_ids, patch_hidden_state,
                        layer_idx, patch_position, max_new_tokens=20, device="cuda"):
    """
    Generate text with a patched hidden state.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt_ids: Token IDs for the prompt [1, seq_len]
        patch_hidden_state: Hidden state to patch [hidden_dim]
        layer_idx: Layer to patch at
        patch_position: Position to patch (the "X" token)
        max_new_tokens: How many tokens to generate
        device: Device

    Returns:
        generated_text: The generated continuation
        generated_tokens: List of generated token strings
    """
    blocks = get_transformer_blocks(model)

    # Register patch hook
    hook = PatchAndGenerateHook(patch_position, patch_hidden_state.unsqueeze(0))
    hook.register(blocks[layer_idx])

    generated_ids = prompt_ids.clone()
    generated_tokens = []

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            # First forward pass with patching
            output = model(input_ids=generated_ids)
            next_token = output.logits[0, -1, :].argmax().item()
            generated_tokens.append(tokenizer.decode([next_token]))
            generated_ids = torch.cat([
                generated_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

            # Continue generating WITH hook still active
            for _ in range(max_new_tokens - 1):
                output = model(input_ids=generated_ids)
                next_token = output.logits[0, -1, :].argmax().item()

                # Stop on EOS or newline
                token_str = tokenizer.decode([next_token])
                if next_token == tokenizer.eos_token_id or '\n' in token_str:
                    break

                generated_tokens.append(token_str)
                generated_ids = torch.cat([
                    generated_ids,
                    torch.tensor([[next_token]], device=device)
                ], dim=1)

    # Remove hook after ALL generation is complete
    hook.remove()

    generated_text = ''.join(generated_tokens)
    return generated_text, generated_tokens


def run_descriptive_patchscopes(model, tokenizer, visual_hidden_states,
                                 layer_idx, device, max_new_tokens=20, patch_indices=None):
    """
    Run descriptive Patchscopes for visual tokens at a given layer.

    Args:
        model: The model
        tokenizer: Tokenizer
        visual_hidden_states: [num_patches, hidden_dim]
        layer_idx: Layer to patch at
        device: Device
        max_new_tokens: Max tokens to generate per patch
        patch_indices: Optional list of patch indices to process (default: all)

    Returns:
        List of dicts with patch info and generated descriptions
    """
    num_patches = visual_hidden_states.shape[0]

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(DESCRIPTIVE_PROMPT)
    prompt_ids = torch.tensor(prompt_tokens, device=device).unsqueeze(0)

    # Find position of "X" - prompt ends with "X: " which tokenizes as ["X", ":", " "]
    # So X is at position -3
    patch_position = len(prompt_tokens) - 3

    results = []

    # Use provided indices or all patches
    indices_to_process = patch_indices if patch_indices is not None else range(num_patches)

    for patch_idx in indices_to_process:
        patch_hs = visual_hidden_states[patch_idx]

        generated_text, generated_tokens = generate_with_patch(
            model, tokenizer, prompt_ids, patch_hs,
            layer_idx, patch_position, max_new_tokens, device
        )

        # Calculate grid position
        grid_size = int(math.sqrt(num_patches))
        row = patch_idx // grid_size
        col = patch_idx % grid_size

        results.append({
            "patch_idx": patch_idx,
            "patch_row": row,
            "patch_col": col,
            "generated_text": generated_text.strip(),
            "generated_tokens": generated_tokens,
        })

    return results


def get_center_patch_indices(grid_size, sample_fraction):
    """Get indices of center patches based on sample fraction."""
    if sample_fraction >= 1.0:
        return None  # Process all patches

    # Calculate center region size
    center_size = int(grid_size * math.sqrt(sample_fraction))
    if center_size < 1:
        center_size = 1

    # Calculate start/end of center region
    start = (grid_size - center_size) // 2
    end = start + center_size

    # Get indices of center patches
    indices = []
    for row in range(start, end):
        for col in range(start, end):
            indices.append(row * grid_size + col)

    return indices


def get_random_patch_indices(total_patches, num_patches, seed=None):
    """Get random patch indices for consistent sampling across layers.

    Args:
        total_patches: Total number of patches (e.g., 576 for 24x24 grid)
        num_patches: Number of patches to sample
        seed: Random seed for reproducibility

    Returns:
        Sorted list of patch indices
    """
    if num_patches >= total_patches:
        return list(range(total_patches))

    rng = random.Random(seed)
    indices = rng.sample(range(total_patches), num_patches)
    return sorted(indices)  # Sort for consistent patch ordering


def process_images(model, tokenizer, preprocessor, dataset, num_images,
                   layers, device, max_new_tokens=20, sample_center=1.0,
                   num_patches=None, seed=42):
    """Process multiple images and generate descriptions for each patch.

    Args:
        model: The VLM model
        tokenizer: Tokenizer
        preprocessor: Image preprocessor
        dataset: Dataset to process
        num_images: Number of images to process
        layers: List of layer indices
        device: CUDA device
        max_new_tokens: Max tokens to generate per patch
        sample_center: Fraction of center patches (legacy mode)
        num_patches: Number of random patches per image (new mode, overrides sample_center)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping layer_idx -> list of image results (in logitlens-compatible format)
    """
    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)

    all_results = {l: [] for l in layers}

    for img_idx in range(num_images):
        sample = dataset[img_idx]
        # Get caption from message_list
        caption = sample['message_list'][0]['text'] if sample['message_list'] else "No caption"

        print(f"\n  Processing image {img_idx + 1}/{num_images}...")
        print(f"    Caption: {caption[:60]}...")

        # Preprocess
        inputs = preprocessor({"image": sample["image"], "message": ""})
        image_input_idx = inputs.get("image_input_idx")

        if image_input_idx is None:
            print(f"    Skipping - no image tokens")
            continue

        # Move to device
        batch = {
            "input_ids": torch.tensor(inputs["input_tokens"], device=device).unsqueeze(0),
            "images": torch.tensor(inputs["images"], device=device).unsqueeze(0).half(),
            "image_masks": torch.tensor(inputs["image_masks"], device=device).unsqueeze(0),
            "image_input_idx": torch.tensor(inputs["image_input_idx"], device=device).unsqueeze(0),
        }

        # Get hidden states at all layers
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                output = model(
                    input_ids=batch["input_ids"],
                    images=batch["images"],
                    image_masks=batch["image_masks"],
                    image_input_idx=batch["image_input_idx"],
                    output_hidden_states=True,
                )

        # Find visual token positions
        image_input_idx_np = inputs["image_input_idx"]
        visual_positions = image_input_idx_np[image_input_idx_np >= 0]
        num_visual_tokens = len(visual_positions)

        grid_size = int(math.sqrt(num_visual_tokens))

        # Determine which patches to process
        if num_patches is not None:
            # New mode: random patches with consistent seed per image
            patch_indices = get_random_patch_indices(
                num_visual_tokens, num_patches, seed=seed + img_idx
            )
        else:
            # Legacy mode: center patches
            patch_indices = get_center_patch_indices(grid_size, sample_center)

        num_to_process = len(patch_indices) if patch_indices else num_visual_tokens
        print(f"    Visual tokens: {num_visual_tokens} (processing {num_to_process})")

        # For each layer, extract visual hidden states and generate descriptions
        for layer_idx in layers:
            if layer_idx >= num_layers:
                continue

            print(f"    Layer {layer_idx}...", end=" ", flush=True)

            # Extract visual token hidden states
            hidden_states = output.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
            visual_hidden_states = hidden_states[visual_positions]  # [num_visual, hidden_dim]

            # Run descriptive Patchscopes
            patch_results = run_descriptive_patchscopes(
                model, tokenizer, visual_hidden_states,
                layer_idx, device, max_new_tokens, patch_indices
            )

            # Format in logitlens-compatible structure:
            # Each image has chunks -> patches structure
            all_results[layer_idx].append({
                "image_idx": img_idx,
                "ground_truth_caption": caption,
                "chunks": [{
                    "chunk_name": "Full Image",
                    "patches": [{
                        "patch_idx": p["patch_idx"],
                        "patch_row": p["patch_row"],
                        "patch_col": p["patch_col"],
                        "description": p["generated_text"],  # Single description string
                    } for p in patch_results]
                }]
            })

            print(f"done")

        # Clean up
        del output, batch
        torch.cuda.empty_cache()

    return all_results


def create_html_viewer(results, output_dir, checkpoint_name, dataset):
    """Create an interactive HTML viewer for the results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = sorted(results.keys())

    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Patchscopes Descriptive - Visual Token Descriptions</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background: #f0f2f5;
        }
        h1 { color: #1a1a2e; margin-bottom: 5px; }
        .subtitle { color: #666; margin-bottom: 20px; }
        .controls {
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .controls label { margin-right: 10px; font-weight: bold; }
        .controls select {
            padding: 8px 15px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ddd;
            margin-right: 20px;
        }
        .image-container {
            display: flex;
            gap: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .image-section { flex: 0 0 400px; }
        .image-section img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .caption {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 13px;
            color: #444;
        }
        .grid-section { flex: 1; }
        .patch-grid {
            display: grid;
            gap: 3px;
            background: #ddd;
            padding: 3px;
            border-radius: 5px;
        }
        .patch-cell {
            background: white;
            padding: 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.2s;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .patch-cell:hover {
            background: #e3f2fd;
            transform: scale(1.05);
            z-index: 10;
            white-space: normal;
            position: relative;
        }
        .patch-cell.selected {
            background: #bbdefb;
            border: 2px solid #2196f3;
        }
        .detail-panel {
            margin-top: 20px;
            padding: 20px;
            background: #e8f4fd;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }
        .detail-panel h3 { margin-top: 0; color: #1565c0; }
        .generated-text {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .method-info {
            background: #fff3e0;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #ff9800;
        }
        .layer-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }
        .layer-tab {
            padding: 8px 16px;
            background: #e0e0e0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .layer-tab.active {
            background: #2196f3;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Patchscopes Descriptive - Visual Token Descriptions</h1>
    <p class="subtitle">""" + checkpoint_name + """</p>

    <div class="method-info">
        <strong>Method:</strong> Few-shot descriptive prompt with patched vision tokens<br>
        <strong>Prompt:</strong> "Dog is a type of animal... Mathematics is a field... X is" → patch visual token at X<br>
        <strong>Goal:</strong> See if the model can describe what visual content the patch represents
    </div>

    <div class="controls">
        <label>Select Image:</label>
        <select id="imageSelect" onchange="showImage(this.value)">
"""]

    # Add image options
    if layers and results[layers[0]]:
        for i, img_data in enumerate(results[layers[0]]):
            caption_preview = img_data["ground_truth_caption"][:50] + "..."
            html_parts.append(f'            <option value="{i}">Image {i}: {caption_preview}</option>\n')

    html_parts.append("""        </select>
    </div>

    <div id="content"></div>

    <script>
    const results = """)

    # Embed results as JSON
    html_parts.append(json.dumps(results, indent=2))

    html_parts.append(""";

    const layers = """ + json.dumps(layers) + """;
    let currentLayer = layers[0];
    let currentImage = 0;
    let selectedPatch = null;

    function showImage(imgIdx) {
        currentImage = parseInt(imgIdx);
        renderContent();
    }

    function selectLayer(layer) {
        currentLayer = layer;
        renderContent();
    }

    function selectPatch(patchIdx) {
        selectedPatch = patchIdx;
        renderContent();
    }

    function renderContent() {
        const container = document.getElementById('content');
        const imgData = results[currentLayer][currentImage];

        if (!imgData) {
            container.innerHTML = '<p>No data for this selection</p>';
            return;
        }

        const gridSize = imgData.grid_size;

        let html = `
            <div class="layer-tabs">
                ${layers.map(l => `
                    <button class="layer-tab ${l === currentLayer ? 'active' : ''}"
                            onclick="selectLayer(${l})">Layer ${l}</button>
                `).join('')}
            </div>

            <div class="image-container">
                <div class="image-section">
                    <img src="img_${currentImage}.jpg" alt="Image ${currentImage}">
                    <div class="caption"><strong>Caption:</strong> ${imgData.ground_truth_caption}</div>
                </div>
                <div class="grid-section">
                    <h3>Generated Descriptions (${gridSize}x${gridSize} grid)</h3>
                    <div class="patch-grid" style="grid-template-columns: repeat(${gridSize}, 1fr);">
        `;

        for (const patch of imgData.patches) {
            const isSelected = selectedPatch === patch.patch_idx;
            const text = patch.generated_text || '(empty)';
            const preview = text.substring(0, 30) + (text.length > 30 ? '...' : '');
            html += `
                <div class="patch-cell ${isSelected ? 'selected' : ''}"
                     onclick="selectPatch(${patch.patch_idx})"
                     title="${text}">
                    ${preview}
                </div>
            `;
        }

        html += `
                    </div>
                </div>
            </div>
        `;

        // Detail panel for selected patch
        if (selectedPatch !== null) {
            const patch = imgData.patches.find(p => p.patch_idx === selectedPatch);
            if (patch) {
                html += `
                    <div class="detail-panel">
                        <h3>Patch [${patch.patch_row}, ${patch.patch_col}] - Layer ${currentLayer}</h3>
                        <p><strong>Position:</strong> Row ${patch.patch_row}, Column ${patch.patch_col}</p>
                        <div class="generated-text">
                            <strong>"X is</strong> ${patch.generated_text}<strong>"</strong>
                        </div>
                        <p style="margin-top: 10px; color: #666; font-size: 12px;">
                            Tokens: ${patch.generated_tokens.join(' | ')}
                        </p>
                    </div>
                `;
            }
        }

        container.innerHTML = html;
    }

    // Initial render
    renderContent();
    </script>
</body>
</html>
""")

    # Save HTML
    html_path = output_dir / "descriptive_patchscopes.html"
    with open(html_path, "w") as f:
        f.write(''.join(html_parts))

    # Save images
    from PIL import Image
    for layer_idx in layers:
        if results[layer_idx]:
            for img_data in results[layer_idx]:
                img_idx = img_data["image_idx"]
                img_path = output_dir / f"img_{img_idx}.jpg"
                if not img_path.exists():
                    sample = dataset[img_idx]
                    # sample["image"] is a path string, need to load it
                    img = Image.open(sample["image"])
                    img.save(img_path, "JPEG", quality=85)
            break  # Only need to save images once

    print(f"\n✓ HTML viewer saved to: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Patchscopes with descriptive few-shot prompt")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--layers", type=str, default="0,2,4,8,16",
                        help="Comma-separated layer indices (default: 0,2,4,8,16)")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max tokens to generate per patch")
    parser.add_argument("--output-dir", type=str,
                        default="analysis_results/patchscopes")
    parser.add_argument("--num-patches", type=int, default=10,
                        help="Number of random patches per image (default: 10)")
    parser.add_argument("--sample-center", type=float, default=None,
                        help="Legacy: fraction of center patches to sample (overridden by --num-patches)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for patch selection")
    parser.add_argument("--lite-suffix", type=str, default="",
                        help="Suffix for output dir (e.g., '_lite10' for unified viewer)")
    parser.add_argument("--skip-html", action="store_true",
                        help="Skip HTML viewer generation (for batch runs)")
    args = parser.parse_args()

    device = torch.device("cuda")
    layers = [int(l.strip()) for l in args.layers.split(",")]

    # Determine sampling mode
    if args.sample_center is not None:
        sampling_mode = f"center {args.sample_center:.0%}"
        num_patches = None
        sample_center = args.sample_center
    else:
        sampling_mode = f"{args.num_patches} random patches"
        num_patches = args.num_patches
        sample_center = 1.0

    print("=" * 70)
    print("PATCHSCOPES DESCRIPTIVE - Visual Token Descriptions")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Images: {args.num_images}")
    print(f"Layers: {layers}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Sampling: {sampling_mode}")
    print(f"Seed: {args.seed}")
    print()
    print("Prompt template:")
    print("-" * 50)
    print(DESCRIPTIVE_PROMPT)
    print("-" * 50)
    print()

    # Load model
    print("Loading model...")
    cfg = TrainConfig.load(f"{args.ckpt_path}/config.yaml")
    cfg.model.init_device = "cpu"
    model = Molmo(cfg.model)

    ckpt_file = f"{args.ckpt_path}/model.pt"
    if os.path.getsize(ckpt_file) / (1024**3) < 1.0:
        model.reset_with_pretrained_weights()

    weights = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    model = model.half().cuda().eval()
    torch.cuda.empty_cache()
    print("✓ Model loaded")

    # Load tokenizer and preprocessor
    model_config = ModelConfig.load(
        resource_path(args.ckpt_path, "config.yaml"),
        key="model", validate_paths=False
    )
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=True, shuffle_messages=False,
        is_training=False, require_image_features=True
    )
    tokenizer = preprocessor.tokenizer

    num_layers = len(get_transformer_blocks(model))
    print(f"Model has {num_layers} layers")

    # Validate layers
    layers = [l for l in layers if l < num_layers]
    print(f"Testing layers: {layers}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = PixMoCap(split="validation", mode="captions")
    print(f"Dataset size: {len(dataset)}")

    # Process images
    print("\n" + "=" * 70)
    print("PROCESSING IMAGES")
    print("=" * 70)

    start_time = time.time()
    results = process_images(
        model, tokenizer, preprocessor, dataset,
        args.num_images, layers, device, args.max_new_tokens,
        sample_center=sample_center, num_patches=num_patches, seed=args.seed
    )
    elapsed = time.time() - start_time

    print(f"\n✓ Processed {args.num_images} images in {elapsed:.1f}s")

    # Build checkpoint name for output directory (match logitlens/NN pattern)
    # From: molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336/step12000-unsharded
    # To: train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded
    ckpt_parts = args.ckpt_path.rstrip('/').split('/')
    checkpoint_name = ckpt_parts[-2] + "_" + ckpt_parts[-1]  # e.g., train_..._step12000-unsharded

    # Add lite suffix if specified
    output_dir = Path(args.output_dir) / f"{checkpoint_name}{args.lite_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results in logitlens-compatible format
    for layer_idx, layer_results in results.items():
        json_path = output_dir / f"patchscopes_layer{layer_idx}.json"
        with open(json_path, "w") as f:
            json.dump({
                "checkpoint": args.ckpt_path,
                "split": "validation",
                "num_images": args.num_images,
                "layer_idx": layer_idx,
                "method": "patchscopes_descriptive",
                "prompt": DESCRIPTIVE_PROMPT,
                "max_new_tokens": args.max_new_tokens,
                "num_patches_per_image": num_patches if num_patches else "center",
                "results": layer_results,
            }, f, indent=2)
        print(f"  ✓ Saved: {json_path}")

    # Create HTML viewer (unless skipped)
    if not args.skip_html:
        print("\nCreating HTML viewer...")
        create_html_viewer(results, output_dir, checkpoint_name, dataset)

    # Print sample results
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS (First image, first 5 patches)")
    print("=" * 70)

    for layer_idx in layers[:3]:  # Show first 3 layers
        if results[layer_idx]:
            img_data = results[layer_idx][0]
            print(f"\nLayer {layer_idx}:")
            # Access via new structure: chunks[0]["patches"]
            patches = img_data["chunks"][0]["patches"]
            for patch in patches[:5]:
                desc = patch.get("description", "")[:60]
                print(f"  [{patch['patch_row']},{patch['patch_col']}]: \"{desc}...\"")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

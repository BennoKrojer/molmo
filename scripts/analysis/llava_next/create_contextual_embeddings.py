#!/usr/bin/env python3
"""
Create contextual embeddings from LLaVA-NeXT-34B's finetuned Vicuna LLM backbone (text-only).

LLaVA-1.5 uses a finetuned Vicuna-7B (LLaMA2-based) as its LLM backbone. We extract
contextual embeddings from this finetuned LLM (not vanilla LLaMA/Vicuna) for LatentLens
analysis, following the same approach as Qwen2-VL.

Usage:
    # Test with small number of captions
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/llava_next/create_contextual_embeddings.py \
        --num-captions 1000 --layers 1 2 4 8 16 24 30 31

    # Full extraction with 8 GPUs in parallel:
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python scripts/analysis/llava_next/create_contextual_embeddings.py \
            --dataset vg --num-captions -1 --shard $i --num-shards 8 --embedding-dtype float8 &
    done

    # Merge shards and build caches:
    python scripts/analysis/llava_next/create_contextual_embeddings.py --merge-shards --num-shards 8 --dataset vg
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent to path for shared module
sys.path.insert(0, str(Path(__file__).parent.parent))
from contextual_embeddings_common import add_common_args, run_extraction

MODEL_NAME = "llava-hf/llava-v1.6-34b-hf"
# LLaVA-1.5 uses Vicuna (LLaMA2) backbone: 32 layers
DEFAULT_LAYERS = [1, 2, 4, 8, 16, 24, 30, 31]


def load_llava_model():
    """
    Load LLaVA-NeXT-34B and return a text-only forward function.

    LLaVA's architecture: LlavaNextForConditionalGeneration contains:
      - model.vision_tower (CLIP ViT)
      - model.language_model (LlamaForCausalLM / Vicuna)
      - model.multi_modal_projector (MLP)

    For contextual embeddings, we only need the language_model (text-only).

    Returns:
        (model_forward_fn, tokenizer, num_layers)
    """
    from transformers import LlavaNextForConditionalGeneration, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.text_config.num_hidden_layers
    hidden_size = model.config.text_config.hidden_size
    print(f"Loaded LLaVA-NeXT-34B: {num_layers} layers, hidden_size={hidden_size}")

    # Get a reference to the language model
    language_model = model.language_model

    def model_forward_fn(input_ids, attention_mask):
        """
        Text-only forward pass through LLaVA's Vicuna backbone.

        We call language_model directly (LlamaForCausalLM), bypassing
        the vision tower and projector entirely.
        """
        outputs = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states

    return model_forward_fn, tokenizer, num_layers


def main():
    parser = argparse.ArgumentParser(
        description=f"Create contextual embeddings from {MODEL_NAME} LLM backbone"
    )
    parser.add_argument("--layers", type=int, nargs='+', default=DEFAULT_LAYERS,
                       help=f"Layers to extract (default: {DEFAULT_LAYERS})")
    add_common_args(parser)
    args = parser.parse_args()

    run_extraction(
        args=args,
        model_name=MODEL_NAME,
        default_layers=DEFAULT_LAYERS,
        load_model_fn=load_llava_model,
        model_type_description="LLaVA-NeXT-34B (finetuned Vicuna/LLaMA2 LLM backbone, text-only)",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create contextual embeddings from Molmo-72B's finetuned Qwen2 LLM backbone (text-only).

Molmo-72B uses a finetuned Qwen2-7B as its LLM backbone. We extract contextual
embeddings from this finetuned LLM (not vanilla Qwen2) for LatentLens analysis,
following the same approach as Qwen2-VL.

Usage:
    # Test with small number of captions
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/molmo_72b/create_contextual_embeddings.py \
        --num-captions 1000 --layers 1 2 4 8 16 24 26 27

    # Full extraction with 8 GPUs in parallel:
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python scripts/analysis/molmo_72b/create_contextual_embeddings.py \
            --dataset vg --num-captions -1 --shard $i --num-shards 8 --embedding-dtype float8 &
    done

    # Merge shards and build caches:
    python scripts/analysis/molmo_72b/create_contextual_embeddings.py --merge-shards --num-shards 8 --dataset vg
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent to path for shared module
sys.path.insert(0, str(Path(__file__).parent.parent))
from contextual_embeddings_common import add_common_args, run_extraction

MODEL_NAME = "allenai/Molmo-72B-0924"
# Molmo uses Qwen2 backbone: 28 layers
DEFAULT_LAYERS = [1, 2, 4, 8, 16, 24, 26, 27]


def load_molmo_model():
    """
    Load Molmo-72B and return a text-only forward function.

    Molmo's architecture: MolmoForCausalLM wraps a Molmo model which contains
    the transformer (Qwen2 backbone). When called without images, it processes
    text-only through the LLM.

    Returns:
        (model_forward_fn, tokenizer, num_layers)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Molmo uses a custom tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"Loaded Molmo-72B: {num_layers} layers, hidden_size={model.config.hidden_size}")

    def model_forward_fn(input_ids, attention_mask):
        """
        Text-only forward pass through Molmo's LLM backbone.

        Molmo's model.model is the Molmo class. Its forward() accepts
        input_ids without images (images=None by default), doing text-only processing.
        """
        outputs = model.model.forward(
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
        load_model_fn=load_molmo_model,
        model_type_description="Molmo-72B (finetuned Qwen2 LLM backbone, text-only)",
    )


if __name__ == "__main__":
    main()

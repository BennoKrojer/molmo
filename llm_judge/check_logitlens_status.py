#!/usr/bin/env python3
"""
Quick status check for LLM judge logitlens runs.
Shows which runs are complete, incomplete, or missing.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def get_checkpoint_name(llm, encoder):
    """Get checkpoint name for a model combination."""
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        return f"train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_seed10"
    return f"train_mlp-only_pixmo_cap_resize_{llm}_{encoder}"


def find_available_layers(base_dir, llms, encoders):
    """Find all available input logit lens JSON files."""
    available = defaultdict(dict)
    
    base_path = Path(base_dir)
    for llm in llms:
        for encoder in encoders:
            checkpoint_name = get_checkpoint_name(llm, encoder)
            logitlens_dir = base_path / f"{checkpoint_name}_step12000-unsharded"
            
            if not logitlens_dir.exists():
                continue
            
            layer_files = sorted(logitlens_dir.glob("logit_lens_layer*_topk5_multi-gpu.json"))
            for layer_file in layer_files:
                layer_num = layer_file.stem.replace("logit_lens_layer", "").replace("_topk5_multi-gpu", "")
                available[(llm, encoder)][layer_num] = layer_file
    
    return available


def check_output(output_base, llm, encoder, layer, expected_count, model_suffix="gpt5"):
    """Check if output exists and is complete."""
    base_path = Path(output_base)
    
    # Try different naming patterns
    if llm == "qwen2-7b" and encoder == "vit-l-14-336":
        model_name = f"{llm}_{encoder}_seed10"
    else:
        model_name = f"{llm}_{encoder}"
    
    # Pattern: llm_judge_{model}_layer{num}_{suffix}_cropped
    possible_dirs = [
        base_path / f"llm_judge_{model_name}_layer{layer}_{model_suffix}_cropped",
        base_path / f"llm_judge_{model_name}_layer{layer}_cropped",
    ]
    
    for output_dir in possible_dirs:
        results_file = output_dir / "results_validation.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            total = data.get('total', 0)
            if total >= expected_count:
                return "complete", total
            else:
                return "incomplete", total
    
    return "missing", 0


def main():
    parser = argparse.ArgumentParser(description='Check status of LLM judge logitlens runs')
    parser.add_argument('--base-dir', type=str, default='analysis_results/logit_lens')
    parser.add_argument('--output-base', type=str, default='analysis_results/llm_judge_logitlens')
    parser.add_argument('--expected-count', type=int, default=100, help='Expected number of examples per run')
    parser.add_argument('--llms', nargs='+', default=['olmo-7b', 'qwen2-7b', 'llama3-8b'])
    parser.add_argument('--encoders', nargs='+', default=['vit-l-14-336', 'dinov2-large-336', 'siglip'])
    parser.add_argument('--show-complete', action='store_true', help='Also show complete runs')
    
    args = parser.parse_args()
    
    print("Scanning available layers...")
    available = find_available_layers(args.base_dir, args.llms, args.encoders)
    
    complete = []
    incomplete = []
    missing = []
    
    for (llm, encoder), layers in sorted(available.items()):
        for layer_num in sorted(layers.keys(), key=int):
            status, count = check_output(args.output_base, llm, encoder, layer_num, args.expected_count)
            
            if status == "complete":
                complete.append((llm, encoder, layer_num, count))
            elif status == "incomplete":
                incomplete.append((llm, encoder, layer_num, count, args.expected_count))
            else:
                missing.append((llm, encoder, layer_num))
    
    print("\n" + "=" * 60)
    print("STATUS SUMMARY")
    print("=" * 60)
    print(f"Complete runs:   {len(complete)}")
    print(f"Incomplete runs: {len(incomplete)}")
    print(f"Missing runs:    {len(missing)}")
    print(f"Total runs:      {len(complete) + len(incomplete) + len(missing)}")
    print("=" * 60)
    
    if missing:
        print(f"\nMISSING RUNS ({len(missing)}):")
        for llm, encoder, layer in missing:
            print(f"  {llm:15} + {encoder:20} layer{layer}")
    
    if incomplete:
        print(f"\nINCOMPLETE RUNS ({len(incomplete)}):")
        for llm, encoder, layer, actual, expected in incomplete:
            print(f"  {llm:15} + {encoder:20} layer{layer:3} : {actual:3}/{expected} examples")
    
    if args.show_complete and complete:
        print(f"\nCOMPLETE RUNS ({len(complete)}):")
        for llm, encoder, layer, count in complete:
            print(f"  {llm:15} + {encoder:20} layer{layer:3} : {count} examples")
    
    if missing or incomplete:
        print(f"\nTo regenerate missing/incomplete runs:")
        print(f"  1. Set SKIP_IF_COMPLETE=true in run_all_parallel_logitlens.sh (already set)")
        print(f"  2. Run: bash llm_judge/run_all_parallel_logitlens.sh")
        print(f"\nOr to force regenerate everything:")
        print(f"  Set SKIP_IF_COMPLETE=false in run_all_parallel_logitlens.sh")


if __name__ == "__main__":
    main()


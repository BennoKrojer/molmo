#!/usr/bin/env python3
"""Compare pass@5 (paper) vs pass@1 (new) LLM judge results.

Loads existing pass@5 numbers from paper_plots/data.json and
pass@1 results from analysis_results/llm_judge_topk1/.

Usage:
    python scripts/analysis/corpus_ablation/compare_topk.py
"""
import json
from pathlib import Path


def load_topk1_results():
    """Load pass@1 evaluation results."""
    base = Path("analysis_results/llm_judge_topk1")
    results = {}

    # Map directory names to data.json-style keys
    method_map = {
        "embeddinglens": "nn",
        "logitlens": "logitlens",
        "latentlens": "contextual",
    }
    model_map = {
        "olmo-vit": "olmo-7b+vit-l-14-336",
        "llama-siglip": "llama3-8b+siglip",
        "qwen-dino": "qwen2-7b+dinov2-large-336",
    }

    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        method_key = method_map.get(method_dir.name)
        if not method_key:
            continue

        for model_dir in sorted(method_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_key = model_map.get(model_dir.name)
            if not model_key:
                continue

            rf = model_dir / "evaluation_results.json"
            if not rf.exists():
                continue

            data = json.load(open(rf))
            layer_results = {}
            for r in data:
                total = r.get("total_count", 0)
                if total > 0:
                    pct = r["interpretable_count"] / total * 100
                    layer_results[str(r["layer"])] = round(pct, 2)

            if method_key not in results:
                results[method_key] = {}
            results[method_key][model_key] = layer_results

    return results


def main():
    # Load pass@5 from paper
    with open("paper_plots/data.json") as f:
        paper_data = json.load(f)

    # Load pass@1
    topk1 = load_topk1_results()

    models = {
        "olmo-vit": "olmo-7b+vit-l-14-336",
        "llama-siglip": "llama3-8b+siglip",
        "qwen-dino": "qwen2-7b+dinov2-large-336",
    }
    methods = ["nn", "logitlens", "contextual"]
    method_labels = {
        "nn": "EmbeddingLens",
        "logitlens": "LogitLens",
        "contextual": "LatentLens",
    }

    print("=" * 80)
    print("Pass@5 (paper) vs Pass@1 (new) Comparison")
    print("=" * 80)

    for short_name, model_key in models.items():
        print(f"\n{'─' * 70}")
        print(f"Model: {model_key}")
        print(f"{'─' * 70}")
        print(f"{'Method':<15} {'Pass@5 avg':>10} {'Pass@1 avg':>10} {'Delta':>8}")
        print(f"{'─' * 45}")

        for method in methods:
            label = method_labels[method]

            # Pass@5
            p5 = paper_data.get(method, {}).get(model_key, {})
            p5_avg = sum(p5.values()) / len(p5) if p5 else 0

            # Pass@1
            p1 = topk1.get(method, {}).get(model_key, {})
            p1_avg = sum(p1.values()) / len(p1) if p1 else 0

            delta = p1_avg - p5_avg
            delta_str = f"{delta:+.1f}pp" if p1 else "N/A"
            p1_str = f"{p1_avg:.1f}%" if p1 else "pending"

            print(f"{label:<15} {p5_avg:>9.1f}% {p1_str:>10} {delta_str:>8}")

            # Per-layer detail
            if p1:
                layers = sorted(set(list(p5.keys()) + list(p1.keys())), key=lambda x: int(x))
                details = []
                for l in layers:
                    v5 = p5.get(l)
                    v1 = p1.get(l)
                    if v5 is not None and v1 is not None:
                        details.append(f"L{l}: {v5:.0f}→{v1:.0f}")
                    elif v1 is not None:
                        details.append(f"L{l}: ?→{v1:.0f}")
                if details:
                    print(f"  {'':15} {', '.join(details)}")

    # Summary table for rebuttal
    print(f"\n{'=' * 80}")
    print("Summary for rebuttal (avg across layers):")
    print(f"{'=' * 80}")
    print(f"{'Model':<30} {'Method':<15} {'Pass@5':>8} {'Pass@1':>8} {'Drop':>8}")
    for short_name, model_key in models.items():
        for method in methods:
            label = method_labels[method]
            p5 = paper_data.get(method, {}).get(model_key, {})
            p1 = topk1.get(method, {}).get(model_key, {})
            p5_avg = sum(p5.values()) / len(p5) if p5 else 0
            p1_avg = sum(p1.values()) / len(p1) if p1 else 0
            if p1:
                drop = p5_avg - p1_avg
                print(f"{model_key:<30} {label:<15} {p5_avg:>7.1f}% {p1_avg:>7.1f}% {drop:>+7.1f}pp")
            else:
                print(f"{model_key:<30} {label:<15} {p5_avg:>7.1f}% {'pending':>8}")


if __name__ == "__main__":
    main()

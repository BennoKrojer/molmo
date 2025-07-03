"""
Standalone script to analyze existing mosaic token localization results.
Just loads the JSON and prints summary statistics without re-running experiments.
"""

import json
import numpy as np
from pathlib import Path
import argparse

def print_summary_statistics(results):
    """Print aggregate statistics about the localization experiments."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Collect all experiments
    all_noise_experiments = []
    all_token_replacements = []
    
    for image_result in results["experiments"]:
        for token_exp in image_result["token_experiments"]:
            if token_exp["noise_replacement"]:
                all_noise_experiments.append(token_exp["noise_replacement"])
            
            for replacement in token_exp["token_replacements"]:
                all_token_replacements.append(replacement)
    
    total_experiments = len(all_noise_experiments) + len(all_token_replacements)
    
    print(f"Total Images Processed: {len(results['experiments'])}")
    print(f"Total Token Positions Tested: {len(all_noise_experiments)}")
    print(f"Total Replacement Experiments: {len(all_token_replacements)}")
    print(f"Total Experiments: {total_experiments}")
    
    if not all_noise_experiments:
        print("No experiments found to analyze!")
        return
    
    print("\n" + "-"*60)
    print("NOISE REPLACEMENT ANALYSIS")
    print("-"*60)
    
    # Noise replacement statistics
    noise_changed_responses = sum(1 for exp in all_noise_experiments if not exp["responses_identical"])
    noise_response_change_rate = noise_changed_responses / len(all_noise_experiments) * 100
    
    noise_edit_distances = [exp["num_differences"] for exp in all_noise_experiments]
    avg_noise_edit_distance = np.mean(noise_edit_distances)
    median_noise_edit_distance = np.median(noise_edit_distances)
    
    # Perfect localization: exactly 1 token changed
    perfect_noise_localization = sum(1 for exp in all_noise_experiments if exp["num_differences"] == 1)
    perfect_noise_rate = perfect_noise_localization / len(all_noise_experiments) * 100
    
    # No change at all
    no_change_noise = sum(1 for exp in all_noise_experiments if exp["num_differences"] == 0)
    no_change_noise_rate = no_change_noise / len(all_noise_experiments) * 100
    
    print(f"Response Change Rate: {noise_response_change_rate:.1f}% ({noise_changed_responses}/{len(all_noise_experiments)})")
    print(f"Average Edit Distance: {avg_noise_edit_distance:.2f} tokens")
    print(f"Median Edit Distance: {median_noise_edit_distance:.1f} tokens")
    print(f"Perfect Localization Rate (exactly 1 change): {perfect_noise_rate:.1f}% ({perfect_noise_localization}/{len(all_noise_experiments)})")
    print(f"No Change Rate: {no_change_noise_rate:.1f}% ({no_change_noise}/{len(all_noise_experiments)})")
    
    if all_token_replacements:
        print("\n" + "-"*60)
        print("TOKEN REPLACEMENT ANALYSIS")
        print("-"*60)
        
        # Token replacement statistics
        token_changed_responses = sum(1 for exp in all_token_replacements if not exp["responses_identical"])
        token_response_change_rate = token_changed_responses / len(all_token_replacements) * 100
        
        token_edit_distances = [exp["num_differences"] for exp in all_token_replacements]
        avg_token_edit_distance = np.mean(token_edit_distances)
        median_token_edit_distance = np.median(token_edit_distances)
        
        # Perfect localization: exactly 1 token changed
        perfect_token_localization = sum(1 for exp in all_token_replacements if exp["num_differences"] == 1)
        perfect_token_rate = perfect_token_localization / len(all_token_replacements) * 100
        
        # No change at all
        no_change_token = sum(1 for exp in all_token_replacements if exp["num_differences"] == 0)
        no_change_token_rate = no_change_token / len(all_token_replacements) * 100
        
        # Color prediction accuracy: does replacing token i with token from source predict source's color?
        color_prediction_successes = 0
        color_prediction_attempts = 0
        
        for exp in all_token_replacements:
            source_color = exp.get("source_ground_truth_color_at_position")
            target_color = exp.get("target_ground_truth_color_at_position")
            
            if source_color and target_color and source_color != target_color:
                color_prediction_attempts += 1
                
                # Check if any of the changed tokens match the source color
                for diff in exp["different_positions"]:
                    if diff.get("modified_token") and source_color.lower() in diff["modified_token"].lower():
                        color_prediction_successes += 1
                        break
        
        print(f"Response Change Rate: {token_response_change_rate:.1f}% ({token_changed_responses}/{len(all_token_replacements)})")
        print(f"Average Edit Distance: {avg_token_edit_distance:.2f} tokens")
        print(f"Median Edit Distance: {median_token_edit_distance:.1f} tokens")
        print(f"Perfect Localization Rate (exactly 1 change): {perfect_token_rate:.1f}% ({perfect_token_localization}/{len(all_token_replacements)})")
        print(f"No Change Rate: {no_change_token_rate:.1f}% ({no_change_token}/{len(all_token_replacements)})")
        
        if color_prediction_attempts > 0:
            color_prediction_rate = color_prediction_successes / color_prediction_attempts * 100
            print(f"Color Prediction Success Rate: {color_prediction_rate:.1f}% ({color_prediction_successes}/{color_prediction_attempts})")
        else:
            print("Color Prediction Success Rate: N/A (no valid attempts)")
    
    print("\n" + "-"*60)
    print("COMPARATIVE ANALYSIS")
    print("-"*60)
    
    if all_token_replacements:
        print(f"Noise vs Token Replacement Response Change Rate: {noise_response_change_rate:.1f}% vs {token_response_change_rate:.1f}%")
        print(f"Noise vs Token Replacement Average Edit Distance: {avg_noise_edit_distance:.2f} vs {avg_token_edit_distance:.2f}")
        print(f"Noise vs Token Replacement Perfect Localization: {perfect_noise_rate:.1f}% vs {perfect_token_rate:.1f}%")
    
    # Position-based analysis
    print("\n" + "-"*60)
    print("SPATIAL POSITION ANALYSIS")
    print("-"*60)
    
    position_stats = {}
    for exp in all_noise_experiments:
        pos = exp.get("token_position")
        if pos is not None:
            if pos not in position_stats:
                position_stats[pos] = {"noise_changes": 0, "noise_total": 0, "token_changes": 0, "token_total": 0}
            position_stats[pos]["noise_total"] += 1
            if not exp["responses_identical"]:
                position_stats[pos]["noise_changes"] += 1
    
    for exp in all_token_replacements:
        pos = exp.get("token_position")
        if pos is not None:
            if pos not in position_stats:
                position_stats[pos] = {"noise_changes": 0, "noise_total": 0, "token_changes": 0, "token_total": 0}
            position_stats[pos]["token_total"] += 1
            if not exp["responses_identical"]:
                position_stats[pos]["token_changes"] += 1
    
    if position_stats:
        # Find most/least responsive positions
        position_response_rates = []
        for pos, stats in position_stats.items():
            total_experiments = stats["noise_total"] + stats["token_total"]
            total_changes = stats["noise_changes"] + stats["token_changes"]
            if total_experiments > 0:
                response_rate = total_changes / total_experiments
                position_response_rates.append((pos, response_rate, total_experiments))
        
        if position_response_rates:
            position_response_rates.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Most Responsive Positions (top 3):")
            for i, (pos, rate, total) in enumerate(position_response_rates[:3]):
                print(f"  Position {pos}: {rate*100:.1f}% response rate ({int(rate*total)}/{total} experiments)")
            
            print(f"Least Responsive Positions (bottom 3):")
            for i, (pos, rate, total) in enumerate(position_response_rates[-3:]):
                print(f"  Position {pos}: {rate*100:.1f}% response rate ({int(rate*total)}/{total} experiments)")
    
    print("\n" + "="*80)
    print("INTERPRETATION SUMMARY")
    print("="*80)
    
    if perfect_noise_rate > 50:
        print("✓ STRONG spatial localization detected - most noise replacements affect exactly 1 output token")
    elif perfect_noise_rate > 25:
        print("~ MODERATE spatial localization detected - some noise replacements show localized effects")
    else:
        print("✗ WEAK spatial localization detected - noise replacements cause distributed changes")
    
    if all_token_replacements and color_prediction_attempts > 0:
        if color_prediction_rate > 50:
            print("✓ STRONG color prediction ability - token swaps successfully predict source colors")
        elif color_prediction_rate > 25:
            print("~ MODERATE color prediction ability - some token swaps predict source colors")
        else:
            print("✗ WEAK color prediction ability - token swaps rarely predict source colors")
    
    if no_change_noise_rate > 30:
        print("⚠ HIGH no-change rate suggests some visual tokens may be redundant or less important")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyze existing mosaic token localization results")
    parser.add_argument("--results_file", type=str, help="Path to the JSON results file")
    parser.add_argument("--results_dir", type=str, help="Directory containing results (will find latest JSON)")
    
    args = parser.parse_args()
    
    if args.results_file:
        results_file = Path(args.results_file)
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        # Find the most recent JSON file
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {results_dir}")
            return
        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results file: {results_file}")
    else:
        # Default path
        results_dir = Path("analysis_results/mosaic_token_localization")
        if results_dir.exists():
            json_files = list(results_dir.glob("**/*.json"))
            if json_files:
                results_file = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent results file: {results_file}")
            else:
                print(f"No JSON files found in {results_dir}")
                return
        else:
            print("No results directory found. Please specify --results_file or --results_dir")
            return
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Checkpoint: {results.get('checkpoint', 'Unknown')}")
    print(f"Dataset: {results.get('dataset', 'Unknown')}")
    print(f"Prompt: {results.get('prompt', 'Unknown')}")
    
    print_summary_statistics(results)

if __name__ == "__main__":
    main() 
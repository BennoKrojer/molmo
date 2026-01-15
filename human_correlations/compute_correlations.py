#!/usr/bin/env python3
"""
Compute correlation between human judgements and LLM judge evaluations for layer 0.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def parse_instance_id(instance_id):
    """
    Parse instance ID to extract model components.
    Format: train_mlp-only_pixmo_cap_resize_{llm}_{encoder}_step12000-unsharded_{image_index}_patch{patch_index}
    Special case: qwen2-7b_vit-l-14-336_seed10
    """
    # Handle special case for seed10
    if '_seed10_' in instance_id:
        pattern = r'train_mlp-only_pixmo_cap_resize_(.+?)_(.+?)_seed10_step12000-unsharded_(\d+)_patch(\d+)'
        match = re.match(pattern, instance_id)
        if match:
            llm, encoder, image_idx, patch_idx = match.groups()
            return {
                'llm': llm,
                'encoder': encoder,
                'model_key': f"{llm}_{encoder}_seed10",
                'image_index': int(image_idx),
                'patch_index': int(patch_idx),
                'full_id': instance_id
            }
    else:
        pattern = r'train_mlp-only_pixmo_cap_resize_(.+?)_(.+?)_step12000-unsharded_(\d+)_patch(\d+)'
        match = re.match(pattern, instance_id)
        if match:
            llm, encoder, image_idx, patch_idx = match.groups()
            return {
                'llm': llm,
                'encoder': encoder,
                'model_key': f"{llm}_{encoder}",
                'image_index': int(image_idx),
                'patch_index': int(patch_idx),
                'full_id': instance_id
            }
    return None


def load_human_results(results_dir):
    """Load all human judgement results."""
    human_data = defaultdict(list)  # instance_id -> list of judgements
    
    results_path = Path(results_dir) / "results"
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_path}")
    
    for result_file in results_path.glob("evaluation_*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        user_id = data.get('userId', 'unknown')
        session_id = data.get('sessionId', 'unknown')
        
        for result in data.get('results', []):
            instance_id = result.get('instanceId')
            if not instance_id:
                continue
            
            selected_words = result.get('selectedWords', [])
            none_selected = result.get('noneSelected', False)
            
            # Convert selected words to a structured format
            word_selections = {}
            for word_data in selected_words:
                word = word_data.get('word', '')
                relation = word_data.get('relation', 'unknown')
                word_selections[word] = relation
            
            human_data[instance_id].append({
                'user_id': user_id,
                'session_id': session_id,
                'selected_words': word_selections,
                'none_selected': none_selected,
                'is_interpretable': len(word_selections) > 0 or not none_selected
            })
    
    return human_data


def load_llm_judge_results(base_dir, model_key, layer=0):
    """Load LLM judge results for a specific model and layer."""
    # Map model_key to directory name format
    # Format: llm_judge_{llm}_{encoder}_layer{layer}_gpt5_cropped
    # Special case: qwen2-7b_vit-l-14-336_seed10 -> qwen2-7b_vit-l-14-336_seed10
    
    # Try to find the results file
    results_file = Path(base_dir) / f"llm_judge_{model_key}_layer{layer}_gpt5_cropped" / "results_validation.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to instance-based format
    # Key: (image_index, patch_row, patch_col)
    llm_data = {}
    
    for image_path, patches in data.get('responses', {}).items():
        for patch_data in patches:
            patch_row = patch_data.get('patch_row')
            patch_col = patch_data.get('patch_col')
            image_index = patch_data.get('image_index')
            tokens_used = patch_data.get('tokens_used', [])
            gpt_response = patch_data.get('gpt_response', {})
            
            # Store the LLM judgement
            concrete_words = set(gpt_response.get('concrete_words', []))
            abstract_words = set(gpt_response.get('abstract_words', []))
            global_words = set(gpt_response.get('global_words', []))
            is_interpretable = gpt_response.get('interpretable', False)
            
            # Create a key that we can match
            key = (image_index, patch_row, patch_col)
            
            llm_data[key] = {
                'tokens_used': tokens_used,
                'concrete_words': concrete_words,
                'abstract_words': abstract_words,
                'global_words': global_words,
                'all_selected_words': concrete_words | abstract_words | global_words,
                'is_interpretable': is_interpretable,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'image_index': image_index
            }
    
    return llm_data


def load_data_json(data_json_path):
    """Load data.json to get mapping from instance_id to image_index/patch info."""
    with open(data_json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    for item in data:
        instance_id = item.get('id')
        if not instance_id:
            continue
        
        parsed = parse_instance_id(instance_id)
        if parsed:
            # Store mapping from instance_id to image_index and patch info
            mapping[instance_id] = {
                'image_index': item.get('index'),
                'original_index': item.get('original_index'),
                'patch_index': item.get('patch_index'),
                'patch_row': item.get('patch_row'),
                'patch_col': item.get('patch_col'),
                'model_key': parsed['model_key'],
                'candidates': item.get('candidates', [])
            }
    
    return mapping


def match_instances(human_data, llm_data_by_model, data_mapping):
    """Match human judgements with LLM judgements."""
    matched_data = []
    unmatched = []
    
    # Build mapping from (model_key, patch_row, patch_col) -> list of (instance_id, candidates)
    # Since multiple instances might have same patch position, we'll match by position first
    position_mapping = defaultdict(list)
    for instance_id, mapping_info in data_mapping.items():
        if instance_id not in human_data:
            continue
        model_key = mapping_info['model_key']
        patch_row = mapping_info.get('patch_row')
        patch_col = mapping_info.get('patch_col')
        
        if patch_row is not None and patch_col is not None:
            key = (model_key, patch_row, patch_col)
            position_mapping[key].append((instance_id, mapping_info))
    
    # Now match LLM data to human data by position
    for model_key, llm_data in llm_data_by_model.items():
        for llm_key, llm_judgement in llm_data.items():
            image_index, patch_row, patch_col = llm_key
            
            if patch_row is None or patch_col is None:
                unmatched.append(f"{model_key}_{image_index}_{patch_row}_{patch_col}")
                continue
            
            # Try to match by model_key and patch position
            match_key = (model_key, patch_row, patch_col)
            
            if match_key in position_mapping:
                # Found potential matches - try to find the best one
                # If there's only one, use it
                candidates_list = position_mapping[match_key]
                
                if len(candidates_list) == 1:
                    instance_id, mapping_info = candidates_list[0]
                    matched_data.append({
                        'instance_id': instance_id,
                        'model_key': model_key,
                        'human_judgements': human_data[instance_id],
                        'llm_judgement': llm_judgement,
                        'candidates': mapping_info.get('candidates', [])
                    })
                else:
                    # Multiple candidates - try to match by token similarity
                    # For now, just take the first one
                    instance_id, mapping_info = candidates_list[0]
                    matched_data.append({
                        'instance_id': instance_id,
                        'model_key': model_key,
                        'human_judgements': human_data[instance_id],
                        'llm_judgement': llm_judgement,
                        'candidates': mapping_info.get('candidates', [])
                    })
            else:
                # Try fuzzy matching - check if patch positions are close (within 1)
                found = False
                for (m_key, p_row, p_col), candidates_list in position_mapping.items():
                    if m_key == model_key:
                        if (p_row is not None and p_col is not None and
                            abs(p_row - patch_row) <= 1 and abs(p_col - patch_col) <= 1):
                            # Found a close match
                            instance_id, mapping_info = candidates_list[0]
                            matched_data.append({
                                'instance_id': instance_id,
                                'model_key': model_key,
                                'human_judgements': human_data[instance_id],
                                'llm_judgement': llm_judgement,
                                'candidates': mapping_info.get('candidates', [])
                            })
                            found = True
                            break
                
                if not found:
                    unmatched.append(f"{model_key}_{image_index}_{patch_row}_{patch_col}")
    
    if unmatched:
        print(f"WARNING: {len(unmatched)} LLM instances could not be matched to human data")
    
    return matched_data


def compute_interpretability_correlation(matched_data):
    """
    Compute binary interpretability correlation metrics.
    A patch is interpretable if at least 1 of the 5 nearest neighbor tokens was marked as interpretable.
    """
    # Collect binary labels: 1 = interpretable, 0 = not interpretable
    llm_labels = []
    human_labels = []
    
    per_model_data = defaultdict(lambda: {'llm': [], 'human': []})
    
    for item in matched_data:
        instance_id = item['instance_id']
        model_key = item['model_key']
        human_judgements = item['human_judgements']
        llm_judgement = item['llm_judgement']
        # Handle both NN format (strings) and contextual format ([phrase, token] pairs)
        candidates = set(c[0] if isinstance(c, list) else c for c in item['candidates'])
        
        # LLM interpretability: binary (1 if interpretable, 0 if not)
        llm_interpretable = 1 if llm_judgement['is_interpretable'] else 0
        
        # Human interpretability: at least 1 of the 5 candidates was selected
        # Count it as interpretable if at least 1 human selected at least 1 candidate
        human_interpretable = 0
        for human_judgement in human_judgements:
            selected_words = set(human_judgement['selected_words'].keys())
            # Check if any of the selected words are in the candidates
            if selected_words & candidates:
                human_interpretable = 1
                break
        
        llm_labels.append(llm_interpretable)
        human_labels.append(human_interpretable)
        per_model_data[model_key]['llm'].append(llm_interpretable)
        per_model_data[model_key]['human'].append(human_interpretable)
    
    # Convert to numpy arrays for correlation computation
    llm_labels = np.array(llm_labels)
    human_labels = np.array(human_labels)
    
    # Debug: print label distribution
    print(f"\nLabel Distribution:")
    print(f"  Human interpretable: {np.sum(human_labels)}/{len(human_labels)} ({100*np.mean(human_labels):.1f}%)")
    print(f"  LLM interpretable: {np.sum(llm_labels)}/{len(llm_labels)} ({100*np.mean(llm_labels):.1f}%)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(human_labels, llm_labels)
    print(f"\nConfusion Matrix (Human vs LLM):")
    print(f"                LLM=0  LLM=1")
    print(f"  Human=0 (NI)   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  Human=1 (I)   {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Compute correlation metrics
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(llm_labels, human_labels)
    
    # Spearman correlation
    spearman_r, spearman_p = spearmanr(llm_labels, human_labels)
    
    # Cohen's kappa
    kappa = cohen_kappa_score(human_labels, llm_labels)
    
    # Accuracy (simple agreement)
    accuracy = np.mean(llm_labels == human_labels)
    
    # Per-model metrics
    per_model_metrics = {}
    for model_key, data in per_model_data.items():
        model_llm = np.array(data['llm'])
        model_human = np.array(data['human'])
        
        if len(model_llm) > 1:
            model_pearson_r, model_pearson_p = pearsonr(model_llm, model_human)
            model_spearman_r, model_spearman_p = spearmanr(model_llm, model_human)
        else:
            model_pearson_r, model_pearson_p = 0.0, 1.0
            model_spearman_r, model_spearman_p = 0.0, 1.0
        
        model_kappa = cohen_kappa_score(model_human, model_llm)
        model_accuracy = np.mean(model_llm == model_human)
        
        per_model_metrics[model_key] = {
            'n': len(model_llm),
            'pearson_r': float(model_pearson_r),
            'pearson_p': float(model_pearson_p),
            'spearman_r': float(model_spearman_r),
            'spearman_p': float(model_spearman_p),
            'cohens_kappa': float(model_kappa),
            'accuracy': float(model_accuracy)
        }
    
    return {
        'n': len(llm_labels),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'cohens_kappa': float(kappa),
        'accuracy': float(accuracy),
        'per_model': per_model_metrics
    }


def compute_inter_annotator_agreement(human_data):
    """Compute inter-annotator agreement for instances with multiple human judgements."""
    iaa_metrics = {}
    
    for instance_id, judgements in human_data.items():
        if len(judgements) < 2:
            continue
        
        # Interpretability agreement
        interpretable_votes = [j['is_interpretable'] for j in judgements]
        interpretable_agreement = len(set(interpretable_votes)) == 1
        
        # Word selection agreement (Jaccard between annotators)
        word_sets = [set(j['selected_words'].keys()) for j in judgements]
        if len(word_sets) >= 2:
            jaccards = []
            for i in range(len(word_sets)):
                for j in range(i + 1, len(word_sets)):
                    if len(word_sets[i]) == 0 and len(word_sets[j]) == 0:
                        jaccard = 1.0
                    elif len(word_sets[i]) == 0 or len(word_sets[j]) == 0:
                        jaccard = 0.0
                    else:
                        intersection = word_sets[i] & word_sets[j]
                        union = word_sets[i] | word_sets[j]
                        jaccard = len(intersection) / len(union) if union else 0.0
                    jaccards.append(jaccard)
            avg_jaccard = np.mean(jaccards) if jaccards else 0.0
        else:
            avg_jaccard = 1.0
        
        iaa_metrics[instance_id] = {
            'num_annotators': len(judgements),
            'interpretability_agreement': interpretable_agreement,
            'word_selection_jaccard': avg_jaccard
        }
    
    return iaa_metrics


def load_human_study_llm_results(results_file):
    """Load LLM judge results from human study evaluation."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to instance-based format: instance_id -> llm_judgement
    llm_data = {}
    for result in data.get('results', []):
        instance_id = result.get('instance_id')
        if not instance_id or 'gpt_response' not in result:
            continue
        
        gpt_response = result['gpt_response']
        
        llm_data[instance_id] = {
            'tokens_used': result.get('candidates', []),
            'concrete_words': set(gpt_response.get('concrete_words', [])),
            'abstract_words': set(gpt_response.get('abstract_words', [])),
            'global_words': set(gpt_response.get('global_words', [])),
            'all_selected_words': set(gpt_response.get('concrete_words', [])) | 
                                 set(gpt_response.get('abstract_words', [])) |
                                 set(gpt_response.get('global_words', [])),
            'is_interpretable': gpt_response.get('interpretable', False),
            'patch_row': result.get('patch_row'),
            'patch_col': result.get('patch_col'),
            'image_url': result.get('image_url')
        }
    
    return llm_data


def match_instances_direct(human_data, llm_data, data_mapping):
    """Match human judgements with LLM judgements using direct instance ID matching."""
    matched_data = []
    unmatched = []
    
    for instance_id, human_judgements in human_data.items():
        if instance_id not in llm_data:
            unmatched.append(instance_id)
            continue
        
        if instance_id not in data_mapping:
            unmatched.append(instance_id)
            continue
        
        mapping_info = data_mapping[instance_id]
        llm_judgement = llm_data[instance_id]
        
        # Extract model key from instance ID
        parsed = parse_instance_id(instance_id)
        if not parsed:
            unmatched.append(instance_id)
            continue
        
        model_key = parsed['model_key']
        
        matched_data.append({
            'instance_id': instance_id,
            'model_key': model_key,
            'human_judgements': human_judgements,
            'llm_judgement': llm_judgement,
            'candidates': mapping_info.get('candidates', [])
        })
    
    if unmatched:
        print(f"WARNING: {len(unmatched)} instances could not be matched")
        print(f"  First few unmatched: {unmatched[:5]}")
    
    return matched_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute human-LLM correlation metrics')
    parser.add_argument('--human-data-dir', type=str, 
                       default='interp_data',
                       help='Directory containing human judgement data')
    parser.add_argument('--llm-results-file', type=str,
                       default=None,
                       help='JSON file with LLM judge results from human study')
    parser.add_argument('--llm-results-dir', type=str,
                       default='../analysis_results/llm_judge_nearest_neighbors',
                       help='Directory containing LLM judge results (old format)')
    parser.add_argument('--output', type=str,
                       default='correlation_results.json',
                       help='Output file for correlation results')
    parser.add_argument('--layer', type=int, default=0,
                       help='Layer to analyze (default: 0, for old format only)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    human_data_dir = Path(args.human_data_dir)
    if not human_data_dir.is_absolute():
        human_data_dir = script_dir / human_data_dir
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    print("Loading human judgement data...")
    human_data = load_human_results(str(human_data_dir))
    print(f"Loaded {len(human_data)} unique instances with human judgements")
    
    print("Loading data.json for instance mapping...")
    data_json_path = human_data_dir / "data.json"
    data_mapping = load_data_json(str(data_json_path))
    print(f"Loaded mapping for {len(data_mapping)} instances")
    
    # Check if we're using new human study LLM results or old validation set results
    if args.llm_results_file:
        print(f"Loading LLM judge results from {args.llm_results_file}...")
        llm_results_file = Path(args.llm_results_file)
        if not llm_results_file.is_absolute():
            llm_results_file = script_dir / llm_results_file
        
        llm_data = load_human_study_llm_results(str(llm_results_file))
        print(f"  Loaded {len(llm_data)} LLM judgements")
        
        print("Matching instances...")
        matched_data = match_instances_direct(human_data, llm_data, data_mapping)
        print(f"Matched {len(matched_data)} instances")
    else:
        print("Loading LLM judge results (old format)...")
        llm_results_dir = Path(args.llm_results_dir)
        if not llm_results_dir.is_absolute():
            llm_results_dir = script_dir / llm_results_dir
        
        # Get all unique model keys from human data
        model_keys = set()
        for instance_id in human_data.keys():
            if instance_id in data_mapping:
                model_keys.add(data_mapping[instance_id]['model_key'])
        
        llm_data_by_model = {}
        for model_key in model_keys:
            llm_data = load_llm_judge_results(str(llm_results_dir), model_key, layer=args.layer)
            if llm_data is not None:
                llm_data_by_model[model_key] = llm_data
                print(f"  Loaded LLM data for {model_key}: {len(llm_data)} instances")
            else:
                print(f"  WARNING: No LLM data found for {model_key}")
        
        print("Matching instances...")
        matched_data = match_instances(human_data, llm_data_by_model, data_mapping)
        print(f"Matched {len(matched_data)} instances")
    
    print("Computing interpretability correlation metrics...")
    correlation_metrics = compute_interpretability_correlation(matched_data)
    
    # Aggregate results
    results = {
        'layer': args.layer,
        'total_matched_instances': len(matched_data),
        'total_human_instances': len(human_data),
        'interpretability_correlation': correlation_metrics
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n=== Summary ===")
    print(f"Total matched instances: {results['total_matched_instances']}")
    print(f"\nOverall Binary Interpretability Correlation (n={correlation_metrics['n']}):")
    print(f"  Pearson r: {correlation_metrics['pearson_r']:.3f} (p={correlation_metrics['pearson_p']:.4f})")
    print(f"  Spearman ρ: {correlation_metrics['spearman_r']:.3f} (p={correlation_metrics['spearman_p']:.4f})")
    print(f"  Cohen's κ: {correlation_metrics['cohens_kappa']:.3f}")
    print(f"  Accuracy: {correlation_metrics['accuracy']:.3f}")
    
    print(f"\nPer-Model Metrics:")
    for model_key, model_results in correlation_metrics['per_model'].items():
        print(f"\n  {model_key} (n={model_results['n']}):")
        print(f"    Pearson r: {model_results['pearson_r']:.3f} (p={model_results['pearson_p']:.4f})")
        print(f"    Spearman ρ: {model_results['spearman_r']:.3f} (p={model_results['spearman_p']:.4f})")
        print(f"    Cohen's κ: {model_results['cohens_kappa']:.3f}")
        print(f"    Accuracy: {model_results['accuracy']:.3f}")


if __name__ == '__main__':
    main()


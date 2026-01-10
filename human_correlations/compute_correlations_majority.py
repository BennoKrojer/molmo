#!/usr/bin/env python3
"""
Compute correlation between human judgements and LLM judge evaluations.
Uses MAJORITY vote criterion: interpretable if >=50% of annotators selected at least one candidate.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def parse_instance_id(instance_id):
    """Parse instance ID to extract model components."""
    if '_seed10_' in instance_id:
        pattern = r'train_mlp-only_pixmo_cap_resize_(.+?)_(.+?)_seed10_step12000-unsharded_(\d+)_patch(\d+)'
        match = re.match(pattern, instance_id)
        if match:
            llm, encoder, image_idx, patch_idx = match.groups()
            return {'model_key': f"{llm}_{encoder}_seed10"}
    else:
        pattern = r'train_mlp-only_pixmo_cap_resize_(.+?)_(.+?)_step12000-unsharded_(\d+)_patch(\d+)'
        match = re.match(pattern, instance_id)
        if match:
            llm, encoder, image_idx, patch_idx = match.groups()
            return {'model_key': f"{llm}_{encoder}"}
    return None


def load_human_results(results_dir):
    """Load all human judgement results."""
    human_data = defaultdict(list)
    results_path = Path(results_dir) / "results"
    
    for result_file in results_path.glob("evaluation_*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
        user_id = data.get('userId', 'unknown')
        
        for result in data.get('results', []):
            instance_id = result.get('instanceId')
            if not instance_id:
                continue
            
            selected_words = {}
            for word_data in result.get('selectedWords', []):
                word = word_data.get('word', '')
                relation = word_data.get('relation', 'unknown')
                selected_words[word] = relation
            
            human_data[instance_id].append({
                'user_id': user_id,
                'selected_words': selected_words
            })
    
    return human_data


def load_llm_results(results_file):
    """Load LLM judge results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    llm_data = {}
    for result in data.get('results', []):
        instance_id = result.get('instance_id')
        if instance_id and 'gpt_response' in result:
            gpt = result['gpt_response']
            llm_data[instance_id] = {
                'is_interpretable': gpt.get('interpretable', False),
                'candidates': result.get('candidates', [])
            }
    
    return llm_data


def load_data_json(data_json_path):
    """Load data.json to get candidates mapping."""
    with open(data_json_path, 'r') as f:
        data = json.load(f)
    return {item['id']: item.get('candidates', []) for item in data}


def compute_correlation(human_data, llm_data, candidates_map):
    """
    Compute binary interpretability correlation metrics.
    
    Human criterion: MAJORITY VOTE - interpretable if >=50% of annotators 
                     selected at least one of the top-5 candidates.
    LLM criterion: Whatever GPT said (is_interpretable field).
    """
    llm_labels = []
    human_labels = []
    per_model_data = defaultdict(lambda: {'llm': [], 'human': []})
    
    matched = 0
    unmatched = 0
    
    for instance_id, judgements in human_data.items():
        if instance_id not in llm_data:
            unmatched += 1
            continue
        
        parsed = parse_instance_id(instance_id)
        if not parsed:
            unmatched += 1
            continue
        
        candidates = set(candidates_map.get(instance_id, []))
        llm_result = llm_data[instance_id]
        model_key = parsed['model_key']
        
        # LLM label
        llm_interp = 1 if llm_result['is_interpretable'] else 0
        
        # Human label: MAJORITY VOTE
        n_annotators = len(judgements)
        n_yes = 0
        for j in judgements:
            selected = set(j['selected_words'].keys())
            if selected & candidates:
                n_yes += 1
        
        # >=50% of annotators must agree
        human_interp = 1 if (n_yes / n_annotators) >= 0.5 else 0
        
        llm_labels.append(llm_interp)
        human_labels.append(human_interp)
        per_model_data[model_key]['llm'].append(llm_interp)
        per_model_data[model_key]['human'].append(human_interp)
        matched += 1
    
    llm_labels = np.array(llm_labels)
    human_labels = np.array(human_labels)
    
    print(f"\nMatched: {matched}, Unmatched: {unmatched}")
    print(f"\nLabel Distribution:")
    print(f"  Human interpretable: {np.sum(human_labels)}/{len(human_labels)} ({100*np.mean(human_labels):.1f}%)")
    print(f"  LLM interpretable: {np.sum(llm_labels)}/{len(llm_labels)} ({100*np.mean(llm_labels):.1f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(human_labels, llm_labels)
    print(f"\nConfusion Matrix (Human vs LLM):")
    print(f"                LLM=0  LLM=1")
    print(f"  Human=0 (NI)   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  Human=1 (I)    {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Metrics
    spearman_r, spearman_p = spearmanr(llm_labels, human_labels)
    kappa = cohen_kappa_score(human_labels, llm_labels)
    accuracy = np.mean(llm_labels == human_labels)
    
    # Per-model
    per_model_metrics = {}
    for model_key, data in per_model_data.items():
        m_llm = np.array(data['llm'])
        m_human = np.array(data['human'])
        if len(m_llm) > 1:
            m_spearman, _ = spearmanr(m_llm, m_human)
        else:
            m_spearman = 0.0
        m_kappa = cohen_kappa_score(m_human, m_llm) if len(set(m_human)) > 1 and len(set(m_llm)) > 1 else 0.0
        m_accuracy = np.mean(m_llm == m_human)
        per_model_metrics[model_key] = {
            'n': len(m_llm),
            'spearman_r': float(m_spearman),
            'cohens_kappa': float(m_kappa),
            'accuracy': float(m_accuracy)
        }
    
    return {
        'n': len(llm_labels),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'cohens_kappa': float(kappa),
        'accuracy': float(accuracy),
        'per_model': per_model_metrics
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-data-dir', default='interp_data_nn')
    parser.add_argument('--llm-results-file', default='llm_judge_results/human_study_llm_results.json')
    parser.add_argument('--output', default='correlation_results_majority.json')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    human_dir = script_dir / args.human_data_dir
    llm_file = script_dir / args.llm_results_file
    output_file = script_dir / args.output
    
    print("Loading human judgements...")
    human_data = load_human_results(str(human_dir))
    print(f"  Loaded {len(human_data)} instances")
    
    print("Loading LLM judge results...")
    llm_data = load_llm_results(str(llm_file))
    print(f"  Loaded {len(llm_data)} instances")
    
    print("Loading candidates mapping...")
    candidates_map = load_data_json(str(human_dir / "data.json"))
    print(f"  Loaded {len(candidates_map)} instances")
    
    print("\n" + "="*60)
    print("Computing correlations (MAJORITY VOTE criterion)...")
    print("="*60)
    
    metrics = compute_correlation(human_data, llm_data, candidates_map)
    
    results = {
        'methodology': 'majority_vote_>=50%',
        'total_human_instances': len(human_data),
        'total_llm_instances': len(llm_data),
        'metrics': metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"RESULTS (n={metrics['n']})")
    print("="*60)
    print(f"  Spearman ρ: {metrics['spearman_r']:.3f} (p={metrics['spearman_p']:.2e})")
    print(f"  Cohen's κ:  {metrics['cohens_kappa']:.3f}")
    print(f"  Accuracy:   {metrics['accuracy']:.1%}")
    
    print(f"\nPer-Model:")
    for model, m in sorted(metrics['per_model'].items()):
        print(f"  {model} (n={m['n']}): ρ={m['spearman_r']:.2f}, κ={m['cohens_kappa']:.2f}, acc={m['accuracy']:.1%}")
    
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()

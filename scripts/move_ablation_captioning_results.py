#!/usr/bin/env python3
"""
Move captioning evaluation results from nearest_neighbors/ablations to captioning_evaluation/ablations
"""
import shutil
from pathlib import Path

ABLATION_CHECKPOINTS = [
    "train_mlp-only_pixmo_cap_first-sentence_olmo-7b_vit-l-14-336",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_linear",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10",
    "train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11",
    "train_mlp-only_pixmo_points_resize_olmo-7b_vit-l-14-336",
]

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    for checkpoint_name in ABLATION_CHECKPOINTS:
        src_dir = project_root / "analysis_results" / "nearest_neighbors" / "ablations" / f"{checkpoint_name}_step12000-unsharded"
        target_dir = project_root / "analysis_results" / "captioning_evaluation" / "ablations" / f"{checkpoint_name}_step12000-unsharded"
        
        llm_judge_file = src_dir / "nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer0_llm_judge_validation.json"
        viz_dir = src_dir / "visualized_llm_caption_judgement"
        
        if llm_judge_file.exists():
            print(f"Processing: {checkpoint_name}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move llm_judge JSON
            target_json = target_dir / llm_judge_file.name
            print(f"  Moving {llm_judge_file} -> {target_json}")
            shutil.move(str(llm_judge_file), str(target_json))
            
            # Move visualizations if they exist
            if viz_dir.exists():
                target_viz = target_dir / viz_dir.name
                print(f"  Moving {viz_dir} -> {target_viz}")
                shutil.move(str(viz_dir), str(target_viz))
            
            print(f"  ✓ Moved to {target_dir}")
        else:
            print(f"Skipping {checkpoint_name} - no captioning evaluation results yet")
    
    print("\nDone!")

if __name__ == "__main__":
    main()


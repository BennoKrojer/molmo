source ../../env/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)

run_eval() {
  JSON_PATH="$1"
  SPLIT="validation"
  # Compute expected output file name
  STEM=$(basename "$JSON_PATH")
  OUT_DIR=$(dirname "$JSON_PATH")
  OUT_FILE="$OUT_DIR/${STEM%.*}_llm_judge_${SPLIT}.json"
  # Only run if output missing or incomplete
  if [ ! -f "$OUT_FILE" ]; then
    echo "Running evaluation for $JSON_PATH (no output found)"
    python3 eval_captioning_gpt-judge.py --results_file "$JSON_PATH" --split "$SPLIT" --fallback-dataset-images --resume --create-visualizations
  else
    # If exists, only build missing visualizations and fill gaps
    echo "Filling gaps for $JSON_PATH (output exists)"
    python3 eval_captioning_gpt-judge.py --results_file "$JSON_PATH" --split "$SPLIT" --fallback-dataset-images --resume --create-visualizations
  fi
}

run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed10_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_seed11_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap_multi-gpu.json
# run_eval analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmoe-1b_7b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json
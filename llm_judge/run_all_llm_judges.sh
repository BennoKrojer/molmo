source ../../env/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
API_KEY=$(cat llm_judge/api_key.txt)
python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5 --save_images
# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_dinov2-large-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5
# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmo-7b_siglip_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5

# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_qwen2-7b_siglip_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5
# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_qwen2-7b_dinov2-large-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5
# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_qwen2-7b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5

python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5 --resume
# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_llama3-8b_dinov2-large-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5
python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_llama3-8b_siglip_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5

# python3 llm_judge/run_llm_judge_pixmo.py --input_json analysis_results/nearest_neighbors/train_mlp-only_pixmo_cap_resize_olmoe-1b_7b_vit-l-14-336_step12000-unsharded/nearest_neighbors_analysis_pixmo_cap.json --api_key $API_KEY --num_images 20 --num_samples 5
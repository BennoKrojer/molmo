import os
import PIL
import json
import numpy as np
from collections import defaultdict
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import random
import base64
import io
from openai import OpenAI
import pickle
import argparse
import sys
from pathlib import Path
from prompts import IMAGE_PROMPT
from utils import (
    calculate_square_bbox_from_patch, 
    get_high_confidence_words,
    clip_bbox_to_image,
    draw_bbox_on_image,
    calculate_expanded_bbox_from_patch,
    process_image_with_mask,
    sample_valid_patch_positions
)

# Import the dataset class to load images
from olmo.data.pixmo_datasets import PixMoCap


def get_gpt_response(client, image, prompt):
    """
    Get GPT-5 response for image interpretability analysis.
    
    Args:
        image (PIL.Image): Image with red bounding box
        prompt (str): Formatted prompt with candidate words
        
    Returns:
        dict: JSON response with interpretability analysis
    """
    
    # Convert PIL image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Format the prompt with candidate words
    try:
        response = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_str}"
                    },
                    {"type": "input_text", "text": prompt},
                    
                ],
            }],
            reasoning={
                "effort": "low"
            },
            text={
                "verbosity": "low"
            }
        )
        
        # Parse the response as JSON
        response_text = response.output_text
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback if no JSON found
                return {
                    "interpretable": False,
                    "words": [],
                    "reasoning": "Could not parse response as JSON"
                }
        except json.JSONDecodeError:
            # Fallback for invalid JSON
            return {
                "interpretable": False,
                "words": [],
                "reasoning": f"JSON parsing error. Raw response: {response_text}"
            }
            
    except Exception as e:
        # Handle API errors
        print(f"Error calling OpenAI API: {e}")
        return {
            "interpretable": False,
            "words": [],
            "reasoning": f"API error: {str(e)}"
        }


def save_incremental_results(results, save_path, image_idx, total_images):
    """Save incremental results after each image."""
    if save_path:
        # Convert defaultdict to regular dict for JSON serialization
        accuracy_percentage = (results['accuracy'] * 100 / results['total']) if results['total'] > 0 else 0
        results_copy = {
            'accuracy': accuracy_percentage,
            'correct': results['accuracy'],
            'total': results['total'],
            'responses': dict(results['responses']),
            'processed_images': results['processed_images'],
            'total_images': results['total_images'],
            'progress_percentage': (results['processed_images'] / results['total_images'] * 100) if results['total_images'] > 0 else 0
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_copy, f, indent=4, default=str)
        
        print(f"  → Saved intermediate results (image {image_idx+1}/{total_images})")


def convert_pixmo_format_to_llm_judge_format(pixmo_data, split_name="train", num_images=None):
    """
    Convert PixMo format to the format expected by run_llm_judge.
    
    Args:
        pixmo_data (dict): Data from the PixMo analysis JSON
        split_name (str): Which split to process ("train" or "validation")
        num_images (int): Number of images to process (None for all)
        
    Returns:
        list: List of image entries in the format expected by run_llm_judge
    """
    # Load the dataset to get image paths
    dataset = PixMoCap(split=split_name, mode="captions")
    
    # Get the split data
    split_data = pixmo_data.get("splits", {}).get(split_name, {})
    images_data = split_data.get("images", [])
    
    if num_images is not None:
        images_data = images_data[:num_images]
    
    converted_data = []
    
    for image_data in images_data:
        image_idx = image_data.get("image_idx", 0)
        
        # Get the actual image path from the dataset
        try:
            example_data = dataset.get(image_idx, np.random)
            image_path = example_data["image"]
        except Exception as e:
            print(f"Warning: Could not load image {image_idx} from dataset: {e}")
            continue
        
        # Convert patches format
        patches = []
        for chunk in image_data.get("chunks", []):
            for patch in chunk.get("patches", []):
                # Convert the patch format
                converted_patch = {
                    "patch_idx": patch.get("patch_idx", 0),
                    "patch_row": patch.get("patch_row", 0),
                    "patch_col": patch.get("patch_col", 0),
                    "nearest_neighbors": patch.get("nearest_neighbors", [])
                }
                patches.append(converted_patch)
        
        # Create the image entry in the expected format
        image_entry = {
            "image_path": image_path,
            "patches": patches,
            "image_idx": image_idx,
            "ground_truth_caption": image_data.get("ground_truth_caption", "")
        }
        
        converted_data.append(image_entry)
    
    return converted_data


def run_llm_judge_pixmo(client, input_json_path, split_name="train", image_indices=None, patch_size=21.33, bbox_size=3, save_images=False, save_results=None, num_samples=36, num_images=None, resume=False):
    """
    Run LLM judge evaluation on PixMo format data.
    
    Args:
        client: OpenAI client instance
        input_json_path (str): Path to PixMo analysis JSON file
        split_name (str): Which split to process ("train" or "validation")
        image_indices (list): List of specific image indices to process (if None, process all images)
        patch_size (float): Size of each patch in pixels (calculated automatically from 512/24)
        bbox_size (int): Size of bounding box in patches (e.g., 3 for 3x3)
        save_images (bool): Whether to save images with bounding boxes
        save_results (str): Path to save results JSON file (optional)
        num_samples (int): Number of random patches to sample per image (default: 36)
        num_images (int): Number of images to process (None for all)
        resume (bool): If True, loads existing results file and resumes (skips completed images)
    
    Returns:
        dict: Results including accuracy, total count, and individual responses
    """
    accuracy = 0
    total = 0
    gpt_responses = defaultdict(list)
    
    # Create images directory if saving images
    images_dir = None
    if save_images and save_results:
        images_dir = Path(save_results).parent / "llm_judge_images"
        images_dir.mkdir(exist_ok=True)
        print(f"Images will be saved to: {images_dir}")
    
    # Load the PixMo JSON file (try multi-gpu version if original doesn't exist)
    json_path_to_use = input_json_path
    if not os.path.exists(input_json_path):
        # Try with _multi-gpu suffix
        base_path = input_json_path.replace('.json', '')
        multi_gpu_path = f"{base_path}_multi-gpu.json"
        if os.path.exists(multi_gpu_path):
            json_path_to_use = multi_gpu_path
            print(f"Original JSON not found, using multi-gpu version: {multi_gpu_path}")
        else:
            raise FileNotFoundError(f"Neither {input_json_path} nor {multi_gpu_path} exist")
    
    with open(json_path_to_use, 'r') as f:
        pixmo_data = json.load(f)
    
    # Convert to the format expected by the original run_llm_judge
    print(f"Converting {split_name} split data to LLM judge format...")
    image_data = convert_pixmo_format_to_llm_judge_format(pixmo_data, split_name, num_images)
    
    # Initialize results structure for incremental saving (with optional resume)
    existing_correct = 0
    existing_total = 0
    existing_responses = {}
    processed_image_paths = set()
    
    if resume and save_results and os.path.exists(save_results):
        try:
            with open(save_results, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict) and 'responses' in existing_data:
                existing_responses = existing_data.get('responses', {})
                # Try to use existing counts if present; otherwise compute
                if all(k in existing_data for k in ['correct', 'total']):
                    existing_correct = int(existing_data.get('correct', 0))
                    existing_total = int(existing_data.get('total', 0))
                else:
                    # Compute from per-patch results
                    for _, results_list in existing_responses.items():
                        for r in results_list:
                            gptr = r.get('gpt_response', {}) if isinstance(r, dict) else {}
                            if gptr.get('interpretable') is True:
                                existing_correct += 1
                            existing_total += 1
                processed_image_paths = set(existing_responses.keys())
                print(f"Resume enabled: found {len(processed_image_paths)} completed images in existing results; skipping them.")
            else:
                # Backward-compat: file may be just the responses dict
                if isinstance(existing_data, dict):
                    existing_responses = existing_data
                    for _, results_list in existing_responses.items():
                        for r in results_list:
                            gptr = r.get('gpt_response', {}) if isinstance(r, dict) else {}
                            if gptr.get('interpretable') is True:
                                existing_correct += 1
                            existing_total += 1
                    processed_image_paths = set(existing_responses.keys())
                    print(f"Resume enabled (legacy file): found {len(processed_image_paths)} completed images; skipping them.")
        except Exception as e:
            print(f"Warning: Could not load existing results for resume: {e}")
            existing_correct = 0
            existing_total = 0
            existing_responses = {}
            processed_image_paths = set()
    
    # Filter images if specific indices are provided
    if image_indices is not None:
        image_data = [image_data[i] for i in image_indices if i < len(image_data)]
    
    # Apply resume skip logic
    if processed_image_paths:
        image_data = [entry for entry in image_data if entry.get('image_path') not in processed_image_paths]
    
    # Prepare responses dict and counters
    gpt_responses = defaultdict(list, ((k, v) for k, v in existing_responses.items())) if existing_responses else defaultdict(list)
    accuracy = existing_correct
    total = existing_total
    
    incremental_results = {
        'accuracy': accuracy,
        'correct': accuracy,
        'total': total,
        'responses': gpt_responses,
        'processed_images': len(processed_image_paths),
        'total_images': len(processed_image_paths) + len(image_data)
    }
    
    print(f"Processing {len(image_data)} images...")

    for idx, image_entry in enumerate(tqdm(image_data)):
        # Validate required keys
        if 'image_path' not in image_entry or 'patches' not in image_entry:
            print(f"Warning: Image entry {idx} missing required keys 'image_path' or 'patches'")
            continue
            
        image_path = image_entry['image_path']
        annotations = image_entry['patches']
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue
            
        # Process the original image to get both the processed image and mask
        try:
            processed_image, image_mask = process_image_with_mask(image_path)
            actual_patch_size = 512 / 24  # ~21.33
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            continue
        
        if not annotations:
            print(f"Warning: No annotations found for image {image_path}")
            continue
        
        # Sample valid patch positions that don't include padded areas
        try:
            sampled_positions = sample_valid_patch_positions(image_mask, bbox_size=bbox_size, num_samples=num_samples)
        except Exception as e:
            print(f"Warning: Could not sample patch positions for image {image_path}: {e}")
            continue
            
        if not sampled_positions:
            print(f"Warning: No valid patch positions found for image {image_path}")
            continue
            
        print(f"Sampled {len(sampled_positions)} valid positions for image {os.path.basename(image_path)}")
        
        for patch_row, patch_col in sampled_positions:
            bbox = calculate_square_bbox_from_patch(patch_row, patch_col, 
                                                  patch_size=actual_patch_size, size=bbox_size)
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            
            # Find the center patch and get its nearest neighbors
            center_row = patch_row + bbox_size // 2
            center_col = patch_col + bbox_size // 2
            
            # Find the patch that matches the center position
            high_conf_tokens = []
            for patch in annotations:
                if patch.get("patch_row") == center_row and patch.get("patch_col") == center_col:
                    high_conf_tokens = patch.get("nearest_neighbors", [])[:5]  # Top 5
                    break
            
            tokens = [token_info['token'] for token_info in high_conf_tokens]
            
            if not tokens:
                continue

            formatted_prompt = IMAGE_PROMPT.format(candidate_words=str(tokens))
            response = get_gpt_response(client, image_with_bbox, formatted_prompt)
            
            # Create result object
            result = {
                'patch_row': patch_row,
                'patch_col': patch_col,
                'bbox_size': bbox_size,
                'high_confidence_tokens': high_conf_tokens,
                'tokens_used': tokens,
                'gpt_response': response,
                'original_image_path': image_path,
                'processed_image_used': True,
                'image_index': idx,
                'ground_truth_caption': image_entry.get('ground_truth_caption', '')
            }
            gpt_responses[image_path].append(result)
            
            # Save image with bounding box if requested
            if save_images and images_dir:
                image_filename = f"image_{idx}_patch_{patch_row}_{patch_col}_bbox.jpg"
                image_save_path = images_dir / image_filename
                image_with_bbox.save(image_save_path, "JPEG", quality=95)
                result['saved_image_path'] = str(image_save_path)
            
            # Display results
            print(f"Image: {os.path.basename(image_path)}, Patch: ({patch_row}, {patch_col})")
            print(f"Tokens: {tokens}")
            print(f"Response: {response}")
            
            if response.get('interpretable') == True:
                accuracy += 1
            total += 1
            
            # Update incremental results
            incremental_results['accuracy'] = accuracy
            incremental_results['total'] = total
            incremental_results['correct'] = accuracy
            incremental_results['responses'] = gpt_responses
        
        # Save incremental results after each image
        incremental_results['processed_images'] = len(processed_image_paths) + idx + 1
        save_incremental_results(incremental_results, save_results, idx, len(image_data))
    
    # Calculate final accuracy
    final_accuracy = (accuracy * 100 / total) if total > 0 else 0
    print(f'Accuracy: {final_accuracy:.2f}% ({accuracy}/{total})')
    
    # Save results (now always save by default)
    results = {
        'accuracy': final_accuracy,
        'correct': accuracy,
        'total': total,
        'responses': gpt_responses
    }
    
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Results saved to {save_results}")
    else:
        print("Warning: No save_results path provided, results not saved to file")
    
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Run LLM judge evaluation on PixMo format data')
    
    # Required arguments
    parser.add_argument('--input_json', type=str, required=True,
                       help='Path to PixMo analysis JSON file')
    parser.add_argument('--api_key', type=str, 
                       help='OpenAI API key (can also be set via OPENAI_API_KEY env var)')
    
    # Optional arguments
    parser.add_argument('--split', type=str, choices=['train', 'validation'], default='train',
                       help='Which split to process (default: train)')
    parser.add_argument('--image_indices', type=int, nargs='+',
                       help='Specific image indices to process (default: all images)')
    parser.add_argument('--num_images', type=int,
                       help='Number of images to process (default: all in split)')
    parser.add_argument('--patch_size', type=float, default=21.33,
                       help='Size of each patch in pixels (default: 21.33)')
    parser.add_argument('--bbox_size', type=int, default=3,
                       help='Size of bounding box in patches (default: 3 for 3x3)')
    parser.add_argument('--num_samples', type=int, default=36,
                       help='Number of random patches to sample per image (default: 36)')
    parser.add_argument('--save_images', action='store_true',
                       help='Save images with bounding boxes to results directory')
    parser.add_argument('--save_results', type=str,
                       help='Path to save results json file (default: auto-generated from input path)')
    parser.add_argument('--model', type=str, default='gpt-5',
                       help='OpenAI model to use (default: gpt-5)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results file if present (skip completed images)')
    
    args = parser.parse_args()
    
    
    # Auto-generate save_results path if not provided
    if args.save_results is None:
        # Get the parent directory of the input JSON file
        input_path = Path(args.input_json)
        parent_dir = input_path.parent
        # Create a results filename based on the parent directory name and split
        parent_name = parent_dir.name
        results_filename = f"llm_judge_evaluation_{args.split}_{parent_name}.json"
        args.save_results = str(parent_dir / results_filename)
        print(f"Auto-generated results path: {args.save_results}")
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key must be provided via --api_key argument or OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        # Test the client with a simple call
        client.models.list()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)
    
    print(f"Starting LLM judge evaluation on PixMo data...")
    print(f"Input JSON file: {args.input_json}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Patch size: {args.patch_size}")
    print(f"Bounding box size: {args.bbox_size}x{args.bbox_size}")
    print(f"Number of samples per image: {args.num_samples}")
    if args.num_images:
        print(f"Number of images to process: {args.num_images}")
    if args.image_indices:
        print(f"Processing specific image indices: {args.image_indices}")
    if args.save_images:
        print("Images with bounding boxes will be saved to llm_judge_images/ directory")
    
    # Run the evaluation
    results = run_llm_judge_pixmo(
        client=client,
        input_json_path=args.input_json,
        split_name=args.split,
        image_indices=args.image_indices,
        num_images=args.num_images,
        patch_size=args.patch_size,
        bbox_size=args.bbox_size,
        save_images=args.save_images,
        save_results=args.save_results,
        num_samples=args.num_samples,
        resume=args.resume
    )
    
    print("\nEvaluation completed successfully!")
    print(f"Final accuracy: {results['accuracy']:.2f}%")


if __name__ == "__main__":
    main()

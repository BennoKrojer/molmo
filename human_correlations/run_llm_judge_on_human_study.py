#!/usr/bin/env python3
"""
Run LLM judge evaluation on the specific patches from the human study.
This ensures we're evaluating the same instances that humans judged.

Supports two data types:
- nn: Token-level data (interp_data_nn) with candidates as word list
- contextual: Contextual data (interp_data_contextual) with candidates as [sentence, token] tuples
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import time
import requests
from io import BytesIO

# Import existing utilities from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'llm_judge'))
from prompts import IMAGE_PROMPT, IMAGE_PROMPT_WITH_CROP
from utils import (
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    resize_and_pad
)
# Import word extraction from contextual script (same function used there)
from run_single_model_with_viz_contextual import extract_full_word_from_token


def convert_contextual_candidates_to_words(candidates):
    """
    Convert contextual candidates from [sentence, token] tuples to full words.
    Uses extract_full_word_from_token from run_single_model_with_viz_contextual.py.

    Args:
        candidates: List of [sentence, token] tuples

    Returns:
        List of full words extracted from the sentences
    """
    words = []
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
            sentence, token = candidate[0], candidate[1]
            word = extract_full_word_from_token(sentence, token)
            if word:
                words.append(word)
        else:
            # Fallback: if already a string, use as-is
            if isinstance(candidate, str):
                words.append(candidate)
    return words


def get_gpt_response(client, image, cropped_image, prompt, api_provider="openai", model="gpt-5"):
    """Get LLM response for image interpretability analysis."""
    import base64
    import io
    
    # Encode main image
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Add cropped image if provided
    crop_str = None
    if cropped_image is not None:
        buffered_crop = io.BytesIO()
        cropped_image.save(buffered_crop, format="PNG")
        crop_str = base64.b64encode(buffered_crop.getvalue()).decode("utf-8")
    
    if api_provider == "openrouter":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        if crop_str is not None:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{crop_str}"}
            })
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        response_text = response.choices[0].message.content
    else:
        # OpenAI GPT-5 format
        content = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_str}"},
            {"type": "input_text", "text": prompt},
        ]
        
        if crop_str is not None:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{crop_str}"})
        
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": content,
            }],
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        )
        
        response_text = response.output_text
    
    # Try to extract JSON
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        json_str = response_text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response: {response_text}")
            return {
                "interpretable": False,
                "concrete_words": [],
                "abstract_words": [],
                "global_words": [],
                "reasoning": response_text
            }
    else:
        print(f"No JSON found in response: {response_text}")
        return {
            "interpretable": False,
            "concrete_words": [],
            "abstract_words": [],
            "global_words": [],
            "reasoning": response_text
        }


def evaluate_human_study_instances(data_json_path, api_key, output_dir, use_cropped_region=True,
                                   skip_if_exists=False, resume=True, data_type='nn'):
    """
    Evaluate the specific instances from the human study.

    Args:
        data_json_path: Path to human study data.json
        api_key: OpenAI API key
        output_dir: Output directory for results
        use_cropped_region: Whether to include cropped region in prompt
        skip_if_exists: Skip if output file already exists
        resume: Resume from existing partial results
        data_type: Type of data - 'nn' for token-level or 'contextual' for sentence-level
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load human study data
    print(f"Loading human study data from {data_json_path}...")
    with open(data_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} instances to evaluate")
    
    # Output file
    output_file = output_path / "human_study_llm_results.json"
    
    # Check if we should skip
    if skip_if_exists and output_file.exists():
        print(f"Output file {output_file} already exists. Skipping.")
        return
    
    # Load existing results if resuming
    existing_results = {}
    if resume and output_file.exists():
        print(f"Loading existing results from {output_file}...")
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            existing_results = {r['instance_id']: r for r in existing_data.get('results', [])}
        print(f"Found {len(existing_results)} existing results")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Process each instance
    results = []
    for idx, instance in enumerate(tqdm(data, desc="Evaluating instances")):
        instance_id = instance.get('id')
        
        # Skip if already evaluated
        if instance_id in existing_results:
            results.append(existing_results[instance_id])
            continue
        
        try:
            # Get instance info
            image_url = instance.get('image_url')
            caption = instance.get('caption', '')
            raw_candidates = instance.get('candidates', [])
            patch_row = instance.get('patch_row')
            patch_col = instance.get('patch_col')
            patch_type = instance.get('patch_type', 'unknown')

            # For contextual data, also get layer info
            layer = instance.get('layer') if data_type == 'contextual' else None

            # Convert candidates based on data type
            if data_type == 'contextual':
                # Contextual: candidates are [sentence, token] tuples -> extract full words
                candidates = convert_contextual_candidates_to_words(raw_candidates)
            else:
                # NN: candidates are already words
                candidates = raw_candidates
            
            # Load and preprocess image (same as main LLM judge)
            try:
                # Download image from URL
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content))
                
                # Convert to numpy array and preprocess (same as main LLM judge)
                image_np = np.array(pil_image.convert('RGB'))
                processed_image_np, image_mask = resize_and_pad(image_np, (512, 512), normalize=False)
                processed_image_np = (processed_image_np * 255).astype(np.uint8)
                processed_image = Image.fromarray(processed_image_np)
                actual_patch_size = 512 / 24  # ~21.33 (same as main LLM judge)
            except Exception as e:
                print(f"\nError loading image for {instance_id}: {e}")
                results.append({
                    'instance_id': instance_id,
                    'error': f'Failed to load image: {e}',
                    'candidates': candidates,
                    'patch_row': patch_row,
                    'patch_col': patch_col
                })
                continue
            
            # Create bounding box using same utility as main LLM judge
            bbox_size = 3  # 3x3 patch region
            bbox = calculate_square_bbox_from_patch(patch_row, patch_col, 
                                                   patch_size=actual_patch_size, size=bbox_size)
            
            # Draw bounding box using same utility as main LLM judge
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            
            # Create cropped region
            cropped_image = None
            if use_cropped_region:
                cropped_image = processed_image.crop(bbox)
            
            # Format prompt
            if use_cropped_region:
                prompt = IMAGE_PROMPT_WITH_CROP.format(
                    caption=caption,
                    candidate_words=json.dumps(candidates)
                )
            else:
                prompt = IMAGE_PROMPT.format(
                    caption=caption,
                    candidate_words=json.dumps(candidates)
                )
            
            # Get LLM response
            gpt_response = get_gpt_response(client, image_with_bbox, cropped_image, prompt,
                                          api_provider="openai", model="gpt-5")
            
            # Save result
            result = {
                'instance_id': instance_id,
                'model': instance.get('model'),
                'image_url': image_url,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'patch_type': patch_type,
                'bbox': list(bbox),
                'candidates': candidates,
                'gpt_response': gpt_response,
                'caption': caption
            }
            # Add layer info for contextual data
            if data_type == 'contextual' and layer is not None:
                result['layer'] = layer
                result['raw_candidates'] = raw_candidates  # Keep original [sentence, token] tuples
            results.append(result)
            
            # Save intermediate results every 10 instances
            if (idx + 1) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump({
                        'total_instances': len(data),
                        'evaluated': len(results),
                        'results': results
                    }, f, indent=2)
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nError processing instance {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'instance_id': instance_id,
                'error': str(e),
                'candidates': candidates,
                'patch_row': patch_row,
                'patch_col': patch_col
            })
    
    # Save final results
    output_data = {
        'total_instances': len(data),
        'evaluated': len(results),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved results to {output_file}")
    
    # Compute statistics
    interpretable_count = sum(1 for r in results if r.get('gpt_response', {}).get('interpretable', False))
    total_with_response = sum(1 for r in results if 'gpt_response' in r)
    
    print(f"\nStatistics:")
    print(f"  Total instances: {len(results)}")
    print(f"  With LLM response: {total_with_response}")
    if total_with_response > 0:
        print(f"  Interpretable: {interpretable_count}/{total_with_response} ({100*interpretable_count/total_with_response:.1f}%)")
    else:
        print(f"  Interpretable: 0/0 (N/A)")


def main():
    parser = argparse.ArgumentParser(description='Run LLM judge on human study instances')
    parser.add_argument('--data-json', type=str,
                       default='interp_data/data.json',
                       help='Path to human study data.json (relative to script directory)')
    parser.add_argument('--api-key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--output-dir', type=str,
                       default='llm_judge_results',
                       help='Output directory (relative to script directory)')
    parser.add_argument('--use-cropped-region', action='store_true', default=True,
                       help='Include cropped region in prompt')
    parser.add_argument('--skip-if-exists', action='store_true',
                       help='Skip if output file exists')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from partial results')
    parser.add_argument('--data-type', type=str, default='nn', choices=['nn', 'contextual'],
                       help='Type of data: nn (token-level) or contextual (sentence-level with token)')

    args = parser.parse_args()

    # Resolve paths relative to script directory if not absolute
    script_dir = Path(__file__).parent
    data_json = Path(args.data_json)
    if not data_json.is_absolute():
        data_json = script_dir / data_json

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    evaluate_human_study_instances(
        str(data_json),
        args.api_key,
        str(output_dir),
        use_cropped_region=args.use_cropped_region,
        skip_if_exists=args.skip_if_exists,
        resume=args.resume,
        data_type=args.data_type
    )


if __name__ == '__main__':
    main()


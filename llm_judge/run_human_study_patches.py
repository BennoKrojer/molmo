#!/usr/bin/env python3
"""
Run LLM judge evaluation on the specific patches from the human study.
This ensures we're evaluating the same instances that humans judged.
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

# Import existing utilities
from prompts import IMAGE_PROMPT, IMAGE_PROMPT_WITH_CROP
from utils import (
    process_image_with_mask,
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    sample_valid_patch_positions
)
from olmo.data.pixmo_datasets import PixMoCap


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


def load_image_from_url(url):
    """Load image from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def evaluate_human_study_instances(data_json_path, api_key, output_dir, use_cropped_region=True,
                                   skip_if_exists=False, resume=True):
    """
    Evaluate the specific instances from the human study.
    
    Args:
        data_json_path: Path to human study data.json
        api_key: OpenAI API key
        output_dir: Output directory for results
        use_cropped_region: Whether to include cropped region in prompt
        skip_if_exists: Skip if output file already exists
        resume: Resume from existing partial results
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
            candidates = instance.get('candidates', [])
            patch_row = instance.get('patch_row')
            patch_col = instance.get('patch_col')
            patch_type = instance.get('patch_type', 'unknown')
            
            # Load image
            try:
                image = load_image_from_url(image_url)
            except Exception as e:
                print(f"\nError loading image for {instance_id}: {e}")
                results.append({
                    'instance_id': instance_id,
                    'error': f'Failed to load image: {e}',
                    'candidates': candidates
                })
                continue
            
            # Create bounding box for the patch
            # Assuming 24x24 patches (336/14 = 24)
            patch_size = 24
            bbox_size = 3  # 3x3 patch region
            
            x1 = patch_col * patch_size
            y1 = patch_row * patch_size
            x2 = x1 + bbox_size * patch_size
            y2 = y1 + bbox_size * patch_size
            
            # Ensure bbox is within image bounds
            img_width, img_height = image.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Draw bounding box on image
            image_with_bbox = image.copy()
            draw = ImageDraw.Draw(image_with_bbox)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Create cropped region
            cropped_image = None
            if use_cropped_region:
                cropped_image = image.crop((x1, y1, x2, y2))
            
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
                'bbox': [x1, y1, x2, y2],
                'candidates': candidates,
                'gpt_response': gpt_response,
                'caption': caption
            }
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
                'candidates': candidates
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
    print(f"  Interpretable: {interpretable_count}/{total_with_response} ({100*interpretable_count/total_with_response:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Run LLM judge on human study instances')
    parser.add_argument('--data-json', type=str,
                       default='human_correlations/interp_data/data.json',
                       help='Path to human study data.json')
    parser.add_argument('--api-key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--output-dir', type=str,
                       default='human_correlations/llm_judge_results',
                       help='Output directory')
    parser.add_argument('--use-cropped-region', action='store_true', default=True,
                       help='Include cropped region in prompt')
    parser.add_argument('--skip-if-exists', action='store_true',
                       help='Skip if output file exists')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from partial results')
    
    args = parser.parse_args()
    
    evaluate_human_study_instances(
        args.data_json,
        args.api_key,
        args.output_dir,
        use_cropped_region=args.use_cropped_region,
        skip_if_exists=args.skip_if_exists,
        resume=args.resume
    )


if __name__ == '__main__':
    main()


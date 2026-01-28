#!/usr/bin/env python3
"""Re-run LLM judge on the 6 missing instances using cached image."""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from openai import OpenAI
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'llm_judge'))
from prompts import IMAGE_PROMPT_WITH_CROP
from utils import calculate_square_bbox_from_patch, draw_bbox_on_image, resize_and_pad

def get_gpt_response(client, image, cropped_image, prompt):
    """Get LLM response for image interpretability analysis."""
    import base64
    import io
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    buffered_crop = io.BytesIO()
    cropped_image.save(buffered_crop, format="PNG")
    crop_str = base64.b64encode(buffered_crop.getvalue()).decode("utf-8")
    
    content = [
        {"type": "input_image", "image_url": f"data:image/png;base64,{img_str}"},
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": f"data:image/png;base64,{crop_str}"}
    ]
    
    response = client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "low"},
        text={"verbosity": "low"}
    )
    
    response_text = response.output_text
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
    return {"interpretable": False, "concrete_words": [], "abstract_words": [], "global_words": [], "reasoning": response_text}

def main():
    # Use existing API key file pattern from llm_judge scripts
    api_key_file = Path(__file__).parent.parent / 'llm_judge' / 'api_key.txt'
    if not api_key_file.exists():
        print(f"ERROR: API key file not found: {api_key_file}")
        sys.exit(1)
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    
    # Load cached image
    cached_image_path = Path("cached_images/image_00096.png")
    if not cached_image_path.exists():
        print(f"ERROR: Cached image not found: {cached_image_path}")
        sys.exit(1)
    
    pil_image = Image.open(cached_image_path).convert('RGB')
    image_np = np.array(pil_image)
    processed_image_np, _ = resize_and_pad(image_np, (512, 512), normalize=False)
    processed_image_np = (processed_image_np * 255).astype(np.uint8)
    processed_image = Image.fromarray(processed_image_np)
    actual_patch_size = 512 / 24
    
    # Load data.json to get the 6 missing instances
    with open("interp_data_nn/data.json") as f:
        data = json.load(f)
    
    missing_instances = [item for item in data if '_00096_' in item.get('id', '')]
    print(f"Found {len(missing_instances)} instances to re-run")
    
    # Load existing results
    with open("llm_judge_results/human_study_llm_results.json") as f:
        existing_data = json.load(f)
    
    results = existing_data['results']
    existing_ids = {r['instance_id'] for r in results if 'gpt_response' in r}
    
    client = OpenAI(api_key=api_key)
    
    for instance in missing_instances:
        instance_id = instance['id']
        if instance_id in existing_ids:
            print(f"  Skipping {instance_id} - already has gpt_response")
            continue
        
        print(f"Processing {instance_id}...")
        
        patch_row = instance['patch_row']
        patch_col = instance['patch_col']
        candidates = instance['candidates']
        caption = instance.get('caption', '')
        
        bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=3)
        image_with_bbox = draw_bbox_on_image(processed_image, bbox)
        cropped_image = processed_image.crop(bbox)
        
        prompt = IMAGE_PROMPT_WITH_CROP.format(
            caption=caption,
            candidate_words=json.dumps(candidates)
        )
        
        gpt_response = get_gpt_response(client, image_with_bbox, cropped_image, prompt)
        
        # Update existing result or add new one
        found = False
        for r in results:
            if r['instance_id'] == instance_id:
                r['gpt_response'] = gpt_response
                del r['error']  # Remove error field
                found = True
                break
        
        if not found:
            results.append({
                'instance_id': instance_id,
                'model': instance.get('model'),
                'image_url': instance.get('image_url'),
                'patch_row': patch_row,
                'patch_col': patch_col,
                'patch_type': instance.get('patch_type', 'unknown'),
                'bbox': list(bbox),
                'candidates': candidates,
                'gpt_response': gpt_response,
                'caption': caption
            })
        
        print(f"  interpretable: {gpt_response.get('interpretable', False)}")
        time.sleep(0.5)
    
    # Save updated results
    existing_data['results'] = results
    with open("llm_judge_results/human_study_llm_results.json", 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"\nUpdated results saved. Total with gpt_response: {sum(1 for r in results if 'gpt_response' in r)}")

if __name__ == '__main__':
    main()

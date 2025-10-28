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
from prompts import IMAGE_PROMPT
from utils import (
    calculate_square_bbox_from_center, 
    get_high_confidence_words,
    clip_bbox_to_image,
    draw_bbox_on_image,
    process_image_with_mask,
    sample_valid_patch_positions
)


def get_gpt_response(client, image, prompt):
    """
    Get GPT-5 response for image interpretability analysis.
    
    Args:
        image (PIL.Image): Image with red bounding box
        nn (list): List of nearest neighbor words/tokens
        
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
        

def run_llm_judge(client, input_json_path, num_images=None, image_indices=None, patch_size=28.0, bbox_size=3, show_images=False, save_results=None, num_samples=36):
    """
    Run LLM judge evaluation on image patches.
    
    Args:
        client: OpenAI client instance
        input_json_path (str): Path to JSON file containing list of dicts with 'image_path' and 'patches' keys
        image_indices (list): List of specific image indices to process (if None, process all images)
        patch_size (float): Size of each patch in pixels (calculated automatically from 672/24)
        bbox_size (int): Size of bounding box in patches (e.g., 3 for 3x3)
        show_images (bool): Whether to display images during processing
        save_results (str): Path to save results JSON file (optional)
        num_samples (int): Number of random patches to sample per image (default: 36)
    
    Returns:
        dict: Results including accuracy, total count, and individual responses
    """
    accuracy = 0
    total = 0
    gpt_responses = defaultdict(list)
    
    # Load the input JSON file
    try:
        with open(input_json_path, 'r') as f:
            image_data = json.load(f)
    except Exception as e:
        print(f"Error loading input JSON file {input_json_path}: {e}")
        return {"accuracy": 0, "correct": 0, "total": 0, "responses": {}}
    
    # Filter images if specific indices are provided
    if image_indices is not None:
        image_data = [image_data[i] for i in image_indices if i < len(image_data)]
    
    if num_images is not None:
        image_data = image_data[:num_images]
    
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
            actual_patch_size = 672 / 24  # = 28.0
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            continue
        
        if not annotations:
            print(f"Warning: No annotations found for image {image_path}")
            continue
        
        # Sample valid patch positions that don't include padded areas
        try:
            sampled_center_positions = sample_valid_patch_positions(image_mask, bbox_size=bbox_size, num_samples=num_samples)
        except Exception as e:
            print(f"Warning: Could not sample patch positions for image {image_path}: {e}")
            continue
            
        if not sampled_center_positions:
            print(f"Warning: No valid patch positions found for image {image_path}")
            continue
            
        print(f"Sampled {len(sampled_center_positions)} valid center positions for image {os.path.basename(image_path)}")
        
        for center_row, center_col in sampled_center_positions:
            bbox = calculate_square_bbox_from_center(center_row, center_col, 
                                                  patch_size=actual_patch_size, size=bbox_size)
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            
            # Get tokens from the center patch
            high_conf_tokens = get_high_confidence_words(annotations, center_row, center_col, 
                                                       size=bbox_size)
            
            tokens = [token_info['token'] for token_info in high_conf_tokens]
            
            if not tokens:
                continue

            formatted_prompt = IMAGE_PROMPT.format(candidate_words=str(tokens))
            response = get_gpt_response(client, image_with_bbox, formatted_prompt)
            
            # Create result object
            result = {
                'center_row': center_row,
                'center_col': center_col,
                'bbox_size': bbox_size,
                'high_confidence_tokens': high_conf_tokens,
                'tokens_used': tokens,
                'gpt_response': response,
                'original_image_path': image_path,
                'processed_image_used': True,
                'image_index': idx
            }
            gpt_responses[image_path].append(result)
            
            # Display results
            if show_images:
                image_with_bbox.show()
            
                print(f"Image: {os.path.basename(image_path)}, Center Patch: ({center_row}, {center_col})")
                print(f"Tokens: {tokens}")
                print(f"Formatted prompt: {formatted_prompt}")
                print(f"Response: {response}")
            
            if response.get('interpretable') == True:
                accuracy += 1
            total += 1
    
    # Calculate final accuracy
    final_accuracy = (accuracy * 100 / total) if total > 0 else 0
    print(f'Accuracy: {final_accuracy:.2f}% ({accuracy}/{total})')
    
    # Save results if requested
    results = {
        'accuracy': final_accuracy,
        'correct': accuracy,
        'total': total,
        'responses': gpt_responses
    }
    
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(gpt_responses, f, indent=4, default=str)
        print(f"Results saved to {save_results}")
    
    return results
    

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Run LLM judge evaluation on image patches')
    
    # Required arguments
    parser.add_argument('--input_json', type=str, required=True,
                       help='Path to JSON file containing list of dicts with image_path and patches keys')
    parser.add_argument('--api_key', type=str, 
                       help='OpenAI API key (can also be set via OPENAI_API_KEY env var)')
    
    # Optional arguments
    parser.add_argument('--image_indices', type=int, nargs='+',
                       help='Specific image indices to process (default: all images)')
    parser.add_argument('--patch_size', type=float, default=28.0,
                       help='Size of each patch in pixels (default: 28.0)')
    parser.add_argument('--bbox_size', type=int, default=3,
                       help='Size of bounding box in patches (default: 3 for 3x3)')
    parser.add_argument('--num_samples', type=int, default=36,
                       help='Number of random patches to sample per image (default: 36)')
    parser.add_argument('--show_images', action='store_true',
                       help='Display images during processing')
    parser.add_argument('--save_results', type=str,
                       help='Path to save results json file')
    parser.add_argument('--model', type=str, default='gpt-5',
                       help='OpenAI model to use (default: gpt-5)')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to process from the input JSON (default: all images)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_json):
        print(f"Error: Input JSON file '{args.input_json}' does not exist")
        sys.exit(1)
    
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
    
    print(f"Starting LLM judge evaluation...")
    print(f"Input JSON file: {args.input_json}")
    print(f"Model: {args.model}")
    print(f"Patch size: {args.patch_size}")
    print(f"Bounding box size: {args.bbox_size}x{args.bbox_size}")
    print(f"Number of samples per image: {args.num_samples}")
    if args.image_indices:
        print(f"Processing specific image indices: {args.image_indices}")
    
    # Run the evaluation
    try:
        results = run_llm_judge(
            client=client,
            input_json_path=args.input_json,
            image_indices=args.image_indices,
            patch_size=args.patch_size,
            bbox_size=args.bbox_size,
            show_images=args.show_images,
            save_results=args.save_results,
            num_samples=args.num_samples
        )
        
        print("\nEvaluation completed successfully!")
        print(f"Final accuracy: {results['accuracy']:.2f}%")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

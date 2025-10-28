#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation for Generated Captions

This script evaluates generated captions using GPT-4o as a judge, assessing:
- Faithfulness to the image content
- Accuracy of details mentioned
- Absence of hallucinations
- Overall quality

Usage:
    python eval_captioning_gpt-judge.py --results-file path/to/results.json --split validation
    python eval_captioning_gpt-judge.py --results-file path/to/results.json --split validation --max-images 50
"""

import json
import argparse
import base64
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import statistics

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Create OpenAI client using env var OPENAI_API_KEY
client = OpenAI()

def encode_image(image_path: str) -> str:
    """Convert image file to base64 string for OpenAI API."""
    try:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log.error(f"Error encoding image {image_path}: {e}")
        return ""

def extract_score_from_response(response: str) -> Optional[int]:
    """Extract the numerical score from the LLM response."""
    # Look for patterns like "Score: 8", "8/10", "8 out of 10", etc.
    patterns = [
        r'score:\s*(\d+)/10',
        r'score:\s*(\d+)\s*out\s*of\s*10',
        r'(\d+)/10',
        r'(\d+)\s*out\s*of\s*10',
        r'rating:\s*(\d+)',
        r'(\d+)\s*points',
    ]
    
    response_lower = response.lower()
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    
    # If no pattern found, try to find any number between 1-10
    numbers = re.findall(r'\b([1-9]|10)\b', response)
    if numbers:
        return int(numbers[0])
    
    return None

def create_visualization(image_path: str, generated_caption: str, ground_truth_caption: str, 
                        score: Optional[int], llm_response: str, output_path: str) -> bool:
    """
    Create a visualization showing the image, captions, and LLM evaluation.
    
    Args:
        image_path: Path to the original image
        generated_caption: The generated caption
        ground_truth_caption: The ground truth caption (for reference)
        score: The LLM score (1-10)
        llm_response: The full LLM evaluation response
        output_path: Where to save the visualization
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the original image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image to a reasonable size for visualization
            max_size = 512
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img_width, img_height = img.size
        
        # Create a new image with space for text
        margin = 20
        text_width = 600
        total_width = img_width + text_width + 3 * margin
        total_height = max(img_height + 2 * margin, 800)  # Minimum height for text
        
        # Create the visualization canvas
        vis_img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(vis_img)
        
        # Try to load a font, fall back to default if not available
        try:
            # Try to use a nice font if available
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            try:
                # Try system fonts
                title_font = ImageFont.truetype("arial.ttf", 24)
                subtitle_font = ImageFont.truetype("arial.ttf", 18)
                body_font = ImageFont.truetype("arial.ttf", 14)
            except:
                # Fall back to default font
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
        
        # Paste the original image
        vis_img.paste(img, (margin, margin))
        
        # Calculate text position
        text_x = img_width + 2 * margin
        text_y = margin
        
        # Draw title
        draw.text((text_x, text_y), "LLM Caption Evaluation", fill='black', font=title_font)
        text_y += 40
        
        # Draw score if available
        if score is not None:
            # Color code the score
            if score >= 8:
                score_color = 'green'
            elif score >= 6:
                score_color = 'orange'
            else:
                score_color = 'red'
            
            draw.text((text_x, text_y), f"Score: {score}/10", fill=score_color, font=subtitle_font)
            text_y += 30
        
        # Draw generated caption
        draw.text((text_x, text_y), "Generated Caption:", fill='black', font=subtitle_font)
        text_y += 25
        
        # Wrap and draw the generated caption
        caption_lines = []
        words = generated_caption.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            if bbox[2] <= text_width - 20:  # Leave some margin
                current_line = test_line
            else:
                if current_line:
                    caption_lines.append(current_line)
                current_line = word
        if current_line:
            caption_lines.append(current_line)
        
        for line in caption_lines:
            draw.text((text_x + 10, text_y), line, fill='blue', font=body_font)
            text_y += 20
        
        text_y += 20
        
        # Draw ground truth caption (for reference)
        draw.text((text_x, text_y), "Ground Truth (Reference):", fill='black', font=subtitle_font)
        text_y += 25
        
        # Wrap and draw the ground truth caption
        gt_lines = []
        words = ground_truth_caption.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            if bbox[2] <= text_width - 20:
                current_line = test_line
            else:
                if current_line:
                    gt_lines.append(current_line)
                current_line = word
        if current_line:
            gt_lines.append(current_line)
        
        for line in gt_lines:
            draw.text((text_x + 10, text_y), line, fill='gray', font=body_font)
            text_y += 20
        
        text_y += 20
        
        # Draw LLM evaluation
        draw.text((text_x, text_y), "LLM Evaluation:", fill='black', font=subtitle_font)
        text_y += 25
        
        # Wrap and draw the LLM response
        response_lines = []
        words = llm_response.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            if bbox[2] <= text_width - 20:
                current_line = test_line
            else:
                if current_line:
                    response_lines.append(current_line)
                current_line = word
        if current_line:
            response_lines.append(current_line)
        
        # Limit the number of lines to avoid overflow
        max_lines = 15
        if len(response_lines) > max_lines:
            response_lines = response_lines[:max_lines]
            response_lines.append("... (truncated)")
        
        for line in response_lines:
            draw.text((text_x + 10, text_y), line, fill='black', font=body_font)
            text_y += 18
        
        # Save the visualization
        vis_img.save(output_path, quality=95)
        return True
        
    except Exception as e:
        log.error(f"Error creating visualization: {e}")
        return False

def evaluate_caption_with_llm(image_base64: str, generated_caption: str) -> Tuple[Optional[int], str]:
    """
    Evaluate a generated caption using GPT-4o as a judge, looking only at the image.
    
    Returns:
        Tuple of (score, full_response)
    """
    
    prompt = f"""You are an expert evaluator of image captions. Your task is to assess how well a generated caption describes the image you can see.

EVALUATION CRITERIA:
1. **Faithfulness**: Does the generated caption accurately describe what's actually visible in the image?
2. **Detail Accuracy**: Does it mention the main details and elements present in the image?
3. **No Hallucinations**: Does it avoid describing things that are not actually in the image?
4. **Completeness**: Does it capture the key aspects of the image?

GENERATED CAPTION: "{generated_caption}"

Please analyze the generated caption by looking at the image and assess how well it describes what you can see. Provide a score from 1 to 10, where:
- 10: Perfect - highly faithful, accurate, complete, no hallucinations
- 8-9: Excellent - very faithful with minor omissions or slight inaccuracies
- 6-7: Good - generally accurate but missing some details or has minor inaccuracies
- 4-5: Fair - some accuracy but missing important details or has notable inaccuracies
- 2-3: Poor - significant inaccuracies or hallucinations
- 1: Very Poor - largely incorrect or hallucinated content

Provide your reasoning and then give your final score in the format: "Score: X/10"

Your evaluation:"""

    try:
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        }
    ],
            max_tokens=1000,
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        
        full_response = response.choices[0].message.content
        score = extract_score_from_response(full_response)
        
        return score, full_response
        
    except Exception as e:
        log.error(f"Error calling OpenAI API: {e}")
        return None, f"API Error: {e}"

def load_results_and_images(results_file: Path, split: str, max_images: int) -> Tuple[List[Dict], Path]:
    """Load results from JSON file and find images directory."""
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    log.info(f"Loading results from {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Get split data
    split_data = results.get("splits", {}).get(split, {})
    if not split_data:
        raise ValueError(f"No data found for split '{split}'")
    
    images = split_data.get("images", [])
    if not images:
        raise ValueError(f"No images found in split '{split}'")
    
    log.info(f"Found {len(images)} images in {split} split")
    
    # Limit number of images to process
    images_to_process = images[:max_images]
    log.info(f"Processing {len(images_to_process)} images")
    
    # Find images directory
    images_dir = results_file.parent / "images"
    if not images_dir.exists():
        log.warning(f"Images directory not found at {images_dir}")
        images_dir = None
    
    return images_to_process, images_dir

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated captions using LLM-as-a-judge")
    
    parser.add_argument("--results-file", type=str, required=True,
                       help="Path to the JSON results file from the generation script")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Which split to evaluate (default: validation)")
    parser.add_argument("--max-images", type=int, default=2,
                       help="Maximum number of images to evaluate (default: 100)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file for evaluation results (default: auto-generated)")
    parser.add_argument("--create-visualizations", action="store_true",
                       help="Create visualization images showing image, caption, and evaluation")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    
    try:
        # Load results and find images
        images_to_process, images_dir = load_results_and_images(results_file, args.split, args.max_images)
        
        # Prepare output file path
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = results_file.parent / f"llm_judge_evaluation_{args.split}.json"
        
        # Create visualization directory if requested
        viz_dir = None
        if args.create_visualizations:
            viz_dir = results_file.parent / "visualized_llm_caption_judgement"
            viz_dir.mkdir(exist_ok=True)
            log.info(f"Visualizations will be saved to {viz_dir}")
        
        # Initialize results
        evaluation_results = {
            "results_file": str(results_file),
            "split": args.split,
            "max_images": args.max_images,
            "evaluations": [],
            "summary": {}
        }
        
        # Process each image
        scores = []
        successful_evaluations = 0
        
        for i, image_data in enumerate(images_to_process):
            image_idx = image_data.get("image_idx", i)
            ground_truth_caption = image_data.get("ground_truth_caption", "")
            generated_caption = image_data.get("generated_response", "")
            image_filename = image_data.get("image_filename", "")
            
            log.info(f"Evaluating image {i+1}/{len(images_to_process)} (index {image_idx})")
            
            # Check if we have the required data
            if not generated_caption:
                log.warning(f"Skipping image {image_idx}: missing generated caption data")
                continue
            
            # Try to load image
            image_base64 = ""
            image_path = None
            if images_dir and image_filename:
                image_path = images_dir / image_filename
                if image_path.exists():
                    image_base64 = encode_image(str(image_path))
                else:
                    log.warning(f"Image file not found: {image_path}")
            
            if not image_base64:
                log.warning(f"Skipping image {image_idx}: could not load image")
                continue
            
            # Evaluate with LLM
            score, full_response = evaluate_caption_with_llm(image_base64, generated_caption)
            
            evaluation_result = {
                "image_idx": image_idx,
                "image_filename": image_filename,
                "ground_truth_caption": ground_truth_caption,  # Keep for reference
                "generated_caption": generated_caption,
                "score": score,
                "llm_response": full_response,
                "evaluation_successful": score is not None
            }
            
            evaluation_results["evaluations"].append(evaluation_result)
            
            if score is not None:
                scores.append(score)
                successful_evaluations += 1
                log.info(f"Image {image_idx}: Score {score}/10")
                
                # Create visualization if requested
                if args.create_visualizations and image_path:
                    viz_filename = f"eval_image_{image_idx:04d}_score_{score}.jpg"
                    viz_path = viz_dir / viz_filename
                    success = create_visualization(
                        str(image_path), 
                        generated_caption, 
                        ground_truth_caption, 
                        score, 
                        full_response, 
                        str(viz_path)
                    )
                    if success:
                        log.info(f"Created visualization: {viz_path}")
                    else:
                        log.warning(f"Failed to create visualization for image {image_idx}")
            else:
                log.warning(f"Image {image_idx}: Could not extract score from LLM response")
        
        # Calculate summary statistics
        if scores:
            evaluation_results["summary"] = {
                "total_images": len(images_to_process),
                "successful_evaluations": successful_evaluations,
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores),
                "max_score": max(scores),
                "score_distribution": {
                    "1-2": len([s for s in scores if 1 <= s <= 2]),
                    "3-4": len([s for s in scores if 3 <= s <= 4]),
                    "5-6": len([s for s in scores if 5 <= s <= 6]),
                    "7-8": len([s for s in scores if 7 <= s <= 8]),
                    "9-10": len([s for s in scores if 9 <= s <= 10])
                }
            }
            
            # Print summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Total images processed: {len(images_to_process)}")
            print(f"Successful evaluations: {successful_evaluations}")
            print(f"Mean score: {evaluation_results['summary']['mean_score']:.2f}/10")
            print(f"Median score: {evaluation_results['summary']['median_score']:.2f}/10")
            print(f"Standard deviation: {evaluation_results['summary']['std_score']:.2f}")
            print(f"Score range: {evaluation_results['summary']['min_score']}-{evaluation_results['summary']['max_score']}")
            print("\nScore Distribution:")
            for range_label, count in evaluation_results['summary']['score_distribution'].items():
                percentage = (count / successful_evaluations) * 100
                print(f"  {range_label}: {count} images ({percentage:.1f}%)")
        else:
            log.error("No successful evaluations completed")
            evaluation_results["summary"] = {
                "error": "No successful evaluations"
            }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        log.info(f"Evaluation results saved to {output_file}")
        
        if args.create_visualizations and viz_dir:
            log.info(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        log.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()

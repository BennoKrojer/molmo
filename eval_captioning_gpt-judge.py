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
import time
import numpy as np

# Optional dataset fallback for image loading
try:
    from olmo.data.pixmo_datasets import PixMoCap
    HAS_PIXMO = True
except Exception:
    HAS_PIXMO = False

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
        
        # Helper: safe text wrapping with fallback if font measurement fails
        def safe_wrap(draw_obj, text, font_obj, max_width_px, approx_char_px: int = 8):
            if not isinstance(text, str):
                text = ""
            words = text.split()
            if not words:
                return []
            lines = []
            current = ""
            try:
                for word in words:
                    test = (current + " " + word) if current else word
                    bbox = draw_obj.textbbox((0, 0), test, font=font_obj)
                    if bbox[2] <= max_width_px - 20:
                        current = test
                    else:
                        if current:
                            lines.append(current)
                        current = word
                if current:
                    lines.append(current)
                return lines
            except Exception:
                # Fallback: approximate wrap without font metrics
                if approx_char_px <= 0:
                    approx_char_px = 8
                max_chars = max(10, max_width_px // approx_char_px)
                line = []
                count = 0
                for word in words:
                    wlen = len(word) + (1 if count > 0 else 0)
                    if count + wlen <= max_chars:
                        line.append(word)
                        count += wlen
                    else:
                        if line:
                            lines.append(" ".join(line))
                        line = [word]
                        count = len(word)
                if line:
                    lines.append(" ".join(line))
                return lines

        # Helper: measure text height safely
        def measure_text_height(draw_obj, text, font_obj, fallback: int = 18):
            try:
                bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
                return max(1, bbox[3] - bbox[1])
            except Exception:
                return fallback

        # Helper: draw a single line with limit
        def draw_line_with_limit(draw_obj, x, y, text, font_obj, fill_color, bottom_limit_px, fallback_height: int = 18):
            h = measure_text_height(draw_obj, text, font_obj, fallback=fallback_height)
            if y + h > bottom_limit_px:
                # Try to place truncation marker if space allows
                th = measure_text_height(draw_obj, "... (truncated)", font_obj, fallback=fallback_height)
                if y + th <= bottom_limit_px:
                    draw_obj.text((x, y), "... (truncated)", fill=fill_color, font=font_obj)
                    y += th
                return y, True
            draw_obj.text((x, y), text, fill=fill_color, font=font_obj)
            y += h
            return y, False

        # Helper: draw wrapped lines within bottom limit, truncate if needed
        def draw_lines_with_limit(draw_obj, x, y, lines, font_obj, fill_color, line_spacing_px, bottom_limit_px):
            truncated = False
            for line in lines:
                y, was_trunc = draw_line_with_limit(draw_obj, x, y, line, font_obj, fill_color, bottom_limit_px)
                if was_trunc:
                    truncated = True
                    break
                # spacing after each line
                if y + line_spacing_px > bottom_limit_px:
                    # can't place spacing, safe exit
                    truncated = True
                    break
                y += line_spacing_px
            return y, truncated

        # Helper: draw a header/label respecting bottom limit
        def draw_label_with_limit(draw_obj, x, y, text, font_obj, fill_color, bottom_limit_px, spacing_after_px):
            y, trunc = draw_line_with_limit(draw_obj, x, y, text, font_obj, fill_color, bottom_limit_px, fallback_height=22)
            if trunc:
                return y, True
            if y + spacing_after_px > bottom_limit_px:
                return y, True
            return y + spacing_after_px, False
        
        # Paste the original image
        vis_img.paste(img, (margin, margin))
        
        # Calculate text position
        text_x = img_width + 2 * margin
        text_y = margin
        
        # Bottom limit for all right-column text
        bottom_limit = total_height - margin
        
        # Draw title
        text_y, stop = draw_label_with_limit(draw, text_x, text_y, "LLM Caption Evaluation", title_font, 'black', bottom_limit, spacing_after_px=16)
        if stop:
            return True
        
        # Draw score if available
        if score is not None:
            # Color code the score
            if score >= 8:
                score_color = 'green'
            elif score >= 6:
                score_color = 'orange'
            else:
                score_color = 'red'
            
            text_y, stop = draw_label_with_limit(draw, text_x, text_y, f"Score: {score}/10", subtitle_font, score_color, bottom_limit, spacing_after_px=12)
            if stop:
                return True
        
        # Draw generated caption
        text_y, stop = draw_label_with_limit(draw, text_x, text_y, "Generated Caption:", subtitle_font, 'black', bottom_limit, spacing_after_px=8)
        if stop:
            return True
        
        # Wrap and draw the generated caption
        caption_lines = safe_wrap(draw, generated_caption, body_font, text_width)
        
        text_y, _ = draw_lines_with_limit(draw, text_x + 10, text_y, caption_lines, body_font, 'blue', 4, bottom_limit)
        if text_y >= bottom_limit:
            return True
        if text_y + 12 > bottom_limit:
            return True
        text_y += 12
        
        # Draw ground truth caption (for reference)
        text_y, stop = draw_label_with_limit(draw, text_x, text_y, "Ground Truth (Reference):", subtitle_font, 'black', bottom_limit, spacing_after_px=8)
        if stop:
            return True
        
        # Wrap and draw the ground truth caption
        gt_lines = safe_wrap(draw, ground_truth_caption, body_font, text_width)
        
        text_y, _ = draw_lines_with_limit(draw, text_x + 10, text_y, gt_lines, body_font, 'gray', 4, bottom_limit)
        if text_y >= bottom_limit:
            return True
        if text_y + 12 > bottom_limit:
            return True
        text_y += 12
        
        # Draw LLM evaluation
        text_y, stop = draw_label_with_limit(draw, text_x, text_y, "LLM Evaluation:", subtitle_font, 'black', bottom_limit, spacing_after_px=8)
        if stop:
            return True
        
        # Wrap and draw the LLM response
        response_lines = safe_wrap(draw, llm_response, body_font, text_width)
        
        # Draw response within available space; helper will handle truncation
        text_y, _ = draw_lines_with_limit(draw, text_x + 10, text_y, response_lines, body_font, 'black', 4, bottom_limit)
        
        # Save the visualization
        vis_img.save(output_path, quality=95)
        return True
        
    except Exception as e:
        log.error(f"Error creating visualization: {e}")
        return False

def _try_parse_structured_judgement(response: str) -> Optional[Dict]:
    """Parse a strict JSON judgement with sub-scores if present.
    Expected keys: faithfulness, detail_accuracy, hallucinations, completeness, overall_score, reasoning.
    Returns a dict or None if parsing fails.
    """
    text = response.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            obj = json.loads(json_str)
            required_keys = [
                "faithfulness",
                "detail_accuracy",
                "hallucinations",
                "completeness",
                "overall_score",
                "reasoning",
            ]
            if all(k in obj for k in required_keys):
                for k in ["faithfulness", "detail_accuracy", "hallucinations", "completeness", "overall_score"]:
                    obj[k] = int(obj[k])
                return obj
        except Exception:
            return None
    return None


def evaluate_caption_with_llm(image_base64: str, generated_caption: str, judge_model: str = "gpt-4o") -> Tuple[Optional[int], str, Optional[Dict]]:
    """
    Evaluate a generated caption using an LLM judge, looking only at the image.
    Returns (score, full_response, structured_dict|None).
    """
    prompt = (
        "You are an expert evaluator of image captions. Assess how well the generated caption describes the visible content.\n\n"
        "Return ONLY a strict JSON object with these keys and integer scores (1-10):\n"
        "{\"faithfulness\": int, \"detail_accuracy\": int, \"hallucinations\": int, \"completeness\": int, \"overall_score\": int, \"reasoning\": string}.\n"
        "Definitions: Faithfulness=accuracy to image; Detail_accuracy=coverage of salient details; "
        "Hallucinations=absence of non-existent content (higher=better); Completeness=captures key aspects; "
        "Overall_score=holistic quality.\n\n"
        f"Generated_caption: \"{generated_caption}\""
    )

    try:
        last_err = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=judge_model,
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
                    temperature=0.1,
                )
                break
            except Exception as e:
                last_err = e
                sleep_s = 2 * (attempt + 1)
                log.warning(f"LLM judge call failed (attempt {attempt+1}/3): {e}. Retrying in {sleep_s}s...")
                time.sleep(sleep_s)
        else:
            raise last_err
        
        full_response = response.choices[0].message.content
        structured = _try_parse_structured_judgement(full_response)
        if structured is not None:
            score = int(structured.get("overall_score"))
            return score, full_response, structured
        score = extract_score_from_response(full_response)
        return score, full_response, None
        
    except Exception as e:
        log.error(f"Error calling OpenAI API: {e}")
        return None, f"API Error: {e}", None

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
    
    parser.add_argument("--results-file", "--results_file", type=str, required=True,
                       help="Path to the JSON results file from the generation script")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                       help="Which split to evaluate (default: validation)")
    parser.add_argument("--max-images", type=int, default=100,
                       help="Maximum number of images to evaluate (default: 100)")
    parser.add_argument("--output-file", "--output_file", type=str, default=None,
                       help="Output file for evaluation results (default: auto-generated)")
    parser.add_argument("--create-visualizations", action="store_true",
                       help="Create visualization images showing image, caption, and evaluation")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                       help="LLM judge model identifier (default: gpt-4o)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from an existing output file and skip evaluated items")
    parser.add_argument("--evaluate-ground-truth", action="store_true",
                       help="Evaluate ground-truth captions instead of generated (upper bound)")
    parser.add_argument("--fallback-dataset-images", action="store_true",
                       help="If images are missing/unreadable, load from PixMoCap by image_idx for evaluation/visualization")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    
    try:
        # Prepare output file path
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            # If evaluating ground-truth as an upper bound, default to project root
            if args.evaluate_ground_truth:
                project_root = Path(__file__).resolve().parent
                output_file = project_root / f"llm_judge_upper_bound_{args.split}.json"
            else:
                # Save alongside input JSON, using a similar stem
                output_file = results_file.parent / f"{results_file.stem}_llm_judge_{args.split}.json"

        # Fast path: if visualization requested and an evaluation JSON already exists, only create visualizations
        if args.create_visualizations and output_file.exists() and not args.resume:
            log.info(f"Visualization-only mode: using existing evaluations at {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            # Determine images dir from the generation results location
            images_dir = results_file.parent / "images"
            if not images_dir.exists():
                log.warning(f"Images directory not found at {images_dir}")
                images_dir = None
            viz_dir = results_file.parent / "visualized_llm_caption_judgement"
            viz_dir.mkdir(exist_ok=True)

            made = 0
            for ev in evaluation_results.get("evaluations", []):
                image_filename = ev.get("image_filename")
                image_path = None
                if images_dir and image_filename:
                    candidate = images_dir / image_filename
                    if candidate.exists():
                        image_path = candidate
                if image_path is None and args.fallback_dataset_images and HAS_PIXMO:
                    try:
                        ds = PixMoCap(split=args.split, mode="captions")
                        ex = ds.get(ev.get("image_idx", 0), np.random)
                        ds_path = Path(ex["image"]) if ex and "image" in ex else None
                        if ds_path and ds_path.exists():
                            image_path = ds_path
                            log.info(f"Visualization fallback: loaded dataset image for idx {ev.get('image_idx')}")
                    except Exception as e:
                        log.warning(f"Visualization fallback failed for idx {ev.get('image_idx')}: {e}")
                if image_path is None:
                    continue
                score = ev.get("score")
                if score is None:
                    continue
                viz_filename = f"eval_image_{ev.get('image_idx', 0):04d}_score_{score}.jpg"
                viz_path = viz_dir / viz_filename
                if viz_path.exists():
                    continue
                created = create_visualization(
                    str(image_path),
                    ev.get("evaluated_caption", ev.get("generated_caption", "")),
                    ev.get("ground_truth_caption", ""),
                    score,
                    ev.get("llm_response", ""),
                    str(viz_path),
                )
                if created:
                    made += 1
            log.info(f"Created {made} visualization(s) at {viz_dir}")
            return

        # Load results and find images (normal path)
        images_to_process, images_dir = load_results_and_images(results_file, args.split, args.max_images)
        
        # Create visualization directory if requested
        viz_dir = None
        if args.create_visualizations:
            viz_dir = results_file.parent / "visualized_llm_caption_judgement"
            viz_dir.mkdir(exist_ok=True)
            log.info(f"Visualizations will be saved to {viz_dir}")
        
        # Initialize or resume results
        evaluation_results = {
            "results_file": str(results_file),
            "split": args.split,
            "max_images": args.max_images,
            "judge_model": args.judge_model,
            "evaluations": [],
            "summary": {}
        }
        failure_counts = defaultdict(int)
        failure_examples = defaultdict(list)
        processed_indices = set()
        if args.resume and output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    prev = json.load(f)
                if isinstance(prev, dict) and "evaluations" in prev:
                    evaluation_results = prev
                    # Skip ONLY previously successful evaluations; retry failures
                    processed_indices = {e.get("image_idx") for e in prev.get("evaluations", []) if e.get("evaluation_successful")}
                    num_prev = len(prev.get("evaluations", []))
                    num_success = len(processed_indices)
                    num_failed = num_prev - num_success
                    log.info(f"Resuming from {output_file}: {num_success} successes will be skipped, {num_failed} failures will be retried")
            except Exception as e:
                log.warning(f"Could not resume from {output_file}: {e}")
        
        # Process each image
        scores = []
        successful_evaluations = 0
        
        for i, image_data in enumerate(images_to_process):
            image_idx = image_data.get("image_idx", i)
            if args.resume and image_idx in processed_indices:
                log.info(f"Skipping image {image_idx}: already evaluated (resume)")
                continue
            ground_truth_caption = image_data.get("ground_truth_caption", "")
            generated_caption = image_data.get("generated_response", "")
            image_filename = image_data.get("image_filename", "")
            
            log.info(f"Evaluating image {i+1}/{len(images_to_process)} (index {image_idx})")
            
            # Choose which caption to evaluate
            if args.evaluate_ground_truth:
                caption_to_eval = ground_truth_caption
                evaluated_text_kind = "ground_truth"
            else:
                caption_to_eval = generated_caption
                evaluated_text_kind = "generated"
            
            # Check if we have the required data
            if not caption_to_eval:
                log.warning(f"Skipping image {image_idx}: missing {evaluated_text_kind} caption data")
                evaluation_results["evaluations"].append({
                    "image_idx": image_idx,
                    "image_filename": image_filename,
                    "ground_truth_caption": ground_truth_caption,
                    "generated_caption": generated_caption,
                    "evaluated_caption": caption_to_eval,
                    "evaluated_text_kind": evaluated_text_kind,
                    "evaluation_successful": False,
                    "failure_reason": "missing_caption",
                })
                failure_counts["missing_caption"] += 1
                if len(failure_examples["missing_caption"]) < 10:
                    failure_examples["missing_caption"].append(image_idx)
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
            # Fallback: load from dataset if requested
            if not image_base64 and args.fallback_dataset_images and HAS_PIXMO:
                try:
                    ds = PixMoCap(split=args.split, mode="captions")
                    ex = ds.get(image_idx, np.random)
                    ds_image_path = Path(ex["image"]) if ex and "image" in ex else None
                    if ds_image_path and ds_image_path.exists():
                        image_base64 = encode_image(str(ds_image_path))
                        image_path = ds_image_path
                        log.info(f"Loaded image via dataset fallback for idx {image_idx}")
                except Exception as e:
                    log.warning(f"Dataset fallback failed for idx {image_idx}: {e}")
            
            if not image_base64:
                log.warning(f"Skipping image {image_idx}: could not load image")
                evaluation_results["evaluations"].append({
                    "image_idx": image_idx,
                    "image_filename": image_filename,
                    "ground_truth_caption": ground_truth_caption,
                    "generated_caption": generated_caption,
                    "evaluated_caption": caption_to_eval,
                    "evaluated_text_kind": evaluated_text_kind,
                    "evaluation_successful": False,
                    "failure_reason": "missing_image",
                })
                failure_counts["missing_image"] += 1
                if len(failure_examples["missing_image"]) < 10:
                    failure_examples["missing_image"].append(image_idx)
                continue
            
            # Evaluate with LLM
            score, full_response, structured = evaluate_caption_with_llm(image_base64, caption_to_eval, judge_model=args.judge_model)
            
            evaluation_result = {
                "image_idx": image_idx,
                "image_filename": image_filename,
                "ground_truth_caption": ground_truth_caption,  # Keep for reference
                "generated_caption": generated_caption,
                "evaluated_caption": caption_to_eval,
                "evaluated_text_kind": evaluated_text_kind,
                "score": score,
                "llm_response": full_response,
                "evaluation_successful": score is not None
            }
            if structured is not None:
                evaluation_result["structured"] = structured
            
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
                # Record explicit failure reason
                reason = "score_parse_failed"
                if isinstance(full_response, str) and full_response.startswith("API Error:"):
                    reason = "judge_api_error"
                evaluation_result["failure_reason"] = reason
                failure_counts[reason] += 1
                if len(failure_examples[reason]) < 10:
                    failure_examples[reason].append(image_idx)
                log.warning(f"Image {image_idx}: evaluation failed ({reason})")
        
        # Calculate summary statistics across ALL evaluations in file (supports resume)
        all_evals = evaluation_results.get("evaluations", [])
        # Successful items and scores
        all_successes = [ev for ev in all_evals if ev.get("evaluation_successful") and isinstance(ev.get("score"), (int, float))]
        all_scores = [float(ev.get("score")) for ev in all_successes]
        total_images_summary = len(all_evals)

        # Aggregate sub-scores if present
        subs = {"faithfulness": [], "detail_accuracy": [], "hallucinations": [], "completeness": []}
        for ev in all_successes:
            st = ev.get("structured")
            if st:
                for k in subs.keys():
                    v = st.get(k)
                    if isinstance(v, (int, float)):
                        subs[k].append(float(v))

        # Failures recomputed from all evaluations
        fail_counts_all = defaultdict(int)
        fail_examples_all = defaultdict(list)
        for ev in all_evals:
            if not ev.get("evaluation_successful"):
                reason = ev.get("failure_reason", "unknown")
                fail_counts_all[reason] += 1
                if len(fail_examples_all[reason]) < 10:
                    fail_examples_all[reason].append(ev.get("image_idx"))

        if all_scores:
            summary = {
                "total_images": total_images_summary,
                "successful_evaluations": len(all_successes),
                "mean_score": statistics.mean(all_scores),
                "median_score": statistics.median(all_scores),
                "std_score": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "score_distribution": {
                    "1-2": len([s for s in all_scores if 1 <= s <= 2]),
                    "3-4": len([s for s in all_scores if 3 <= s <= 4]),
                    "5-6": len([s for s in all_scores if 5 <= s <= 6]),
                    "7-8": len([s for s in all_scores if 7 <= s <= 8]),
                    "9-10": len([s for s in all_scores if 9 <= s <= 10])
                },
                "judge_model": args.judge_model,
                "failures": {k: int(v) for k, v in fail_counts_all.items()},
                "failure_examples": {k: v for k, v in fail_examples_all.items()},
            }
            for k, arr in subs.items():
                if arr:
                    summary[f"mean_{k}"] = statistics.mean(arr)
            evaluation_results["summary"] = summary
            
            # Print summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Total images processed: {total_images_summary}")
            print(f"Successful evaluations: {len(all_successes)}")
            print(f"Mean score: {evaluation_results['summary']['mean_score']:.2f}/10")
            print(f"Median score: {evaluation_results['summary']['median_score']:.2f}/10")
            print(f"Standard deviation: {evaluation_results['summary']['std_score']:.2f}")
            print(f"Score range: {evaluation_results['summary']['min_score']}-{evaluation_results['summary']['max_score']}")
            print("\nScore Distribution:")
            for range_label, count in evaluation_results['summary']['score_distribution'].items():
                percentage = (count / max(1, len(all_successes))) * 100
                print(f"  {range_label}: {count} images ({percentage:.1f}%)")
        else:
            log.error("No successful evaluations completed")
            evaluation_results["summary"] = {
                "error": "No successful evaluations",
                "total_images": total_images_summary,
                "successful_evaluations": 0,
                "failures": {k: int(v) for k, v in fail_counts_all.items()},
                "failure_examples": {k: v for k, v in fail_examples_all.items()},
            }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        log.info(f"Evaluation results saved to {output_file}")
        
        # Backfill visualizations for all evaluated items (including resumed ones)
        if args.create_visualizations:
            viz_dir = results_file.parent / "visualized_llm_caption_judgement"
            viz_dir.mkdir(exist_ok=True)
            made = 0
            for ev in evaluation_results.get("evaluations", []):
                image_filename = ev.get("image_filename")
                if not images_dir or not image_filename:
                    continue
                image_path = images_dir / image_filename
                if not image_path.exists():
                    continue
                score = ev.get("score")
                if score is None:
                    continue
                viz_filename = f"eval_image_{ev.get('image_idx', 0):04d}_score_{score}.jpg"
                viz_path = viz_dir / viz_filename
                if viz_path.exists():
                    continue
                created = create_visualization(
                    str(image_path),
                    ev.get("evaluated_caption", ev.get("generated_caption", "")),
                    ev.get("ground_truth_caption", ""),
                    score,
                    ev.get("llm_response", ""),
                    str(viz_path),
                )
                if created:
                    made += 1
            log.info(f"Ensured visualizations, newly created {made} file(s) at {viz_dir}")
        
    except Exception as e:
        log.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()

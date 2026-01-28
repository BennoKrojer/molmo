"""
Explore PixMo Points training data to understand what the model sees during training.

This script:
1. Loads examples exactly as they appear during training
2. Shows the preprocessed prompt and expected output
3. Visualizes images with red dots at the point coordinates
4. Saves everything as annotated images with text overlays

Usage:
    python scripts/explore_pixmo_points.py --num-examples 20 --output-dir explore_pixmo_points_output
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

from olmo.data.pixmo_datasets import PixMoPoints
from olmo.data.data_formatter import DataFormatter
from olmo.data.model_preprocessor import load_image
from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.util import resource_path


def draw_text_on_image(image, text, position="top", max_width=800, font_size=16):
    """Draw wrapped text on image with background."""
    draw = ImageDraw.Draw(image)
    
    # Try to load a better font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Wrap text to fit width
    wrapper = textwrap.TextWrapper(width=max_width // (font_size // 2))
    wrapped_lines = []
    for line in text.split('\n'):
        wrapped_lines.extend(wrapper.wrap(line))
    
    # Calculate text height
    line_height = font_size + 4
    text_height = len(wrapped_lines) * line_height + 20
    
    # Create background rectangle
    if position == "top":
        bg_y = 0
    else:  # bottom
        bg_y = image.height - text_height
    
    draw.rectangle([(0, bg_y), (image.width, bg_y + text_height)], fill=(0, 0, 0, 200))
    
    # Draw text
    y_offset = bg_y + 10
    for line in wrapped_lines:
        draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += line_height
    
    return image


def visualize_example(example_data, formatter, output_path, example_idx):
    """Visualize a single training example with annotations."""
    
    # Load image (it's a file path in the dataset)
    image_array = load_image(example_data['image'])
    image = Image.fromarray(image_array)
    
    # Create a copy for drawing
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Process each message in the example
    messages = example_data['message_list']
    
    all_text_info = []
    
    for msg_idx, msg in enumerate(messages):
        label = msg.get('label', 'unknown')
        points = msg.get('points', np.array([]))
        style = msg.get('style', 'unknown')
        point_scale = msg.get('point_scale', 1)
        
        # Generate prompt - sample from prompts used during training
        from olmo.data.data_formatter import GENERAL_PROMPTS_V1, apply_keyword_prompt
        rng = np.random.RandomState(42 + msg_idx)
        
        if style == "point_count":
            prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["point_count"], 
                                         {"label": label.lower()}, rng, dbg=False)
        elif style == "pointing":
            prompt = apply_keyword_prompt(GENERAL_PROMPTS_V1["pointing"], 
                                         {"label": label.lower()}, rng, dbg=False)
        else:
            prompt = f"Label: {label}"
        
        # Format output as it would be during training
        # Add image to msg for format_points to work correctly
        msg_with_image = msg.copy()
        msg_with_image['image'] = image_array
        output = formatter.format_points(msg_with_image)
        
        all_text_info.append(f"[Example {msg_idx+1}]")
        all_text_info.append(f"Label: {label}")
        all_text_info.append(f"Style: {style}")
        all_text_info.append(f"Num points: {len(points)}")
        all_text_info.append(f"Prompt: {prompt}")
        all_text_info.append(f"Expected Output: {output}")
        all_text_info.append("")
        
        # Draw points on image
        if len(points) > 0:
            h, w = image_array.shape[:2]
            for point in points:
                x, y = point
                # Convert from normalized coordinates (0-100 scale) to pixel coordinates
                px = int(x * w / 100)
                py = int(y * h / 100)
                
                # Draw a red dot
                radius = 8
                draw.ellipse([px - radius, py - radius, px + radius, py + radius], 
                           fill='red', outline='yellow', width=2)
                
                # Draw a small cross in the center for precision
                cross_size = 3
                draw.line([px - cross_size, py, px + cross_size, py], fill='yellow', width=1)
                draw.line([px, py - cross_size, px, py + cross_size], fill='yellow', width=1)
    
    # Add text summary at the bottom
    text_summary = "\n".join(all_text_info)
    
    # Create a figure with text and image
    # We'll create a larger canvas to fit both
    text_height = max(400, len(all_text_info) * 20)
    canvas = Image.new('RGB', (max(image.width, 1000), image.height + text_height), 'white')
    
    # Paste the annotated image
    canvas.paste(annotated_image, (0, 0))
    
    # Draw text at the bottom
    draw_canvas = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    y_offset = image.height + 10
    for line in all_text_info:
        draw_canvas.text((10, y_offset), line, fill='black', font=font)
        y_offset += 20
    
    # Save
    canvas.save(output_path)
    
    return all_text_info


def main():
    parser = argparse.ArgumentParser(description="Explore PixMo Points training data")
    parser.add_argument("--num-examples", type=int, default=20, 
                       help="Number of examples to visualize")
    parser.add_argument("--output-dir", type=str, default="explore_pixmo_points_output",
                       help="Directory to save visualizations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                       help="Which split to explore")
    parser.add_argument("--kind", type=str, default="basic", choices=["basic", "high_frequency"],
                       help="Which kind of pointing data")
    parser.add_argument("--counting", action="store_true",
                       help="Use counting mode (point_count) instead of pointing mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PixMo Points dataset...")
    print(f"  Split: {args.split}")
    print(f"  Kind: {args.kind}")
    print(f"  Counting mode: {args.counting}")
    
    # Load dataset exactly as training does
    dataset = PixMoPoints(split=args.split, kind=args.kind, counting=args.counting)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create data formatter to process examples
    formatter = DataFormatter()
    formatter.system_prompt = "demo_or_style"
    formatter.message_format = "none"
    formatter.prompt_templates = "v1"  # Use the template version that has point_count prompts
    
    # Sample random examples
    rng = np.random.RandomState(args.seed)
    
    # Collect statistics
    stats = {
        "total_messages": 0,
        "messages_with_objects": 0,
        "messages_without_objects": 0,
        "total_points": 0,
    }
    
    all_summaries = []
    
    print(f"\nProcessing {args.num_examples} examples...")
    
    for i in range(args.num_examples):
        # Get a random example
        idx = rng.randint(0, len(dataset))
        example_data = dataset.get(idx, rng)
        
        # Visualize and save
        output_path = output_dir / f"example_{i:03d}_idx{idx}.jpg"
        text_info = visualize_example(example_data, formatter, output_path, i)
        
        all_summaries.append({
            "example_idx": i,
            "dataset_idx": idx,
            "text_info": text_info,
            "image_path": str(output_path)
        })
        
        # Update statistics
        for msg in example_data['message_list']:
            stats["total_messages"] += 1
            points = msg.get('points', np.array([]))
            if len(points) == 0:
                stats["messages_without_objects"] += 1
            else:
                stats["messages_with_objects"] += 1
                stats["total_points"] += len(points)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{args.num_examples}...")
    
    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "statistics": stats,
            "examples": all_summaries
        }, f, indent=2)
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total messages processed: {stats['total_messages']}")
    print(f"Messages with objects: {stats['messages_with_objects']} ({100*stats['messages_with_objects']/stats['total_messages']:.1f}%)")
    print(f"Messages WITHOUT objects: {stats['messages_without_objects']} ({100*stats['messages_without_objects']/stats['total_messages']:.1f}%)")
    print(f"Total points across all examples: {stats['total_points']}")
    if stats['messages_with_objects'] > 0:
        print(f"Average points per message (when present): {stats['total_points']/stats['messages_with_objects']:.2f}")
    
    print(f"\n✓ Saved {args.num_examples} visualizations to {output_dir}/")
    print(f"✓ Saved summary to {summary_path}")


if __name__ == "__main__":
    main()


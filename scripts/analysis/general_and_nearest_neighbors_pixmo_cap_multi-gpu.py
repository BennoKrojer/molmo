"""This script takes the checkpoint we trained on the PixMo-Cap dataset and runs inference on the train and validation splits.
For each image, and each of the visual tokens in such an image:
a) we log the top5 nearest neighbor vocabularies from the LLM tokenizer
b) we check if any of the top5 nearest neighbors match any word from the ground truth caption (fuzzy matching)
c) we track token position statistics across all images

This script now supports both PixMo-Cap and Mosaic datasets via the --dataset argument.
This is the multi-GPU version that uses FSDP for model sharding to handle large models.

Usage: torchrun --nproc_per_node=2 scripts/toy_color_prediction_analyses/general_and_nearest_neighbors_pixmo_cap_multi-gpu.py --ckpt-path <path>
"""
import logging
import sys
import json
import re
from pathlib import Path
import argparse
import math

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

# Add translation library (robust to env/version issues)
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except Exception as e:
    TRANSLATION_AVAILABLE = False
    print(f"Warning: translation disabled (googletrans import error: {e}). To enable, install compatible versions, e.g.:\n"
          f"  pip install 'googletrans==4.0.0rc1' 'httpx<0.24' 'httpcore<1.0'")

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap, ColorMosaicDataset
from olmo.data.model_preprocessor import load_image, resize_and_pad, siglip_resize_and_pad
from olmo.torch_util import get_local_rank, get_world_size

log = logging.getLogger(__name__)

# Global translator instance
_translator = None
_translation_cache = {}
_cache_file = Path("translation_cache.json")
_cache_dirty = False
_cache_save_counter = 0

def load_translation_cache():
    """Load translation cache from file."""
    global _translation_cache
    if _cache_file.exists():
        try:
            with open(_cache_file, 'r', encoding='utf-8') as f:
                _translation_cache = json.load(f)
            print(f"Loaded {len(_translation_cache)} cached translations from {_cache_file}")
        except Exception as e:
            print(f"Could not load translation cache: {e}")
            _translation_cache = {}
    else:
        _translation_cache = {}

def save_translation_cache():
    """Save translation cache to file."""
    global _cache_dirty
    if _cache_dirty:
        try:
            with open(_cache_file, 'w', encoding='utf-8') as f:
                json.dump(_translation_cache, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(_translation_cache)} translations to cache")
            _cache_dirty = False
        except Exception as e:
            print(f"Could not save translation cache: {e}")

def get_translator():
    """Get or create a global translator instance."""
    global _translator
    if _translator is None and TRANSLATION_AVAILABLE:
        _translator = Translator()
        # Load cache when we first create the translator
        load_translation_cache()
    return _translator

def contains_chinese(text):
    """Check if text contains Chinese characters."""
    if not text:
        return False
    # Check for Chinese characters (CJK Unified Ideographs)
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def translate_chinese_token(token_text):
    """Translate Chinese tokens to English, return list of possible translations."""
    global _cache_dirty, _cache_save_counter
    
    if not TRANSLATION_AVAILABLE or not contains_chinese(token_text):
        return [token_text]
    
    # Check cache first
    if token_text in _translation_cache:
        cached_result = _translation_cache[token_text]
        if cached_result != token_text:  # If we have a translation
            return [token_text, cached_result]
        else:
            return [token_text]
    
    translator = get_translator()
    if translator is None:
        return [token_text]
    
    try:
        # Clean the token text
        clean_text = token_text.strip()
        if not clean_text:
            return [token_text]
        
        # Translate to English - use 'zh-cn' instead of 'zh' for Chinese
        result = translator.translate(clean_text, src='zh-cn', dest='en')
        if result and result.text:
            translated = result.text.lower().strip()
            
            # Cache the result
            _translation_cache[token_text] = translated
            _cache_dirty = True
            _cache_save_counter += 1
            
            # Save cache every 100 translations
            if _cache_save_counter >= 100:
                save_translation_cache()
                _cache_save_counter = 0
            
            # Return both original and translated versions
            if translated != clean_text.lower():
                print(f"Translated '{token_text}' -> '{translated}'")
                return [token_text, translated]
            else:
                return [token_text]
        else:
            print(f"Translation failed for '{token_text}': No result")
            # Cache the failure (store original token)
            _translation_cache[token_text] = token_text
            _cache_dirty = True
            return [token_text]
    except Exception as e:
        print(f"Translation failed for '{token_text}': {e}")
        # Cache the failure (store original token)
        _translation_cache[token_text] = token_text
        _cache_dirty = True
        return [token_text]

def decode_token(tokenizer, idx):
    """Decode a token and ensure it's a proper Unicode string."""
    token = tokenizer.decode([int(idx)])
    # Convert to actual characters by encoding and decoding through utf-8
    return token.encode('utf-8').decode('utf-8')

def clear_gpu_memory():
    """Clear CUDA cache to free up memory."""
    import gc
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Force another garbage collection after CUDA cleanup
    gc.collect()

def normalize_text_for_matching(text):
    """Normalize text for fuzzy matching: lowercase, remove punctuation, split by whitespace, filter stopwords."""
    # Define common stopwords to exclude from matching
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 
        'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'the', 'this', 'these', 
        'those', 'they', 'them', 'their', 'there', 'then', 'than', 'but', 'or', 'so', 'if', 'when', 
        'where', 'why', 'how', 'what', 'who', 'which', 'can', 'could', 'would', 'should', 'may', 'might'
    }
    
    # Define visual/task words that indicate interface or visual elements
    visual_task_words = {
        'image', 'photo', 'picture', 'img', 'display', 'frame', 'screen', 'view', 'window', 'panel', 'icon', 'cursor', 'pixel', 'resolution', 'monitor', 'camera',
        'lens', 'focus', 'zoom', 'crop', 'filter', 'edit', 'save', 'load', 'file', 'folder', 'document',
        'page', 'layout', 'design', 'graphic', 'visual', 'render', 'preview', 'thumbnail', 'gallery',
        'album', 'slide', 'presentation', 'video', 'clip', 'frame', 'shot', 'capture', 'record'
    }
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters, but preserve Unicode letters
    # This regex keeps Unicode letters (\w), digits, and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split by whitespace and filter out empty strings and stopwords
    words = [word.strip() for word in text.split() 
             if word.strip() and word.strip() not in stopwords and len(word.strip()) > 1]
    return words, visual_task_words

def check_match_type(token_words, caption_words, visual_task_words):
    """Check what type of match this is: interpretable, visual_task, or none.
    
    Returns:
        tuple: (match_type, matches_list)
        match_type: 'interpretable', 'visual_task', or 'none'
        matches_list: list of match dictionaries
    """
    matches = []
    has_interpretable = False
    has_visual_task = False
    
    for token_word in token_words:
        # Check for visual/task matches first
        if token_word in visual_task_words:
            matches.append({
                "token_word": token_word,
                "match_type": "visual_task",
                "matched_word": token_word
            })
            has_visual_task = True
        
        # Check for interpretable matches with caption
        for caption_word in caption_words:
            if token_word == caption_word:
                matches.append({
                    "token_word": token_word,
                    "match_type": "interpretable", 
                    "matched_word": caption_word
                })
                has_interpretable = True
    
    # Determine overall match type (visual_task takes precedence)
    if has_visual_task:
        match_type = 'visual_task'
    elif has_interpretable:
        match_type = 'interpretable'
    else:
        match_type = 'none'
    
    return match_type, matches

def check_match_type_with_translation(token_text, caption_words, visual_task_words):
    """Enhanced matching that includes Chinese translation support.
    
    Args:
        token_text: Original token text (may contain Chinese)
        caption_words: List of normalized words from the caption
        visual_task_words: Set of visual/task interface words
    
    Returns:
        tuple: (match_type, matches_list)
        match_type: 'interpretable', 'visual_task', or 'none'
        matches_list: list of match dictionaries
    """
    # Get all possible translations of the token
    token_translations = translate_chinese_token(token_text)
    
    # Normalize each translation
    all_token_words = []
    for translation in token_translations:
        normalized_words, _ = normalize_text_for_matching(translation)
        all_token_words.extend(normalized_words)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_token_words = []
    for word in all_token_words:
        if word not in seen:
            seen.add(word)
            unique_token_words.append(word)
    
    # Use the existing check_match_type function
    return check_match_type(unique_token_words, caption_words, visual_task_words)

def check_color_match_with_translation(token_text, ground_truth_color):
    """Check if a token matches the ground truth color, including Chinese translation support.
    
    Args:
        token_text: Original token text (may contain Chinese)
        ground_truth_color: Ground truth color word (in English)
    
    Returns:
        tuple: (is_match, match_details)
        is_match: boolean indicating if there's a match
        match_details: dict with match information if found
    """
    # Get all possible translations of the token
    token_translations = translate_chinese_token(token_text)
    
    # Normalize ground truth color
    gt_color_norm = ground_truth_color.strip().lower()
    
    # Check each translation
    for translation in token_translations:
        token_norm = translation.strip().lower()
        
        # Skip empty tokens to prevent false matches (empty string matches any substring)
        if not token_norm:
            continue
        
        # Check for exact match or substring match
        if token_norm == gt_color_norm or gt_color_norm in token_norm or token_norm in gt_color_norm:
            return True, {
                "token": get_display_token(token_text),
                "token_normalized": token_norm,
                "ground_truth_color": ground_truth_color,
                "match_type": "exact" if token_norm == gt_color_norm else "substring"
            }
    
    return False, None

def patch_idx_to_row_col(patch_idx, patches_per_chunk):
    """Convert patch index to row and column coordinates.
    
    Args:
        patch_idx: Index of the patch (0-based)
        patches_per_chunk: Total number of patches (e.g., 144 for 12x12, 576 for 24x24)
    
    Returns:
        tuple: (row, col) where row and col are 0-based coordinates
    
    Example:
        For 24x24 grid (576 patches): patch_idx=27 -> row=1, col=3
        For 12x12 grid (144 patches): patch_idx=27 -> row=2, col=3
    """
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col

def save_image_with_token_grid(pil_image, save_path, patches_per_chunk, preprocessor=None, interpretable_patches=None):
    """Save both normal image and preprocessed image with grid overlay.
    
    Args:
        pil_image: PIL Image object
        save_path: Path to save the image (without extension)
        patches_per_chunk: Number of patches (e.g., 144 for 12x12, 576 for 24x24)
        preprocessor: Optional preprocessor to apply model preprocessing
        interpretable_patches: Optional set of patch indices that are interpretable
    """
    # Calculate grid dimensions
    grid_size = int(math.sqrt(patches_per_chunk))
    if grid_size * grid_size != patches_per_chunk:
        log.warning(f"patches_per_chunk ({patches_per_chunk}) is not a perfect square, using closest grid")
        grid_size = int(math.sqrt(patches_per_chunk))
    
    # Save the normal image (without grid)
    normal_path = Path(str(save_path).replace('.jpg', '.jpg'))
    pil_image.save(normal_path, "JPEG", quality=95)
    log.info(f"Saved normal image to {normal_path}")
    
    # Prepare preprocessed image if preprocessor is provided
    if preprocessor is not None:
        try:
            # Convert PIL image to numpy array for preprocessing
            image_array = load_image(pil_image)
            
            # Apply the same preprocessing as the model
            processed_image, img_mask = preprocessor.mm_preprocessor.resize_image(
                image_array, 
                (512, 512), 
                is_training=False,
                rng=np.random
            )
            
            # Convert back to PIL image (processed_image is in [0,1] range)
            processed_image = (processed_image * 255).astype(np.uint8)
            processed_pil = Image.fromarray(processed_image)
        except Exception as e:
            log.warning(f"Could not preprocess image: {e}, using original image")
            processed_pil = pil_image
    else:
        processed_pil = pil_image
    
    # Create grid overlay on preprocessed image
    img = processed_pil.copy()
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    img_width, img_height = img.size
    
    # Calculate margins for axes (proportional to image size)
    margin_size = max(50, min(img_width, img_height) // 20)
    
    # Create a new image with margins for axes
    new_width = img_width + margin_size
    new_height = img_height + margin_size
    new_img = Image.new('RGB', (new_width, new_height), 'white')
    
    # Paste the processed image with offset for margins
    new_img.paste(img, (margin_size, margin_size))
    
    # Create new draw object for the enlarged image
    draw = ImageDraw.Draw(new_img)
    
    # Try to load a font, fall back to default if not available
    try:
        # Calculate font size based on margin
        font_size = max(8, margin_size // 3)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        try:
            font_size = max(8, margin_size // 3)
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw grid lines with different colors for interpretable vs non-interpretable patches
    default_grid_color = 'red'
    interpretable_grid_color = 'green'
    line_width = max(1, margin_size // 25)
    
    # Draw cell backgrounds for interpretable patches
    if interpretable_patches:
        for patch_idx in interpretable_patches:
            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
            if row < grid_size and col < grid_size:  # Make sure we're within bounds
                # Calculate cell boundaries
                x1 = margin_size + (col * img_width) // grid_size
                y1 = margin_size + (row * img_height) // grid_size
                x2 = margin_size + ((col + 1) * img_width) // grid_size
                y2 = margin_size + ((row + 1) * img_height) // grid_size
                
                # Draw green border for interpretable cell
                # Top border
                draw.line([(x1, y1), (x2, y1)], fill=interpretable_grid_color, width=line_width + 1)
                # Bottom border
                draw.line([(x1, y2), (x2, y2)], fill=interpretable_grid_color, width=line_width + 1)
                # Left border
                draw.line([(x1, y1), (x1, y2)], fill=interpretable_grid_color, width=line_width + 1)
                # Right border
                draw.line([(x2, y1), (x2, y2)], fill=interpretable_grid_color, width=line_width + 1)
    
    # Draw the main grid lines (red for non-interpretable areas)
    # Vertical lines
    for i in range(grid_size + 1):
        x = margin_size + (i * img_width) // grid_size
        draw.line([(x, margin_size), (x, margin_size + img_height)], fill=default_grid_color, width=line_width)
    
    # Horizontal lines
    for i in range(grid_size + 1):
        y = margin_size + (i * img_height) // grid_size
        draw.line([(margin_size, y), (margin_size + img_width, y)], fill=default_grid_color, width=line_width)
    
    # Redraw green borders on top of red grid for interpretable patches
    if interpretable_patches:
        for patch_idx in interpretable_patches:
            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
            if row < grid_size and col < grid_size:  # Make sure we're within bounds
                # Calculate cell boundaries
                x1 = margin_size + (col * img_width) // grid_size
                y1 = margin_size + (row * img_height) // grid_size
                x2 = margin_size + ((col + 1) * img_width) // grid_size
                y2 = margin_size + ((row + 1) * img_height) // grid_size
                
                # Draw green border for interpretable cell (thicker)
                # Top border
                draw.line([(x1, y1), (x2, y1)], fill=interpretable_grid_color, width=line_width + 2)
                # Bottom border
                draw.line([(x1, y2), (x2, y2)], fill=interpretable_grid_color, width=line_width + 2)
                # Left border
                draw.line([(x1, y1), (x1, y2)], fill=interpretable_grid_color, width=line_width + 2)
                # Right border
                draw.line([(x2, y1), (x2, y2)], fill=interpretable_grid_color, width=line_width + 2)
    
    # Draw axis numbers
    text_color = 'black'
    
    # Top axis (column numbers)
    for i in range(grid_size):
        x = margin_size + (i * img_width) // grid_size + (img_width // grid_size) // 2
        y = margin_size // 3
        text = str(i)
        if font:
            # Get text size for centering
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width//2, y - text_height//2), text, fill=text_color, font=font)
        else:
            draw.text((x-5, y-5), text, fill=text_color)
    
    # Left axis (row numbers)
    for i in range(grid_size):
        x = margin_size // 3
        y = margin_size + (i * img_height) // grid_size + (img_height // grid_size) // 2
        text = str(i)
        if font:
            # Get text size for centering
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width//2, y - text_height//2), text, fill=text_color, font=font)
        else:
            draw.text((x-5, y-5), text, fill=text_color)
    
    # Save the image with grid (preprocessed version)
    grid_path = Path(str(save_path).replace('.jpg', '_grid.jpg'))
    new_img.save(grid_path, "JPEG", quality=95)
    log.info(f"Saved preprocessed image with {grid_size}x{grid_size} token grid to {grid_path}")

def create_interpretability_plot(overall_statistics, output_path):
    """Create a bar plot showing overall interpretability statistics."""
    if not overall_statistics:
        log.warning("No statistics available for plotting")
        return
    
    # Create overall interpretability plot
    interpretable = overall_statistics.get("interpretable_visual_tokens", 0)
    total = overall_statistics.get("total_visual_tokens", 1)
    non_interpretable = total - interpretable
    
    plt.figure(figsize=(8, 6))
    categories = ['Interpretable', 'Non-interpretable']
    values = [interpretable, non_interpretable]
    colors = ['steelblue', 'lightcoral']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    
    # Add percentage labels on bars
    for bar, value in zip(bars, values):
        percentage = (value / total * 100) if total > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                f'{value}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Number of Visual Tokens')
    plt.title(f'Visual Token Interpretability Analysis\n(Total: {total} tokens)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Interpretability plot saved to {output_path}")

def create_token_position_plot(token_position_stats, output_path, top_n=20):
    """Create a bar plot showing top token positions by interpretability percentage."""
    if not token_position_stats:
        log.warning("No token position statistics available for plotting")
        return
    
    # Determine patches_per_chunk from the available patch indices
    max_patch_idx = max(token_position_stats.keys()) if token_position_stats else 0
    # Estimate grid size - assume it's a perfect square
    # Add 1 because patch indices are 0-based
    estimated_total_patches = max_patch_idx + 1
    estimated_grid_size = int(math.sqrt(estimated_total_patches))
    
    # Check if it's likely a perfect square grid (144 or 576 are common)
    if estimated_grid_size * estimated_grid_size == estimated_total_patches:
        patches_per_chunk = estimated_total_patches
    elif estimated_total_patches <= 144:
        patches_per_chunk = 144  # 12x12 grid
    elif estimated_total_patches <= 576:
        patches_per_chunk = 576  # 24x24 grid
    else:
        # For larger grids, find the next perfect square
        estimated_grid_size = int(math.ceil(math.sqrt(estimated_total_patches)))
        patches_per_chunk = estimated_grid_size * estimated_grid_size
    
    # Sort by interpretability percentage and take top N
    sorted_positions = sorted(token_position_stats.items(), 
                            key=lambda x: x[1].get("interpretability_percentage", 0), 
                            reverse=True)[:top_n]
    
    # Create labels with patch_idx and row/col information
    position_labels = []
    for pos, _ in sorted_positions:
        row, col = patch_idx_to_row_col(pos, patches_per_chunk)
        position_labels.append(f"{pos}\n(r{row},c{col})")
    
    percentages = [stats.get("interpretability_percentage", 0) for _, stats in sorted_positions]
    
    plt.figure(figsize=(15, 6))  # Made wider to accommodate longer labels
    bars = plt.bar(range(len(position_labels)), percentages, color='forestgreen', alpha=0.7)
    
    plt.xlabel('Token Position (patch_idx and row,col coordinates)')
    plt.ylabel('Interpretability Percentage (%)')
    plt.title(f'Top {top_n} Token Positions by Interpretability')
    plt.xticks(range(len(position_labels)), position_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Token position plot saved to {output_path}")

def create_color_interpretability_plot(color_statistics, output_path):
    """Create a bar plot showing overall color interpretability statistics."""
    if not color_statistics:
        log.warning("No color statistics available for plotting")
        return
    
    # Create overall color interpretability plot
    total_samples = sum(stats["total_samples"] for stats in color_statistics)
    ground_truth_matches = sum(stats["ground_truth_matches"] for stats in color_statistics)
    non_interpretable = total_samples - ground_truth_matches
    
    plt.figure(figsize=(10, 6))
    categories = ['Interpretable', 'Non-interpretable']
    values = [ground_truth_matches, non_interpretable]
    colors = ['steelblue', 'lightcoral']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    
    # Add percentage labels on bars
    for bar, value in zip(bars, values):
        percentage = (value / total_samples * 100) if total_samples > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_samples*0.01, 
                f'{value}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Number of Color Tokens')
    plt.title(f'Color Token Interpretability Analysis\n(Total: {total_samples} tokens)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Color interpretability plot saved to {output_path}")

def is_already_translated(token_text):
    """Check if a token has already been translated (contains 'translated from' pattern)."""
    return " (translated from " in token_text

def get_display_token(original_token):
    """Get display version of token, showing translation with original Chinese in brackets."""
    # If already translated, return as-is
    if is_already_translated(original_token):
        return original_token
        
    if not TRANSLATION_AVAILABLE or not contains_chinese(original_token):
        return original_token
    
    # Check cache first
    if original_token in _translation_cache:
        cached_result = _translation_cache[original_token]
        if cached_result != original_token:  # If we have a translation
            return f"{cached_result} (translated from {original_token})"
        else:
            return original_token
    
    # If not in cache, try to translate and cache it
    translator = get_translator()
    if translator is None:
        return original_token
    
    try:
        clean_text = original_token.strip()
        if not clean_text:
            return original_token
        
        result = translator.translate(clean_text, src='zh-cn', dest='en')
        if result and result.text:
            translated = result.text.lower().strip()
            
            # Cache the result
            _translation_cache[original_token] = translated
            global _cache_dirty, _cache_save_counter
            _cache_dirty = True
            _cache_save_counter += 1
            
            # Save cache every 100 translations
            if _cache_save_counter >= 100:
                save_translation_cache()
                _cache_save_counter = 0
            
            if translated != clean_text.lower():
                print(f"Translated '{original_token}' -> '{translated}'")
                return f"{translated} (translated from {original_token})"
            else:
                return original_token
        else:
            # Cache the failure
            _translation_cache[original_token] = original_token
            _cache_dirty = True
            return original_token
    except Exception as e:
        print(f"Translation failed for '{original_token}': {e}")
        # Cache the failure
        _translation_cache[original_token] = original_token
        _cache_dirty = True
        return original_token

def process_split(model, preprocessor, dataset, num_images, prompt, use_n_token_only, images_dir=None, split_name="", dataset_type="pixmo_cap", llm_layer: int = 0, skip_generate: bool = False, sanity_check_l0_l1: bool = False, sanity_threshold: float = 0.9):
    """Process a dataset split and return results with distributed processing."""
    # Distribute images across processes
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    images_per_process = num_images // world_size
    start_idx = local_rank * images_per_process
    end_idx = start_idx + images_per_process
    
    # Handle remainder for last process
    if local_rank == world_size - 1:
        end_idx = num_images
    
    if local_rank == 0:
        print(f"Process {local_rank}: Processing images {start_idx} to {end_idx-1}")
    
    split_results = []
    
    # Initialize statistics tracking
    total_visual_tokens = 0
    interpretable_visual_tokens = 0
    visual_task_visual_tokens = 0
    token_position_stats = {}
    
    # For mosaic dataset, also track color statistics
    color_statistics = {} if dataset_type == "mosaic" else None
    
    # Process assigned images for this process
    for i in tqdm(range(start_idx, end_idx), desc=f"Rank {local_rank}"):
        example_data = dataset.get(i, np.random)
        
        # Extract ground truth based on dataset type
        if dataset_type == "pixmo_cap":
            # Extract ground truth caption
            caption_text = ""
            if "message_list" in example_data and len(example_data["message_list"]) > 0:
                message = example_data["message_list"][0]
                caption_text = message.get("text", "")
            ground_truth = caption_text
        else:  # mosaic
            # Extract color sequence
            color_sequence = example_data["metadata"]["color_sequence"]
            ground_truth = color_sequence
        
        # Save image if images_dir is provided (only on main process)
        image_filename = None
        image_filename_grid = None
        temp_pil_image = None
        
        if images_dir is not None and local_rank == 0:
            try:
                image_path = example_data["image"]
                # Load the image from file path using PIL, just like the preprocessor does
                with Image.open(image_path) as pil_image:
                    # Convert to RGB to ensure compatibility
                    pil_image = pil_image.convert("RGB")
                    temp_pil_image = pil_image  # Store for later use
                    
            except Exception as e:
                log.warning(f"Could not load image {i}: {e}")
                temp_pil_image = None
        
        # Prepare matching data based on dataset type
        if dataset_type == "pixmo_cap":
            # Normalize caption words for matching
            caption_words, visual_task_words = normalize_text_for_matching(caption_text)
        else:  # mosaic
            caption_words, visual_task_words = None, None
        
        # Create example with the provided prompt
        if dataset_type == "pixmo_cap":
            example = {
                "image": example_data["image"],
                "messages": [prompt]
            }
        else:  # mosaic
            example = {
                "image": example_data["image"],
                "messages": {
                    "messages": [prompt],
                    "style": "none"
                }
            }

        # Preprocess example
        batch = preprocessor(example, rng=np.random)

        # Initialize image results
        if dataset_type == "pixmo_cap":
            image_results = {
                "image_idx": i,
                "ground_truth_caption": caption_text,
                "chunks": []
            }
        else:  # mosaic
            image_results = {
                "image_idx": i,
                "true_color_sequence": color_sequence,
                "chunks": []
            }

        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                device = torch.device(f"cuda:{local_rank}")
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).to(device)
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).to(device) if batch.get("image_masks") is not None else None
                
                # Select which features to use for analysis:
                #  - llm_layer == 0: use vision backbone projected features (existing behavior)
                #  - llm_layer > 0: use hidden states from the specified LLM layer at visual token positions
                if llm_layer == 0 and not sanity_check_l0_l1:
                    image_features, tokens_before_MLP = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                    # Handle use_n_token_only properly
                    if type(use_n_token_only) == int and use_n_token_only != -1:
                        image_features = image_features[:, :, :use_n_token_only, :]
                    elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                        image_features = image_features[:, :, use_n_token_only, :]
                else:
                    # Forward through the LLM to get hidden states
                    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    image_input_idx_tensor = torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None
                    output = model(
                        input_ids=input_ids,
                        images=images_tensor,
                        image_masks=image_masks_tensor,
                        image_input_idx=image_input_idx_tensor,
                        output_hidden_states=True,
                        last_logits_only=False,
                    )
                    hidden_states = output.hidden_states  # Tuple[Tensor]
                    # Bound-check layer index against available hidden states
                    max_layer_index = len(hidden_states) - 1  # includes final ln_f state
                    assert image_input_idx_tensor is not None, "image_input_idx is required to extract visual tokens from LLM layers"
                    # Helper to gather features from a specific layer index
                    def gather_visual_features_from_layer(hs_tensor):
                        B = hs_tensor.shape[0]
                        num_chunks_local = image_input_idx_tensor.shape[1]
                        patches_per_chunk_local = image_input_idx_tensor.shape[2]
                        d_model_local = hs_tensor.shape[-1]
                        feats = torch.zeros((B, num_chunks_local, patches_per_chunk_local, d_model_local), device=hs_tensor.device, dtype=hs_tensor.dtype)
                        flat_positions_local = image_input_idx_tensor.view(B, -1)
                        valid_mask_local = flat_positions_local >= 0
                        for b in range(B):
                            valid_pos = flat_positions_local[b][valid_mask_local[b]]
                            if valid_pos.numel() == 0:
                                continue
                            gathered = hs_tensor[b, valid_pos.long(), :]
                            feats.view(B, -1, d_model_local)[b, valid_mask_local[b], :] = gathered
                        return feats

                    image_features_l0_for_sanity = None
                    if sanity_check_l0_l1:
                        # Compare layer 0 vs layer 1 visual token embeddings
                        layer0_index = 0
                        layer1_index = 1 if max_layer_index >= 1 else max_layer_index
                        hs0 = hidden_states[layer0_index]
                        hs1 = hidden_states[layer1_index]
                        image_features_l0 = gather_visual_features_from_layer(hs0)
                        image_features_l1 = gather_visual_features_from_layer(hs1)
                        image_features_l0_for_sanity = image_features_l0
                        # Compute cosine similarity over valid positions
                        B, num_chunks, patches_per_chunk, D = image_features_l0.shape
                        v0 = image_features_l0.view(B, -1, D)
                        v1 = image_features_l1.view(B, -1, D)
                        valid = image_input_idx_tensor.view(B, -1) >= 0
                        # Avoid zero vectors by normalizing and masking
                        cosims = []
                        for b in range(B):
                            if valid[b].any():
                                a = v0[b][valid[b]]
                                c = v1[b][valid[b]]
                                a_n = torch.nn.functional.normalize(a, dim=-1)
                                c_n = torch.nn.functional.normalize(c, dim=-1)
                                cos = (a_n * c_n).sum(dim=-1)
                                cosims.append(cos.detach().float().cpu())
                        if len(cosims) > 0:
                            cosims = torch.cat(cosims)
                            mean_cos = float(cosims.mean().item())
                            frac_high = float((cosims >= sanity_threshold).float().mean().item())
                            print(f"[Sanity l0 vs l1] mean_cos={mean_cos:.4f}, frac>= {sanity_threshold} : {frac_high:.4f} (N={cosims.numel()})")
                        # For downstream NN analysis, pick the requested layer (default 1 when sanity flag is used in your run)
                        layer_index = min(max(1, llm_layer), max_layer_index)
                        hs = hidden_states[layer_index]
                        image_features = gather_visual_features_from_layer(hs)
                    else:
                        layer_index = min(llm_layer, max_layer_index)
                        hs = hidden_states[layer_index]
                        image_features = gather_visual_features_from_layer(hs)
                image_results["feature_shape"] = list(image_features.shape)
                image_results["llm_layer_used"] = llm_layer
                
                # Get token embeddings from the model
                # Instead of trying to access sharded embeddings, use cached version
                model_identifier = model.config.tokenizer.identifier
                if "qwen" in model_identifier.lower():
                    cached_embeddings_path = "analysis_results/cached_text_embeddings/Qwen_Qwen2-7B/layer_0_static_vocab.npy"
                elif "dolma" in model_identifier.lower():
                    cached_embeddings_path = "analysis_results/cached_text_embeddings/allenai_OLMo-7B-1024-preview/layer_0_static_vocab.npy"
                elif "llama" in model_identifier.lower():
                    cached_embeddings_path = "analysis_results/cached_text_embeddings/meta-llama_Meta-Llama-3-8B/layer_0_static_vocab.npy"
                
                try:
                    token_embeddings = torch.from_numpy(np.load(cached_embeddings_path)).to(device)
                    if local_rank == 0:
                        print(f"Loaded cached token embeddings from {cached_embeddings_path}, shape: {token_embeddings.shape}")
                except FileNotFoundError:
                    if local_rank == 0:
                        print(f"Cached embeddings not found at {cached_embeddings_path}, trying to access from model...")
                    # Fallback to model access (this might fail with FSDP)
                    token_embeddings = model.transformer.wte.embedding.weight
                
                # Reshape image features to combine batch and chunks dimensions
                batch_size, num_chunks, patches_per_chunk, hidden_dim = image_features.shape
                image_features_reshaped = image_features.view(-1, patches_per_chunk, hidden_dim)
                
                # Normalize the embeddings for cosine similarity
                image_features_norm = torch.nn.functional.normalize(image_features_reshaped, dim=-1)
                token_embeddings_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)
                
                # Compute cosine similarity for each patch
                similarity = torch.matmul(image_features_norm, token_embeddings_norm.T)
                
                # Get top-5 most similar tokens for each patch
                top_k = 5
                top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)

                # If sanity enabled, also compute top-5 for layer 0 and print overlap
                if sanity_check_l0_l1 and 'image_features_l0_for_sanity' in locals() and image_features_l0_for_sanity is not None:
                    feats0_reshaped = image_features_l0_for_sanity.view(-1, patches_per_chunk, hidden_dim)
                    feats0_norm = torch.nn.functional.normalize(feats0_reshaped, dim=-1)
                    sim0 = torch.matmul(feats0_norm, token_embeddings_norm.T)
                    _, top_indices_l0 = torch.topk(sim0, k=top_k, dim=-1)
                    # Compute per-patch overlap
                    A = top_indices.view(-1, top_k)
                    Bti = top_indices_l0.view(-1, top_k)
                    valid_mask_flat = (torch.tensor(batch.get("image_input_idx")).view(1, num_chunks, patches_per_chunk) >= 0).reshape(-1).to(A.device)
                    A_valid = A[valid_mask_flat]
                    B_valid = Bti[valid_mask_flat]
                    if A_valid.numel() > 0:
                        eq = (A_valid.unsqueeze(-1) == B_valid.unsqueeze(-2))
                        overlap_counts = eq.any(dim=-1).sum(dim=-1).float()
                        mean_frac = float((overlap_counts / top_k).mean().item())
                        frac_all5 = float((overlap_counts == top_k).float().mean().item())
                        print(f"[Sanity top5 overlap l{llm_layer} vs l0] mean_frac={mean_frac:.3f}, frac_all5={frac_all5:.3f}, N={overlap_counts.numel()}")
                
                # Clear intermediate tensors
                del similarity, image_features_norm, token_embeddings_norm
                clear_gpu_memory()

                # generated output for reference (optional)
                if not skip_generate:
                    input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
                    output = model.generate(
                        input_ids=input_ids,
                        images=images_tensor,
                        image_masks=image_masks_tensor,
                        image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).to(device) if batch.get("image_input_idx") is not None else None,
                        max_steps=600 if dataset_type == "mosaic" else 200,  # More steps for mosaic (multiple color tokens)
                        is_distributed=False
                    )
                    token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                    decoded = preprocessor.tokenizer.decode(token_ids[0])
                    image_results["generated_response"] = decoded
                else:
                    image_results["generated_response"] = None

                # Clear GPU tensors
                if not skip_generate:
                    del input_ids, output
                del images_tensor, image_masks_tensor, image_features
                clear_gpu_memory()

                # Add ground truth tokens for mosaic
                if dataset_type == "mosaic":
                    ground_truth_text = " ".join(color_sequence)
                    ground_truth_tokens = preprocessor.tokenizer.encode(ground_truth_text)
                    ground_truth_token_strings = [decode_token(preprocessor.tokenizer, token_id) for token_id in ground_truth_tokens]
                    image_results["ground_truth_tokens"] = ground_truth_token_strings

                # Now save the image with grid overlay if we have it (only on main process)
                if temp_pil_image is not None and images_dir is not None and local_rank == 0:
                    try:
                        # Collect interpretable patches for this image
                        interpretable_patches = set()
                        for chunk_idx in range(num_chunks):
                            for patch_idx in range(patches_per_chunk):
                                # Check if this patch will be interpretable based on the matches we found
                                patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                                patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                                patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]
                                
                                # Check for matches based on dataset type
                                if dataset_type == "pixmo_cap":
                                    # Check for caption matches (same logic as below)
                                    caption_match = False
                                    for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                                        match_type, matches = check_match_type_with_translation(token_text, caption_words, visual_task_words)
                                        if match_type == 'interpretable':
                                            caption_match = True
                                            break
                                    if caption_match:
                                        interpretable_patches.add(patch_idx)
                                else:  # mosaic
                                    # Check for color matches
                                    ground_truth_match = False
                                    if patch_idx < len(color_sequence):
                                        corresponding_color = color_sequence[patch_idx].strip().lower()
                                        for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                                            is_match, match_detail = check_color_match_with_translation(token_text, corresponding_color)
                                            if is_match:
                                                ground_truth_match = True
                                                break
                                    if ground_truth_match:
                                        interpretable_patches.add(patch_idx)
                        
                        image_filename_base = f"{split_name}_image_{i:04d}"
                        image_save_path = images_dir / f"{image_filename_base}.jpg"
                        save_image_with_token_grid(temp_pil_image, image_save_path, patches_per_chunk, preprocessor, interpretable_patches)
                        image_filename = f"{image_filename_base}.jpg"
                        image_filename_grid = f"{image_filename_base}_grid.jpg"
                    except Exception as e:
                        log.warning(f"Could not save image {i} with grid: {e}")
                        image_filename = None
                        image_filename_grid = None

                # Add image filenames if saved
                if image_filename:
                    image_results["image_filename"] = image_filename
                if image_filename_grid:
                    image_results["image_filename_grid"] = image_filename_grid

                # Store results for each chunk
                for chunk_idx in range(num_chunks):
                    chunk_results = {
                        "chunk_name": "Full Image" if chunk_idx == 0 else f"Chunk {chunk_idx}",
                        "patches": []
                    }
                    
                    for patch_idx in range(patches_per_chunk):
                        patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                        patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]
                        
                        # Initialize token position stats if not exists
                        if patch_idx not in token_position_stats:
                            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                            token_position_stats[patch_idx] = {
                                "total_occurrences": 0,
                                "interpretable_occurrences": 0,
                                "visual_task_occurrences": 0,
                                "interpretability_percentage": 0.0,
                                "visual_task_percentage": 0.0,
                                "patch_row": row,
                                "patch_col": col
                            }
                        
                        token_position_stats[patch_idx]["total_occurrences"] += 1
                        
                        if dataset_type == "pixmo_cap":
                            # Check for matches using three-category system
                            overall_match_type = 'none'
                            all_matches = []
                            
                            for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                                # Check what type of match this token has (with translation support)
                                match_type, matches = check_match_type_with_translation(token_text, caption_words, visual_task_words)
                                
                                if match_type != 'none':
                                    # Update overall match type (visual_task takes precedence over interpretable)
                                    if match_type == 'visual_task' or (overall_match_type != 'visual_task' and match_type == 'interpretable'):
                                        overall_match_type = match_type
                                    
                                    # Add detailed match info
                                    for match in matches:
                                        all_matches.append({
                                            "token": get_display_token(token_text),
                                            "token_word": match["token_word"],
                                            "matched_word": match["matched_word"],
                                            "match_type": match["match_type"],
                                            "similarity": float(similarity_score)
                                        })
                            
                            # Update statistics
                            total_visual_tokens += 1
                            if overall_match_type == 'interpretable':
                                interpretable_visual_tokens += 1
                                token_position_stats[patch_idx]["interpretable_occurrences"] += 1
                            elif overall_match_type == 'visual_task':
                                visual_task_visual_tokens += 1
                                token_position_stats[patch_idx]["visual_task_occurrences"] += 1
                            
                            # Add row/col information based on patch_idx
                            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                            
                            patch_results = {
                                "patch_idx": patch_idx,
                                "patch_row": row,
                                "patch_col": col,
                                "nearest_neighbors": [
                                    {"token": get_display_token(token), "similarity": float(sim)}
                                    for token, sim in zip(patch_tokens, patch_values)
                                ],
                                "match_type": overall_match_type,
                                "caption_match": overall_match_type == 'interpretable',
                                "visual_task_match": overall_match_type == 'visual_task'
                            }
                            
                            # Add detailed matches if any were found
                            if all_matches:
                                patch_results["matches"] = all_matches
                        
                        else:  # mosaic
                            # Check if this patch matches its corresponding ground truth color position
                            ground_truth_match = False
                            corresponding_color = None
                            match_details = []
                            
                            if patch_idx < len(color_sequence):
                                corresponding_color = color_sequence[patch_idx].strip().lower()
                                
                                # Check each of the top-5 nearest neighbor tokens for matches
                                for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                                    is_match, match_detail = check_color_match_with_translation(token_text, corresponding_color)
                                    if is_match:
                                        ground_truth_match = True
                                        match_detail["similarity"] = float(similarity_score)
                                        match_detail["rank"] = j + 1  # 1-based rank
                                        match_details.append(match_detail)
                            
                            # Update statistics
                            total_visual_tokens += 1
                            if ground_truth_match:
                                interpretable_visual_tokens += 1
                                token_position_stats[patch_idx]["interpretable_occurrences"] += 1
                            
                            # Update color statistics if we have a corresponding color
                            if corresponding_color is not None:
                                if corresponding_color not in color_statistics:
                                    color_statistics[corresponding_color] = {
                                        "ground_truth_matches": 0,
                                        "total_samples": 0
                                    }
                                color_statistics[corresponding_color]["total_samples"] += 1
                                if ground_truth_match:
                                    color_statistics[corresponding_color]["ground_truth_matches"] += 1
                            
                            # Add row/col information based on patch_idx
                            row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                            
                            patch_results = {
                                "patch_idx": patch_idx,
                                "patch_row": row,
                                "patch_col": col,
                                "nearest_neighbors": [
                                    {"token": get_display_token(token), "similarity": float(sim)}
                                    for token, sim in zip(patch_tokens, patch_values)
                                ],
                                "ground_truth_match": ground_truth_match,
                                "corresponding_color": corresponding_color,
                                "caption_match": ground_truth_match  # For compatibility with interactive viewer
                            }
                            
                            # Add detailed match information if any matches were found
                            if match_details:
                                patch_results["matches"] = match_details
                        
                        chunk_results["patches"].append(patch_results)
                    
                    image_results["chunks"].append(chunk_results)

        split_results.append(image_results)
        
        # Clear memory after each image to prevent OOM
        clear_gpu_memory()

    # Gather results from all processes
    all_split_results = [None] * world_size
    all_token_position_stats = [None] * world_size
    if dataset_type == "mosaic":
        all_color_statistics = [None] * world_size
    
    # Use all_gather to collect results from all processes
    dist.all_gather_object(all_split_results, split_results)
    dist.all_gather_object(all_token_position_stats, token_position_stats)
    if dataset_type == "mosaic":
        dist.all_gather_object(all_color_statistics, color_statistics)
    
    # Combine results on main process
    if local_rank == 0:
        # Flatten the gathered results
        combined_results = []
        for process_results in all_split_results:
            combined_results.extend(process_results)
        
        # Combine token position statistics
        combined_token_position_stats = {}
        for process_stats in all_token_position_stats:
            for patch_idx, stats in process_stats.items():
                if patch_idx not in combined_token_position_stats:
                    combined_token_position_stats[patch_idx] = {
                        "total_occurrences": 0,
                        "interpretable_occurrences": 0,
                        "visual_task_occurrences": 0,
                        "interpretability_percentage": 0.0,
                        "visual_task_percentage": 0.0,
                        "patch_row": stats.get("patch_row", 0),
                        "patch_col": stats.get("patch_col", 0)
                    }
                combined_token_position_stats[patch_idx]["total_occurrences"] += stats["total_occurrences"]
                combined_token_position_stats[patch_idx]["interpretable_occurrences"] += stats["interpretable_occurrences"]
                combined_token_position_stats[patch_idx]["visual_task_occurrences"] += stats["visual_task_occurrences"]
        
        # Calculate interpretability percentages for token positions
        for pos_stats in combined_token_position_stats.values():
            total = pos_stats["total_occurrences"]
            interpretable = pos_stats["interpretable_occurrences"]
            visual_task = pos_stats["visual_task_occurrences"]
            pos_stats["interpretability_percentage"] = (interpretable / total * 100) if total > 0 else 0.0
            pos_stats["visual_task_percentage"] = (visual_task / total * 100) if total > 0 else 0.0
        
        if dataset_type == "pixmo_cap":
            # Calculate overall statistics
            total_visual_tokens = sum(pos_stats["total_occurrences"] for pos_stats in combined_token_position_stats.values())
            interpretable_visual_tokens = sum(pos_stats["interpretable_occurrences"] for pos_stats in combined_token_position_stats.values())
            visual_task_visual_tokens = sum(pos_stats["visual_task_occurrences"] for pos_stats in combined_token_position_stats.values())
            
            overall_statistics = {
                "total_visual_tokens": total_visual_tokens,
                "interpretable_visual_tokens": interpretable_visual_tokens,
                "visual_task_visual_tokens": visual_task_visual_tokens,
                "interpretability_percentage": (interpretable_visual_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
                "visual_task_percentage": (visual_task_visual_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
                "token_position_statistics": combined_token_position_stats
            }
            return combined_results, overall_statistics
        else:  # mosaic
            # Combine color statistics
            combined_color_statistics = {}
            for process_stats in all_color_statistics:
                for color, stats in process_stats.items():
                    if color not in combined_color_statistics:
                        combined_color_statistics[color] = {
                            "ground_truth_matches": 0,
                            "total_samples": 0
                        }
                    combined_color_statistics[color]["ground_truth_matches"] += stats["ground_truth_matches"]
                    combined_color_statistics[color]["total_samples"] += stats["total_samples"]
            
            return combined_results, combined_token_position_stats, combined_color_statistics
    else:
        # Non-main processes return None
        if dataset_type == "pixmo_cap":
            return None, None
        else:
            return None, None, None

def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # Set CUDA device
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    
    parser = argparse.ArgumentParser(description="Analyze PixMoCap or Mosaic interpretability (Multi-GPU version)")
    parser.add_argument("--dataset", type=str, choices=["pixmo_cap", "mosaic"], default="pixmo_cap",
                       help="Which dataset to analyze (default: pixmo_cap)")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to checkpoint to analyze")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-running analysis even if results file exists")
    parser.add_argument("--preprocessing_mode", type=str, choices=["padding", "warping"], 
                       help="Preprocessing mode: 'padding' for black padding, 'warping' for direct resize, or omit for config default")
    parser.add_argument("--llm_layer", type=int, default=0,
                       help="If >0, extract visual token embeddings from this LLM layer (0 keeps original vision features)")
    parser.add_argument("--llm_layers", type=int, nargs='+',
                       help="Optional list of LLM layer indices to evaluate; saves one result per layer")
    parser.add_argument("--skip_generate", action="store_true",
                       help="Skip model.generate() inside loop to avoid extra FSDP comm and desync")
    parser.add_argument("--sanity_check_l0_l1", action="store_true",
                       help="Print cosine similarity stats between layer 0 and layer 1 visual tokens")
    parser.add_argument("--sanity_threshold", type=float, default=0.9,
                       help="Threshold for l0 vs l1 cosine similarity (default: 0.9)")
    args = parser.parse_args()
    
    # Hardcoded parameters based on dataset
    checkpoint_path = args.ckpt_path
    if args.dataset == "pixmo_cap":
        prompt = "Describe this image in detail."
        dataset_name = "PixMoCap"
        output_suffix = "pixmo_cap"
        num_train_images = 100
        num_val_images = 100
    else:  # mosaic
        prompt = "What is the sequence of colors in this grid of colors, read from left to right like a page?"
        dataset_name = "ColorMosaicDataset"
        output_suffix = "color_names_mosaic_24x24"
        num_train_images = 10
        num_val_images = 10
    
    if local_rank == 0:
        print(f"Dataset: {args.dataset}")
        print(f"Prompt: {prompt}")
        print(f"Running on {world_size} processes")
        print(f"LLM layer for features: {args.llm_layer}")
        if args.skip_generate:
            print("Skipping generation during analysis (--skip_generate)")
        if args.sanity_check_l0_l1:
            print(f"Sanity check enabled: l0 vs l1 cosine (threshold={args.sanity_threshold})")

    # Setup results directory (only on main process)
    if local_rank == 0:
        ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
        # Add preprocessing mode to folder name if specified
        if args.preprocessing_mode:
            ckpt_name += f"_{args.preprocessing_mode}"
        results_dir = Path("analysis_results/nearest_neighbors") / ckpt_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directory
        images_dir = results_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Check if results already exist
        layer_suffix = f"_layer{args.llm_layer}" if args.llm_layer and args.llm_layer > 0 else ""
        output_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}.json"
        if output_file.exists() and not args.force_rerun:
            print(f"Results file already exists: {output_file}")
            print("Use --force-rerun flag to re-run analysis, or delete the file to start fresh")
            return
    else:
        results_dir = None
        images_dir = None
        output_file = None

    # Wait for main process to set up directories
    dist.barrier()

    if local_rank == 0:
        print("Running full analysis...")

    # Load model with FSDP
    if local_rank == 0:
        print(f"Loading model from {checkpoint_path}")
    
    # Load model on CPU first
    model = Molmo.from_checkpoint(checkpoint_path, device="cpu")
    model.eval()
    
    # Wrap model with FSDP for sharding
    if local_rank == 0:
        print("Wrapping model with FSDP for sharding...")
    
    # Get FSDP wrap policy from the model
    wrap_policy = model.get_fsdp_wrap_policy("by_block_and_size")
    
    # Wrap model in FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    
    if local_rank == 0:
        print(f"Model wrapped with FSDP on device: {device}")

    # Create preprocessor
    if "hf:" in checkpoint_path:
        model_config = model.config
    else:
        model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # Override system prompt kind to avoid length conditioning
    model_config.system_prompt_kind = "style" if args.dataset == "mosaic" else "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    # Override resize behavior based on preprocessing mode
    if args.preprocessing_mode == "padding":
        # Store original resize method
        original_resize = preprocessor.mm_preprocessor.resize
        
        # Override the resize_image method to force black padding
        def resize_image_with_black_padding(image, output_size, is_training, rng):
            # Force use of resize_and_pad with black padding regardless of resize mode
            return resize_and_pad(
                image, output_size, pad_value=0, rng=rng, is_training=is_training,
                resize_method="torch-bilinear")
        
        preprocessor.mm_preprocessor.resize_image = resize_image_with_black_padding
    elif args.preprocessing_mode == "warping":
        # Store original resize method
        original_resize = preprocessor.mm_preprocessor.resize
        
        # Override the resize_image method to force direct resize (no padding)
        def resize_image_with_warping(image, output_size, is_training, rng):
            # Force direct resize without padding (stretches image to fill target size)
            # Use SigLIP method for consistent warping behavior
            return siglip_resize_and_pad(image, output_size)
        
        preprocessor.mm_preprocessor.resize_image = resize_image_with_warping

    use_n_token_only = model_config.vision_backbone.use_n_token_only

    # Initialize results dictionary (only on main process)
    if local_rank == 0:
        all_results = {
            "checkpoint": checkpoint_path,
            "prompt": prompt,
            "dataset": dataset_name,
            "dataset_type": args.dataset,
            "num_processes": world_size,
            "preprocessing_mode": args.preprocessing_mode,
            "llm_layer": args.llm_layer,
            "skip_generate": args.skip_generate,
            "splits": {},
            "overall_statistics": {}
        }
    else:
        all_results = None

    # Decide which layers to run
    if args.llm_layers and len(args.llm_layers) > 0:
        layers_to_run = args.llm_layers
    else:
        # Default: first few (1,2,3,4) then every 4th (8,12,...) up to last block
        n_layers = model.config.n_layers
        base_layers = [l for l in [1, 2, 3, 4] if l < n_layers]
        periodic_layers = list(range(8, n_layers, 4))
        last_block = n_layers - 1
        layers_to_run = base_layers + [l for l in periodic_layers if l not in base_layers]
        if last_block not in layers_to_run and last_block > 0:
            layers_to_run.append(last_block)
    if local_rank == 0:
        print(f"Evaluating layers: {layers_to_run}")

    try:
        for target_layer in layers_to_run:
            if local_rank == 0:
                print(f"\n=== Evaluating LLM layer {target_layer} ===")
            # Re-init results container per layer
            if local_rank == 0:
                all_results = {
                    "checkpoint": checkpoint_path,
                    "prompt": prompt,
                    "dataset": dataset_name,
                    "dataset_type": args.dataset,
                    "num_processes": world_size,
                    "preprocessing_mode": args.preprocessing_mode,
                    "llm_layer": target_layer,
                    "skip_generate": args.skip_generate,
                    "splits": {},
                    "overall_statistics": {}
                }
            # Process train split
            if local_rank == 0:
                print("Processing train split...")
            if args.dataset == "pixmo_cap":
                train_dataset = PixMoCap(split="train", mode="captions")
            else:  # mosaic
                train_dataset = ColorMosaicDataset(split="train", grid_size=24)
            train_results = process_split(
                model, preprocessor, train_dataset, num_train_images, prompt,
                use_n_token_only, images_dir, "train", args.dataset, target_layer, args.skip_generate
            )
            if local_rank == 0:
                if args.dataset == "pixmo_cap":
                    train_images, train_statistics = train_results
                    all_results["splits"]["train"] = {
                        "num_images": num_train_images,
                        "images": train_images,
                        "statistics": train_statistics
                    }
                else:  # mosaic
                    train_images, train_token_position_stats, train_color_statistics = train_results
                    all_results["splits"]["train"] = {
                        "num_images": num_train_images,
                        "images": train_images,
                        "token_position_statistics": train_token_position_stats,
                        "color_statistics": train_color_statistics
                    }
            del train_dataset
            del train_results
            clear_gpu_memory()
            dist.barrier()
            # Process validation split
            if local_rank == 0:
                print("Processing validation split...")
            if args.dataset == "pixmo_cap":
                val_dataset = PixMoCap(split="validation", mode="captions")
            else:  # mosaic
                val_dataset = ColorMosaicDataset(split="validation", grid_size=24)
            val_results = process_split(
                model, preprocessor, val_dataset, num_val_images, prompt,
                use_n_token_only, images_dir, "validation", args.dataset, target_layer, args.skip_generate
            )
            del val_dataset
            clear_gpu_memory()
            dist.barrier()
            # Combine and save
            if local_rank == 0:
                if args.dataset == "pixmo_cap":
                    val_images, val_statistics = val_results
                    all_results["splits"]["validation"] = {
                        "num_images": num_val_images,
                        "images": val_images,
                        "statistics": val_statistics
                    }
                    train_statistics = all_results["splits"]["train"]["statistics"]
                    combined_total_tokens = train_statistics["total_visual_tokens"] + val_statistics["total_visual_tokens"]
                    combined_interpretable_tokens = train_statistics["interpretable_visual_tokens"] + val_statistics["interpretable_visual_tokens"]
                    combined_visual_task_tokens = train_statistics["visual_task_visual_tokens"] + val_statistics["visual_task_visual_tokens"]
                    combined_token_position_stats = {}
                    for split_stats in [train_statistics, val_statistics]:
                        for pos, stats in split_stats["token_position_statistics"].items():
                            if pos not in combined_token_position_stats:
                                combined_token_position_stats[pos] = {
                                    "total_occurrences": 0,
                                    "interpretable_occurrences": 0,
                                    "visual_task_occurrences": 0,
                                    "interpretability_percentage": 0.0,
                                    "visual_task_percentage": 0.0,
                                    "patch_row": stats.get("patch_row", 0),
                                    "patch_col": stats.get("patch_col", 0)
                                }
                            combined_token_position_stats[pos]["total_occurrences"] += stats["total_occurrences"]
                            combined_token_position_stats[pos]["interpretable_occurrences"] += stats["interpretable_occurrences"]
                            combined_token_position_stats[pos]["visual_task_occurrences"] += stats["visual_task_occurrences"]
                    for pos_stats in combined_token_position_stats.values():
                        total = pos_stats["total_occurrences"]
                        interpretable = pos_stats["interpretable_occurrences"]
                        visual_task = pos_stats["visual_task_occurrences"]
                        pos_stats["interpretability_percentage"] = (interpretable / total * 100) if total > 0 else 0.0
                        pos_stats["visual_task_percentage"] = (visual_task / total * 100) if total > 0 else 0.0
                    all_results["overall_statistics"] = {
                        "total_visual_tokens": combined_total_tokens,
                        "interpretable_visual_tokens": combined_interpretable_tokens,
                        "visual_task_visual_tokens": combined_visual_task_tokens,
                        "interpretability_percentage": (combined_interpretable_tokens / combined_total_tokens * 100) if combined_total_tokens > 0 else 0,
                        "visual_task_percentage": (combined_visual_task_tokens / combined_total_tokens * 100) if combined_total_tokens > 0 else 0,
                        "train_statistics": train_statistics,
                        "validation_statistics": val_statistics,
                        "token_position_statistics": combined_token_position_stats
                    }
                else:  # mosaic
                    val_images, val_token_position_stats, val_color_statistics = val_results
                    all_results["splits"]["validation"] = {
                        "num_images": num_val_images,
                        "images": val_images,
                        "token_position_statistics": val_token_position_stats,
                        "color_statistics": val_color_statistics
                    }
                    combined_token_position_stats = {}
                    for split_name, split_data in all_results["splits"].items():
                        split_stats = split_data["token_position_statistics"]
                        for patch_idx, stats in split_stats.items():
                            if patch_idx not in combined_token_position_stats:
                                combined_token_position_stats[patch_idx] = {
                                    "total_occurrences": 0,
                                    "interpretable_occurrences": 0,
                                    "interpretability_percentage": 0.0,
                                    "patch_row": stats.get("patch_row", 0),
                                    "patch_col": stats.get("patch_col", 0)
                                }
                            combined_token_position_stats[patch_idx]["total_occurrences"] += stats["total_occurrences"]
                            combined_token_position_stats[patch_idx]["interpretable_occurrences"] += stats["interpretable_occurrences"]
                    for pos_stats in combined_token_position_stats.values():
                        total = pos_stats["total_occurrences"]
                        interpretable = pos_stats["interpretable_occurrences"]
                        pos_stats["interpretability_percentage"] = (interpretable / total * 100) if total > 0 else 0.0
                    combined_color_statistics = {}
                    for split_name, split_data in all_results["splits"].items():
                        split_stats = split_data["color_statistics"]
                        for color, stats in split_stats.items():
                            if color not in combined_color_statistics:
                                combined_color_statistics[color] = {
                                    "ground_truth_matches": 0,
                                    "total_samples": 0
                                }
                            combined_color_statistics[color]["ground_truth_matches"] += stats["ground_truth_matches"]
                            combined_color_statistics[color]["total_samples"] += stats["total_samples"]
                    for color, stats in combined_color_statistics.items():
                        total = stats["total_samples"]
                        if total > 0:
                            stats["ground_truth_accuracy"] = stats["ground_truth_matches"] / total
                    sorted_color_statistics = []
                    for color, stats in combined_color_statistics.items():
                        stats_with_color = stats.copy()
                        stats_with_color["color"] = color
                        sorted_color_statistics.append(stats_with_color)
                    sorted_color_statistics.sort(key=lambda x: (x.get("ground_truth_accuracy", 0)), reverse=True)
                    total_interpretable_tokens = sum(stats["interpretable_occurrences"] for stats in combined_token_position_stats.values())
                    total_visual_tokens = sum(stats["total_occurrences"] for stats in combined_token_position_stats.values())
                    all_results["overall_statistics"] = {
                        "total_visual_tokens": total_visual_tokens,
                        "interpretable_visual_tokens": total_interpretable_tokens,
                        "interpretability_percentage": (total_interpretable_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
                        "token_position_statistics": combined_token_position_stats,
                        "color_statistics": sorted_color_statistics
                    }
                # Save per-layer outputs
                if local_rank == 0:
                    layer_suffix = f"_layer{target_layer}" if target_layer and target_layer > 0 else ""
                    per_layer_output_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}.json"
                    with open(per_layer_output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                    print(f"Results saved to {per_layer_output_file}")
                    if args.dataset == "pixmo_cap":
                        plot_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}_summary_plot.png"
                        create_interpretability_plot(all_results["overall_statistics"], plot_file)
                        token_plot_file = results_dir / f"token_position_interpretability_plot_multi-gpu{layer_suffix}.png"
                        create_token_position_plot(all_results["overall_statistics"]["token_position_statistics"], token_plot_file)
                    else:
                        color_plot_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu{layer_suffix}_color_plot.png"
                        create_color_interpretability_plot(all_results["overall_statistics"]["color_statistics"], color_plot_file)
                        token_plot_file = results_dir / f"token_position_interpretability_plot_multi-gpu{layer_suffix}.png"
                        create_token_position_plot(all_results["overall_statistics"]["token_position_statistics"], token_plot_file)

    except Exception as e:
        if local_rank == 0:
            log.error(f"Error during processing: {str(e)}")
            output_file = results_dir / f"nearest_neighbors_analysis_{output_suffix}_multi-gpu_partial.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        raise

    # Save any remaining translations to cache and finish
    if local_rank == 0:
        save_translation_cache()
        print(f"Images saved to {images_dir} (Train: {num_train_images}, Validation: {num_val_images})")
        print("Analysis complete!")

    # Wait for all processes to finish
    dist.barrier()

if __name__ == "__main__":
    main() 
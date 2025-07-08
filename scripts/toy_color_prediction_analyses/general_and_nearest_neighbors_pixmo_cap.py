"""This script takes the checkpoint we trained on the PixMo-Cap dataset and runs inference on the train and validation splits.
For each image, and each of the visual tokens in such an image:
a) we log the top5 nearest neighbor vocabularies from the LLM tokenizer
b) we check if any of the top5 nearest neighbors match any word from the ground truth caption (fuzzy matching)
c) we track token position statistics across all images
"""
import logging
import sys
import json
import re
from pathlib import Path
import argparse
import math

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add translation library
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("Warning: googletrans not available. Install with: pip install googletrans==4.0.0rc1")

from olmo.config import ModelConfig
from olmo.data import build_mm_preprocessor
from olmo.model import Molmo
from olmo.util import resource_path
from olmo.data.pixmo_datasets import PixMoCap
from olmo.data.model_preprocessor import load_image, resize_and_pad

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

def update_existing_results(results_file_path):
    """Update existing results JSON with detailed match information and token position stats."""
    print(f"Updating existing results file: {results_file_path}")
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    # Create images directory
    images_dir = results_file_path.parent / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load datasets for image saving
    train_dataset = PixMoCap(split="train", mode="captions")
    val_dataset = PixMoCap(split="validation", mode="captions")
    
    # Create a preprocessor for image processing
    try:
        # Try to get preprocessor from the checkpoint path in results
        checkpoint_path = all_results.get("checkpoint", "")
        if checkpoint_path:
            if "hf:" in checkpoint_path:
                model = Molmo.from_checkpoint(checkpoint_path, device="cpu")
                model_config = model.config
            else:
                model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
            model_config.system_prompt_kind = "none"
            preprocessor = build_mm_preprocessor(
                model_config,
                for_inference=True,
                shuffle_messages=False,
                is_training=False,
                require_image_features=True
            )
        else:
            preprocessor = None
            log.warning("No checkpoint path found, will save images without preprocessing")
    except Exception as e:
        log.warning(f"Could not create preprocessor: {e}, will save images without preprocessing")
        preprocessor = None
    
    # Initialize token position statistics
    token_position_stats = {}
    images_saved = 0
    
    # Determine patches_per_chunk from existing data
    patches_per_chunk = None
    for split_name, split_data in all_results.get("splits", {}).items():
        for image_data in split_data.get("images", []):
            for chunk in image_data.get("chunks", []):
                patches_per_chunk = len(chunk.get("patches", []))
                break
            if patches_per_chunk:
                break
        if patches_per_chunk:
            break
    
    if patches_per_chunk is None:
        log.warning("Could not determine patches_per_chunk from existing data, defaulting to 144")
        patches_per_chunk = 144
    
    print(f"Detected {patches_per_chunk} patches per chunk (grid size: {int(math.sqrt(patches_per_chunk))}x{int(math.sqrt(patches_per_chunk))})")
    
    # Process each split
    for split_name, split_data in all_results.get("splits", {}).items():
        print(f"Processing {split_name} split...")
        print(f"Number of images in {split_name}: {len(split_data.get('images', []))}")
        
        # Select appropriate dataset
        dataset = train_dataset if split_name == "train" else val_dataset
        
        for image_data in tqdm(split_data.get("images", [])):
            image_idx = image_data.get("image_idx", 0)
            
            # Collect interpretable patch indices for this image
            interpretable_patches = set()
            for chunk in image_data.get("chunks", []):
                for patch in chunk.get("patches", []):
                    if patch.get("caption_match", False):
                        interpretable_patches.add(patch.get("patch_idx", -1))
            
            # Save image with token grid overlay (always save/overwrite)
            try:
                example_data = dataset.get(image_idx, np.random)
                image_path = example_data["image"]
                print(f"Processing image {image_idx}, image_path: {image_path}")
                
                # Load the image from file path using PIL, just like the preprocessor does
                with Image.open(image_path) as pil_image:
                    # Convert to RGB to ensure compatibility
                    pil_image = pil_image.convert("RGB")
                    
                    # Save both normal and preprocessed images with token grid overlay
                    image_filename_base = f"{split_name}_image_{image_idx:04d}"
                    image_save_path = images_dir / f"{image_filename_base}.jpg"
                    save_image_with_token_grid(pil_image, image_save_path, patches_per_chunk, preprocessor, interpretable_patches)
                    
                    # Update image filenames in the data
                    image_data["image_filename"] = f"{image_filename_base}.jpg"
                    image_data["image_filename_grid"] = f"{image_filename_base}_grid.jpg"
                    images_saved += 1
                    print(f"Successfully saved {image_filename_base}.jpg and {image_filename_base}_grid.jpg with {len(interpretable_patches)} interpretable patches")
                    
            except Exception as e:
                print(f"Could not save image {image_idx} from {split_name}: {e}")
                log.warning(f"Could not save image {image_idx} from {split_name}: {e}")
            
            # Get ground truth caption and normalize
            ground_truth_caption = image_data.get("ground_truth_caption", "")
            if not ground_truth_caption:
                ground_truth_caption = image_data.get("caption_text", "")
            caption_words, visual_task_words = normalize_text_for_matching(ground_truth_caption)
            
            # Remove caption_words key if it exists
            if "caption_words" in image_data:
                del image_data["caption_words"]
            
            # Process each chunk
            for chunk in image_data.get("chunks", []):
                for patch in chunk.get("patches", []):
                    patch_idx = patch.get("patch_idx", -1)
                    
                    # Add row/col information based on patch_idx
                    if patch_idx >= 0:
                        row, col = patch_idx_to_row_col(patch_idx, patches_per_chunk)
                        patch["patch_row"] = row
                        patch["patch_col"] = col
                    
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
                    
                    # Get nearest neighbors
                    nearest_neighbors = patch.get("nearest_neighbors", [])
                    
                    # Check for matches using new three-category system
                    overall_match_type = 'none'
                    all_matches = []
                    
                    for neighbor in nearest_neighbors:
                        token_text = neighbor.get("token", "")
                        token_words, _ = normalize_text_for_matching(token_text)
                        
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
                                        "similarity": neighbor.get("similarity", 0.0)
                                    })
                    
                    # Update patch with new visual_task_match field
                    patch["visual_task_match"] = (overall_match_type == 'visual_task')
                    
                    # Update statistics
                    if overall_match_type == 'interpretable':
                        token_position_stats[patch_idx]["interpretable_occurrences"] += 1
                    elif overall_match_type == 'visual_task':
                        token_position_stats[patch_idx]["visual_task_occurrences"] += 1
                    
                    # Add detailed matches if any were found and not already present
                    if all_matches and "matches" not in patch:
                        # Update matches to use display tokens
                        display_matches = []
                        for match in all_matches:
                            display_matches.append({
                                "token": get_display_token(match["token"]),
                                "token_word": match["token_word"],
                                "matched_word": match["matched_word"],
                                "match_type": match["match_type"],
                                "similarity": match["similarity"]
                            })
                        patch["matches"] = display_matches
                    
                    # Update nearest_neighbors to use display tokens
                    if "nearest_neighbors" in patch:
                        for neighbor in patch["nearest_neighbors"]:
                            if "token" in neighbor:
                                neighbor["token"] = get_display_token(neighbor["token"])

    # Calculate interpretability percentages for token positions
    for pos_stats in token_position_stats.values():
        total = pos_stats["total_occurrences"]
        interpretable = pos_stats["interpretable_occurrences"]
        visual_task = pos_stats["visual_task_occurrences"]
        pos_stats["interpretability_percentage"] = (interpretable / total * 100) if total > 0 else 0.0
        pos_stats["visual_task_percentage"] = (visual_task / total * 100) if total > 0 else 0.0
    
    # Add token position statistics to results
    all_results["token_position_statistics"] = token_position_stats
    
    # Update overall statistics if they exist
    for split_name, split_data in all_results.get("splits", {}).items():
        if "statistics" in split_data:
            split_stats = split_data["statistics"]
            # Add token position stats to split statistics
            split_stats["token_position_statistics"] = token_position_stats
    
    # Save updated results
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Updated results saved to {results_file_path}")
    print(f"Images saved: {images_saved} (saved to {images_dir})")
    
    # Create token position plot
    plot_path = results_file_path.parent / "token_position_interpretability_plot.png"
    create_token_position_plot(token_position_stats, plot_path)
    
    # Save any remaining translations to cache
    save_translation_cache()
    
    return all_results

def process_split(model, preprocessor, dataset, num_images, prompt, use_n_token_only, images_dir=None, split_name=""):
    """Process a dataset split and return results with detailed match information."""
    split_results = []
    
    # Initialize statistics tracking
    total_visual_tokens = 0
    interpretable_visual_tokens = 0
    visual_task_visual_tokens = 0
    token_position_stats = {}
    
    # Process each image
    for i in tqdm(range(num_images)):
        example_data = dataset.get(i, np.random)
        
        # Extract ground truth caption
        caption_text = ""
        if "message_list" in example_data and len(example_data["message_list"]) > 0:
            message = example_data["message_list"][0]
            caption_text = message.get("text", "")
        
        # Save image if images_dir is provided
        image_filename = None
        image_filename_grid = None
        if images_dir is not None:
            try:
                image_path = example_data["image"]
                # Load the image from file path using PIL, just like the preprocessor does
                with Image.open(image_path) as pil_image:
                    # Convert to RGB to ensure compatibility
                    pil_image = pil_image.convert("RGB")
                    
                    # We'll save the image with token grid overlay after we determine patches_per_chunk
                    temp_pil_image = pil_image  # Store for later use
                    
            except Exception as e:
                log.warning(f"Could not load image {i}: {e}")
                temp_pil_image = None
        else:
            temp_pil_image = None
        
        # Normalize caption words for matching
        caption_words, visual_task_words = normalize_text_for_matching(caption_text)
        
        # Create example with the provided prompt
        example = {
            "image": example_data["image"],
            "messages": [prompt]
        }

        # Preprocess example
        batch = preprocessor(example, rng=np.random)

        # Initialize image results
        image_results = {
            "image_idx": i,
            "ground_truth_caption": caption_text,
            "chunks": []
        }

        # Run inference
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                # Move data to GPU
                images_tensor = torch.tensor(batch.get("images")).unsqueeze(0).cuda()
                image_masks_tensor = torch.tensor(batch.get("image_masks")).unsqueeze(0).cuda() if batch.get("image_masks") is not None else None
                
                image_features, tokens_before_MLP = model.vision_backbone(images_tensor, image_masks_tensor, return_tokens_before_MLP=True)
                # Handle use_n_token_only properly
                if type(use_n_token_only) == int and use_n_token_only != -1:
                    image_features = image_features[:, :, :use_n_token_only, :]
                elif type(use_n_token_only) == list and len(use_n_token_only) > 0:
                    image_features = image_features[:, :, use_n_token_only, :]
                image_results["feature_shape"] = list(image_features.shape)
                
                # Get token embeddings from the model
                token_embeddings = model.transformer.wte.embedding
                
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
                
                # Clear intermediate tensors
                del similarity, image_features_norm, token_embeddings_norm
                clear_gpu_memory()

                # generated output for reference
                input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).cuda()
                output = model.generate(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_masks=image_masks_tensor,
                    image_input_idx=torch.tensor(batch.get("image_input_idx")).unsqueeze(0).cuda() if batch.get("image_input_idx") is not None else None,
                    max_steps=200,
                    is_distributed=False
                )
                token_ids = output.token_ids[:, 0].detach().cpu().numpy()
                decoded = preprocessor.tokenizer.decode(token_ids[0])
                image_results["generated_response"] = decoded

                # Clear GPU tensors
                del images_tensor, image_masks_tensor, image_features, input_ids, output
                clear_gpu_memory()

                # Now save the image with grid overlay if we have it
                if temp_pil_image is not None and images_dir is not None:
                    try:
                        # Collect interpretable patches for this image
                        interpretable_patches = set()
                        for chunk_idx in range(num_chunks):
                            for patch_idx in range(patches_per_chunk):
                                # Check if this patch will be interpretable based on the matches we found
                                patch_values = top_values[chunk_idx, patch_idx].cpu().numpy().tolist()
                                patch_indices = top_indices[chunk_idx, patch_idx].cpu().numpy().tolist()
                                patch_tokens = [decode_token(preprocessor.tokenizer, idx) for idx in patch_indices]
                                
                                # Check for matches (same logic as below)
                                caption_match = False
                                for j, (token_text, similarity_score) in enumerate(zip(patch_tokens, patch_values)):
                                    token_words, _ = normalize_text_for_matching(token_text)
                                    for token_word in token_words:
                                        for caption_word in caption_words:
                                            if token_word == caption_word:
                                                caption_match = True
                                                break
                                        if caption_match:
                                            break
                                    if caption_match:
                                        break
                                
                                if caption_match:
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
                else:
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
                        
                        # Check for matches using new three-category system
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
                        
                        # Update patch with new visual_task_match field
                        patch_results = {
                            "patch_idx": patch_idx,
                            "nearest_neighbors": [
                                {"token": get_display_token(token), "similarity": float(sim)}
                                for token, sim in zip(patch_tokens, patch_values)
                            ],
                            "match_type": overall_match_type
                        }
                        
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
                        patch_results["patch_row"] = row
                        patch_results["patch_col"] = col
                        
                        # Add backward compatibility fields
                        patch_results["caption_match"] = overall_match_type == 'interpretable'
                        patch_results["visual_task_match"] = overall_match_type == 'visual_task'
                        
                        # Add detailed matches if any were found and not already present
                        if all_matches and "matches" not in patch_results:
                            # Update matches to use display tokens
                            display_matches = []
                            for match in all_matches:
                                display_matches.append({
                                    "token": get_display_token(match["token"]),
                                    "token_word": match["token_word"],
                                    "matched_word": match["matched_word"],
                                    "match_type": match["match_type"],
                                    "similarity": match["similarity"]
                                })
                            patch_results["matches"] = display_matches
                        
                        # Update nearest_neighbors to use display tokens
                        if "nearest_neighbors" in patch_results:
                            for neighbor in patch_results["nearest_neighbors"]:
                                if "token" in neighbor:
                                    neighbor["token"] = get_display_token(neighbor["token"])
                        
                        chunk_results["patches"].append(patch_results)
                    
                    image_results["chunks"].append(chunk_results)

        split_results.append(image_results)
    
    # Calculate interpretability percentages for token positions
    for pos_stats in token_position_stats.values():
        total = pos_stats["total_occurrences"]
        interpretable = pos_stats["interpretable_occurrences"]
        visual_task = pos_stats["visual_task_occurrences"]
        pos_stats["interpretability_percentage"] = (interpretable / total * 100) if total > 0 else 0.0
        pos_stats["visual_task_percentage"] = (visual_task / total * 100) if total > 0 else 0.0
    
    overall_statistics = {
        "total_visual_tokens": total_visual_tokens,
        "interpretable_visual_tokens": interpretable_visual_tokens,
        "visual_task_visual_tokens": visual_task_visual_tokens,
        "interpretability_percentage": (interpretable_visual_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
        "visual_task_percentage": (visual_task_visual_tokens / total_visual_tokens * 100) if total_visual_tokens > 0 else 0,
        "token_position_statistics": token_position_stats
    }
    
    return split_results, overall_statistics

def get_display_token(original_token):
    """Get display version of token, showing translation with original Chinese in brackets."""
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

def main():
    parser = argparse.ArgumentParser(description="Analyze PixMoCap interpretability")
    parser.add_argument("--update-only", type=str, help="Path to existing JSON file to update without re-running analysis")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-running analysis even if results file exists")
    args = parser.parse_args()
    
    # If update-only mode, just update the existing file
    if args.update_only:
        results_file = Path(args.update_only)
        if not results_file.exists():
            print(f"Error: File {results_file} does not exist")
            return
        update_existing_results(results_file)
        return
    
    # Hardcoded parameters
    checkpoint_path = "molmo_data/checkpoints/train_mlp-only_pixmo_cap_resize/step3000-unsharded"
    prompt = "Describe this image in detail."
    print(f"Prompt: {prompt}")

    # Setup results directory
    ckpt_name = checkpoint_path.split("/")[-2] + "_" + checkpoint_path.split("/")[-1]
    results_dir = Path("analysis_results/nearest_neighbors") / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Check if results already exist
    output_file = results_dir / "nearest_neighbors_analysis_pixmo_cap.json"
    if output_file.exists() and not args.force_rerun:
        print(f"Results file already exists: {output_file}")
        print("Checking if it needs updates...")
        
        # Load existing results to check what needs updating
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        
        needs_visual_task_update = False
        needs_full_update = False
        
        # Check if visual_task_match fields are missing
        for split_name, split_data in existing_results.get("splits", {}).items():
            for image_data in split_data.get("images", []):
                for chunk in image_data.get("chunks", []):
                    for patch in chunk.get("patches", []):
                        if "visual_task_match" not in patch:
                            needs_visual_task_update = True
                            break
                    if needs_visual_task_update:
                        break
                if needs_visual_task_update:
                    break
            if needs_visual_task_update:
                break
        
        # Check if images are missing
        missing_images = False
        for split_name, split_data in existing_results.get("splits", {}).items():
            for image_data in split_data.get("images", []):
                if "image_filename" not in image_data:
                    missing_images = True
                    break
            if missing_images:
                break
        
        # Check if token position statistics are missing
        missing_token_stats = "token_position_statistics" not in existing_results.get("overall_statistics", {})
        
        # Check if priority logic needs updating (visual_task should take precedence now)
        needs_priority_update = False
        sample_patch_checked = False
        for split_name, split_data in existing_results.get("splits", {}).items():
            for image_data in split_data.get("images", []):
                for chunk in image_data.get("chunks", []):
                    for patch in chunk.get("patches", []):
                        if not sample_patch_checked:
                            # Check if this patch has both interpretable and visual_task matches
                            # but visual_task_match is False (indicating old priority logic)
                            matches = patch.get("matches", [])
                            has_interpretable_match = any(m.get("match_type") == "interpretable" for m in matches)
                            has_visual_task_match = any(m.get("match_type") == "visual_task" for m in matches)
                            current_visual_task_match = patch.get("visual_task_match", False)
                            
                            # If we have both types of matches but visual_task_match is False,
                            # then the old priority logic was used (interpretable took precedence)
                            if has_interpretable_match and has_visual_task_match and not current_visual_task_match:
                                needs_priority_update = True
                                print("- Detected old priority logic: interpretable was taking precedence over visual_task")
                                break
                            sample_patch_checked = True
                    if needs_priority_update:
                        break
                if needs_priority_update:
                    break
            if needs_priority_update:
                break
        
        if needs_visual_task_update or missing_images or missing_token_stats or needs_priority_update:
            print("Existing results need updates. Updating...")
            if needs_visual_task_update:
                print("- Adding visual_task_match fields")
            if missing_images:
                print("- Saving missing images")
            if missing_token_stats:
                print("- Adding token position statistics")
            if needs_priority_update:
                print("- Updating priority logic: visual_task now takes precedence over interpretable")
            
            # Update the existing results
            update_existing_results(output_file)
            print("✓ Results updated successfully!")
            return
        else:
            print("✓ Results file is up to date!")
            print("Use --force-rerun flag to re-run analysis, or delete the file to start fresh")
        return

    print("Running full analysis...")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = Molmo.from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    # Create preprocessor
    if "hf:" in checkpoint_path:
        model_config = model.config
    else:
        model_config = ModelConfig.load(resource_path(checkpoint_path, "config.yaml"), key="model", validate_paths=False)
    # Override system prompt kind to avoid length conditioning
    model_config.system_prompt_kind = "none"
    preprocessor = build_mm_preprocessor(
        model_config,
        for_inference=True,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True
    )

    use_n_token_only = model_config.vision_backbone.use_n_token_only

    # Initialize results dictionary
    all_results = {
        "checkpoint": checkpoint_path,
        "prompt": prompt,
        "dataset": "PixMoCap",
        "splits": {},
        "overall_statistics": {}
    }

    try:
        # Process train split
        log.info("Processing train split...")
        train_dataset = PixMoCap(split="train", mode="captions")
        num_train_images = min(100, len(train_dataset))
        train_results, train_statistics = process_split(model, preprocessor, train_dataset, num_train_images, prompt, use_n_token_only, images_dir, "train")
        all_results["splits"]["train"] = {
            "num_images": num_train_images,
            "images": train_results,
            "statistics": train_statistics
        }
        
        # Clear memory before processing validation split
        clear_gpu_memory()

        # Process validation split
        log.info("Processing validation split...")
        val_dataset = PixMoCap(split="validation", mode="captions")
        num_val_images = min(100, len(val_dataset))
        val_results, val_statistics = process_split(model, preprocessor, val_dataset, num_val_images, prompt, use_n_token_only, images_dir, "validation")
        all_results["splits"]["validation"] = {
            "num_images": num_val_images,
            "images": val_results,
            "statistics": val_statistics
        }
        
        # Combine statistics across splits
        combined_total_tokens = train_statistics["total_visual_tokens"] + val_statistics["total_visual_tokens"]
        combined_interpretable_tokens = train_statistics["interpretable_visual_tokens"] + val_statistics["interpretable_visual_tokens"]
        combined_visual_task_tokens = train_statistics["visual_task_visual_tokens"] + val_statistics["visual_task_visual_tokens"]
        
        # Combine token position statistics
        combined_token_position_stats = {}
        for split_stats in [train_statistics, val_statistics]:
            for pos, stats in split_stats["token_position_statistics"].items():
                if pos not in combined_token_position_stats:
                    combined_token_position_stats[pos] = {
                        "total_occurrences": 0,
                        "interpretable_occurrences": 0,
                        "visual_task_occurrences": 0,
                        "interpretability_percentage": 0.0,
                        "visual_task_percentage": 0.0
                    }
                combined_token_position_stats[pos]["total_occurrences"] += stats["total_occurrences"]
                combined_token_position_stats[pos]["interpretable_occurrences"] += stats["interpretable_occurrences"]
                combined_token_position_stats[pos]["visual_task_occurrences"] += stats["visual_task_occurrences"]
        
        # Recalculate percentages for combined stats
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

    except Exception as e:
        log.error(f"Error during processing: {str(e)}")
        # Save partial results if there's an error
        output_file = results_dir / "nearest_neighbors_analysis_pixmo_cap_partial.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        raise

    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    log.info(f"Results saved to {output_file}")
    log.info(f"Images saved to {images_dir} (Train: {num_train_images}, Validation: {num_val_images})")
    
    # Create and save interpretability plot
    plot_file = results_dir / "nearest_neighbors_analysis_pixmo_cap_summary_plot.png"
    create_interpretability_plot(all_results["overall_statistics"], plot_file)

    # Create and save token position plot
    token_plot_file = results_dir / "token_position_interpretability_plot.png"
    create_token_position_plot(all_results["overall_statistics"]["token_position_statistics"], token_plot_file)

    # Save any remaining translations to cache
    save_translation_cache()

if __name__ == "__main__":
    main() 

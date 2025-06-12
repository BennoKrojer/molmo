"""Generate images for each token in the Molmo tokenizer."""
import logging
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def get_optimal_dimensions(text):
    """Calculate optimal image dimensions based on text length."""
    text_len = len(text)
    
    if text_len == 1:
        # For single characters, use a square
        return 200, 200
    else:
        # For longer text, ensure width is at least text length * 100
        # but cap at 1200 to prevent extremely wide images
        width = min(1200, max(400, text_len * 100))
        height = 200
        return width, height

def get_font_size_and_position(text, img_width, img_height, token_id=None):
    """Find the largest font size that fills the image and calculate its position."""
    # Start with a very large font size
    max_size = 1000
    min_size = 24
    
    # Add safety margin to account for font rendering quirks
    SAFETY_MARGIN = 4  # Increased safety margin
    
    # Scale factor to account for bounding box underestimation
    BBOX_SCALE = 1.1  # Assume bounding box is 10% smaller than actual rendered size
    
    log.info(f"Finding font size for text: '{text}' (token_id: {token_id})")
    log.info(f"Image dimensions: {img_width}x{img_height}")
    
    # Try multiple font options
    font_options = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Common on Linux
        "/System/Library/Fonts/Helvetica.ttc",              # Common on macOS
        "/System/Library/Fonts/Arial.ttf",                  # Another macOS option
        "Arial.ttf",                                        # Windows
    ]
    
    font = None
    for font_path in font_options:
        try:
            # Try to load the font at a small size first to verify it exists
            test_font = ImageFont.truetype(font_path, 24)
            font = test_font
            log.info(f"Successfully loaded font: {font_path}")
            break
        except IOError:
            continue
    
    if font is None:
        log.warning("Could not load any system fonts, using default font")
        font = ImageFont.load_default()
    
    # Binary search for the optimal font size
    left = min_size
    right = max_size
    best_font = None
    best_size = None
    best_metrics = None
    
    while left <= right:
        mid = (left + right) // 2
        try:
            if not isinstance(font, ImageFont.FreeTypeFont):
                break
            test_font = ImageFont.truetype(font.path, mid)
            bbox = test_font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Scale the dimensions to account for bounding box underestimation
            scaled_width = int(text_width * BBOX_SCALE)
            scaled_height = int(text_height * BBOX_SCALE)
            
            # Get the actual visible text height by measuring the text ascent
            # This helps account for font metrics and descenders
            ascent, descent = test_font.getmetrics()
            visible_height = ascent
            
            log.info(f"Trying font size {mid}: text dimensions {text_width}x{text_height} (scaled: {scaled_width}x{scaled_height})")
            
            # For single characters, we want to fill almost the entire image
            if len(text) == 1:
                # Use scaled dimensions for comparison
                if scaled_width <= (img_width - 2*SAFETY_MARGIN) * 0.9 and scaled_height <= (img_height - 2*SAFETY_MARGIN) * 0.9:
                    # This size fits, try a larger one
                    best_font = test_font
                    best_size = mid
                    best_metrics = (text_width, text_height, ascent, descent)
                    left = mid + 1
                else:
                    # Too big, try a smaller one
                    right = mid - 1
            else:
                # For longer text, ensure it fits with some padding
                if scaled_width <= (img_width - 2*SAFETY_MARGIN) * 0.95 and scaled_height <= (img_height - 2*SAFETY_MARGIN) * 0.95:
                    # This size fits, try a larger one
                    best_font = test_font
                    best_size = mid
                    best_metrics = (text_width, text_height, ascent, descent)
                    left = mid + 1
                else:
                    # Too big, try a smaller one
                    right = mid - 1
                    
        except Exception as e:
            log.warning(f"Error setting font size {mid}: {e}")
            right = mid - 1
    
    if best_font is None:
        # If we couldn't find a suitable size, use the minimum size
        try:
            if not isinstance(font, ImageFont.FreeTypeFont):
                best_font = ImageFont.load_default()
            else:
                best_font = ImageFont.truetype(font.path, min_size)
        except Exception as e:
            log.warning(f"Error setting minimum font size: {e}")
            best_font = ImageFont.load_default()
        
        bbox = best_font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        ascent, descent = best_font.getmetrics()
        best_metrics = (text_width, text_height, ascent, descent)
        best_size = min_size
    
    # Calculate position to center the text
    text_width, text_height, ascent, descent = best_metrics
    
    # Calculate vertical position using ascent and descent for proper centering
    # Shift text 5% upward by adjusting the y position
    x = (img_width - text_width) // 2
    y = (img_height - (ascent + descent)) // 2 - descent - int(img_height * 0.05)  # 5% upward shift
    
    # Ensure text stays within all image boundaries with safety margin
    if x < SAFETY_MARGIN:
        x = SAFETY_MARGIN
    if y < SAFETY_MARGIN:
        y = SAFETY_MARGIN
    if x + text_width > img_width - SAFETY_MARGIN:
        x = img_width - text_width - SAFETY_MARGIN
    if y + text_height > img_height - SAFETY_MARGIN:
        y = img_height - text_height - SAFETY_MARGIN
    
    # Verify final position is within bounds
    assert SAFETY_MARGIN <= x <= img_width - text_width - SAFETY_MARGIN, \
        f"Text would extend beyond horizontal bounds: token_id={token_id}, text='{text}', x={x}, width={text_width}, img_width={img_width}"
    assert SAFETY_MARGIN <= y <= img_height - text_height - SAFETY_MARGIN, \
        f"Text would extend beyond vertical bounds: token_id={token_id}, text='{text}', y={y}, height={text_height}, img_height={img_height}"
    
    # Calculate how much of the image height the text actually uses
    height_usage = text_height / img_height
    log.info(f"Using font size {best_size}")
    log.info(f"Final text dimensions: {text_width}x{text_height} (ascent: {ascent}, descent: {descent})")
    log.info(f"Position: ({x}, {y})")
    log.info(f"Text uses {height_usage:.1%} of image height")
    
    return best_font, x, y

def is_missing_glyph(font, text):
    """Check if the font can render the text properly."""
    try:
        # Get the bounding box of the text
        bbox = font.getbbox(text)
        if not bbox:
            return True
        
        # If the text is a single character and not "?", check if it renders as "?"
        if len(text) == 1 and text != "?":
            # Create a small image to render both texts
            test_img = Image.new('RGB', (50, 50), 'white')
            draw = ImageDraw.Draw(test_img)
            
            # Draw both the original text and "?"
            draw.text((0, 0), text, font=font, fill='black')
            draw.text((30, 0), "?", font=font, fill='black')
            
            # Convert to numpy arrays for comparison
            text_region = np.array(test_img.crop((0, 0, 25, 25)))
            question_region = np.array(test_img.crop((30, 0, 55, 25)))
            
            # If they're identical, it's a missing glyph
            return np.array_equal(text_region, question_region)
        
        return False
    except Exception as e:
        log.warning(f"Error checking glyph for text '{text}': {e}")
        return True  # If we can't check, assume it's a missing glyph

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate images for each token in the Molmo tokenizer.')
    parser.add_argument('--output-dir', type=str, default='molmo_data/token_images',
                      help='Directory to save token images (default: molmo_data/token_images)')
    parser.add_argument('--load_model', action='store_true',
                      help='Whether to load the model to get embedding matrix (default: False)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing_images = glob.glob(str(output_dir / "token_*.png"))
    start_token_id = len(existing_images)
    log.info(f"Found {start_token_id} existing token images. Starting from token ID {start_token_id}")

    # Load tokenizer
    log.info("Loading tokenizer...")
    model_name = "Qwen/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = len(tokenizer)
    log.info(f"Tokenizer vocab size: {vocab_size}")

    # Optionally load model to get embedding matrix
    if args.load_model:
        log.info("Loading model to get embedding matrix...")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True)
        embedding_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()
        # save embedding matrix to file
        np.save("embedding_matrix_qwen2_7b.npy", embedding_matrix)
        exit()
        log.info(f"Embedding matrix size: {embedding_matrix.shape[0]}")
        # Only use the first vocab_size rows of the embedding matrix
        # This is because the model's embedding matrix is larger than the tokenizer's vocabulary
        embedding_matrix = embedding_matrix[:vocab_size]
        log.info(f"Using first {vocab_size} rows of embedding matrix")

    # Generate images for each token
    log.info("Generating token images...")
    for token_id in range(0, vocab_size):
        try:
            # Get token text
            token_text = tokenizer.decode([token_id])
            
            # Skip invalid tokens (those that decode to replacement character)
            if len(token_text) == 1 and ord(token_text) == 0xFFFD:  # Unicode replacement character
                log.warning(f"Skipping token_id {token_id} - invalid token (decodes to replacement character)")
                continue
            
            # Calculate optimal image dimensions based on text
            img_width, img_height = get_optimal_dimensions(token_text)
            
            # Skip if width would be too large
            if img_width > 1000:
                log.warning(f"Skipping token_id {token_id}, text: '{token_text}' - requires width {img_width} > 1000")
                continue
            
            # Create a new image with white background
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Get font and position
            font, x, y = get_font_size_and_position(token_text, img_width, img_height, token_id)
            
            # Check if the font can render the token
            if is_missing_glyph(font, token_text):
                log.warning(f"Skipping token_id {token_id}, text: '{token_text}' - missing glyph")
                continue
            
            # Draw the token text
            draw.text((x, y), token_text, font=font, fill='black')
            
            # Save the image
            img.save(output_dir / f"token_{token_id}.png")
            
            if (token_id + 1) % 100 == 0:
                log.info(f"Generated {token_id + 1} token images...")
        except Exception as e:
            log.error(f"Error processing token_id {token_id}, text: '{token_text}': {str(e)}")
            raise

    log.info(f"Successfully generated {vocab_size - start_token_id} new token images in {output_dir}")

if __name__ == "__main__":
    main() 
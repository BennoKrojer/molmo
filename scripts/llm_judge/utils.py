
import json
import os
import random
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    is_valid_image,
)
from transformers.processing_utils import ImagesKwargs
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import logging

def clip_bbox_to_image(bbox, image_width, image_height):
    """
    Clip bounding box coordinates to image boundaries.
    
    Args:
        bbox (tuple): (left, top, right, bottom) coordinates
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        tuple: Clipped (left, top, right, bottom) coordinates
    """
    left, top, right, bottom = bbox
    
    # Clip to image boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    
    return (left, top, right, bottom)

def draw_bbox_on_image(image, bbox, outline_color="red", width=3):
    """
    Draw bounding box on image with better visibility for edge cases.
    
    Args:
        image: PIL Image object
        bbox (tuple): (left, top, right, bottom) coordinates
        outline_color (str): Color of the outline (default: "red")
        width (int): Width of the outline (default: 3)
        fill_alpha (int): Transparency for fill overlay (0-255, default: 30)
        
    Returns:
        PIL Image: Image with bounding box drawn
    """
    new_image = deepcopy(image)
    img_width, img_height = new_image.size
    
    # Clip bounding box to image boundaries
    clipped_bbox = clip_bbox_to_image(bbox, img_width, img_height)
    left, top, right, bottom = clipped_bbox
    
    # Check if bounding box is valid after clipping
    if left >= right or top >= bottom:
        return new_image  # Return original image if bbox is invalid
    
    draw = ImageDraw.Draw(new_image)
    
    # Draw the outline
    draw.rectangle(clipped_bbox, outline=outline_color, width=width)
    
    # Add corner markers for better visibility at edges
    corner_size = min(10, width * 2)
    
    # Top-left corner
    if left == 0 or top == 0:
        draw.line([(left, top), (left + corner_size, top)], fill=outline_color, width=width + 1)
        draw.line([(left, top), (left, top + corner_size)], fill=outline_color, width=width + 1)
    
    # Top-right corner  
    if right == img_width or top == 0:
        draw.line([(right - corner_size, top), (right, top)], fill=outline_color, width=width + 1)
        draw.line([(right, top), (right, top + corner_size)], fill=outline_color, width=width + 1)
    
    # Bottom-left corner
    if left == 0 or bottom == img_height:
        draw.line([(left, bottom - corner_size), (left, bottom)], fill=outline_color, width=width + 1)
        draw.line([(left, bottom), (left + corner_size, bottom)], fill=outline_color, width=width + 1)
    
    # Bottom-right corner
    if right == img_width or bottom == img_height:
        draw.line([(right - corner_size, bottom), (right, bottom)], fill=outline_color, width=width + 1)
        draw.line([(right, bottom - corner_size), (right, bottom)], fill=outline_color, width=width + 1)
    
    return new_image


def calculate_square_bbox_from_center(center_row, center_col, patch_size=28, size=3):
    """
    Calculate bounding box coordinates for a square area of patches centered on a given patch.
    
    Args:
        center_row (int): Row index of the center patch
        center_col (int): Column index of the center patch
        patch_size (int): Size of each patch in pixels (default: 28)
        size (int): Size of the square area (e.g., 3 for 3x3, 5 for 5x5) (default: 3)
        
    Returns:
        tuple: (left, top, right, bottom) coordinates of the square bounding box
    """
    # Calculate offset from center to top-left corner
    offset = size // 2
    
    # Calculate top-left patch position
    top_left_row = center_row - offset
    top_left_col = center_col - offset
    
    # Calculate pixel coordinates
    left = top_left_col * patch_size
    top = top_left_row * patch_size
    right = (top_left_col + size) * patch_size
    bottom = (top_left_row + size) * patch_size
    
    return (left, top, right, bottom)

def get_high_confidence_words(annotations, center_row, center_col, size=3, num_words=5):
    """
    Extract nearest neighbor tokens from annotations for the center patch.
    
    Args:
        annotations (list): List of annotation dictionaries, each containing patch info and nearest_neighbors
        center_row (int): Row index of the center patch
        center_col (int): Column index of the center patch
        size (int): Size of the square area (e.g., 3 for 3x3) (default: 3) - kept for API compatibility
        num_words (int): Number of words to return (default: 5)
        
    Returns:
        list: List of dictionaries, each containing token and similarity
    """
    # Direct lookup at center position
    for annotation in annotations:
        if annotation['patch_row'] == center_row and annotation['patch_col'] == center_col:
            return annotation['nearest_neighbors'][:num_words]

    return []


def resize_and_pad_image(image, desired_size=(336, 336), pad_value=0):
    """
    Resize image to fit inside desired_size while preserving aspect ratio,
    then pad to exactly desired_size.
    
    Args:
        image: numpy array (H, W, 3) with values 0-255 (uint8) or 0-1 (float)
        desired_size: tuple (height, width)
        pad_value: value to use for padding (default 0 for black)
    
    Returns:
        padded_image: numpy array (desired_H, desired_W, 3) in range [0, 1]
        image_mask: numpy array (desired_H, desired_W) bool, True=image, False=padding
    """
    import numpy as np
    import torch
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import convert_image_dtype
    
    desired_height, desired_width = desired_size
    height, width = image.shape[:2]
    
    # Calculate scale factor to fit inside box (preserves aspect ratio)
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    
    # New dimensions after scaling
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)
    
    # Resize using torch
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
    image_tensor = convert_image_dtype(image_tensor)  # Convert to float32 [0, 1]
    
    resized = T.Resize(
        [scaled_height, scaled_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )(image_tensor)
    
    resized = torch.clip(resized, 0.0, 1.0)
    resized = resized.permute(1, 2, 0).numpy()  # CHW -> HWC
    
    # Calculate centered padding
    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    bottom_pad = desired_height - scaled_height - top_pad
    right_pad = desired_width - scaled_width - left_pad
    
    # Apply padding
    padding = [
        [top_pad, bottom_pad],
        [left_pad, right_pad],
        [0, 0]  # No padding on channels
    ]
    
    padded_image = np.pad(resized, padding, mode='constant', constant_values=pad_value)
    
    # Create mask (True where real image, False where padded)
    image_mask = np.ones((scaled_height, scaled_width), dtype=bool)
    image_mask = np.pad(image_mask, padding[:2], mode='constant', constant_values=False)
    
    return padded_image, image_mask


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def process_image_with_mask(image_path):
    """
    Process image and return both the processed image and the mask indicating real vs padded areas.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (processed_image, image_mask) where image_mask is True for real image areas
    """
    image = load_image(image_path)
    processed_image, image_mask = resize_and_pad_image(image, (672, 672))
    processed_image = (processed_image * 255).astype(np.uint8)
    processed_image = Image.fromarray(processed_image)
    return processed_image, image_mask


def sample_valid_patch_positions(image_mask, bbox_size=3, num_samples=36):
    """
    Sample random CENTER patch positions where a bbox can fit entirely within the real image area.
    
    Args:
        image_mask (np.ndarray): Boolean mask where True indicates real image areas
        bbox_size (int): Size of the bounding box in patches (e.g., 3 for 3x3)
        num_samples (int): Number of unique center positions to sample
        
    Returns:
        list: List of (center_row, center_col) tuples representing valid center patch positions
    """
    # Convert 672x672 image mask to 24x24 patch grid
    patch_size = 672 // 24  # = 28
    patch_mask = np.zeros((24, 24), dtype=bool)
    
    # Check each patch position to see if it's entirely within the real image
    for row in range(24):
        for col in range(24):
            # Calculate pixel boundaries for this patch
            start_row = row * patch_size
            end_row = min((row + 1) * patch_size, 672)
            start_col = col * patch_size  
            end_col = min((col + 1) * patch_size, 672)
            
            # Check if this patch is entirely within the real image
            patch_area = image_mask[start_row:end_row, start_col:end_col]
            if patch_area.all():  # All pixels in this patch are real (not padded)
                patch_mask[row, col] = True
    
    # Find valid CENTER positions where a bbox_size x bbox_size area can fit entirely in real image
    offset = bbox_size // 2
    valid_center_positions = []
    
    for center_row in range(offset, 24 - offset):
        for center_col in range(offset, 24 - offset):
            # Check if bbox centered at (center_row, center_col) is entirely valid
            top_left_row = center_row - offset
            top_left_col = center_col - offset
            bbox_area = patch_mask[top_left_row:top_left_row+bbox_size, top_left_col:top_left_col+bbox_size]
            if bbox_area.all():  # All patches in this bbox are in real image
                valid_center_positions.append((center_row, center_col))
    
    # Randomly sample from valid center positions
    if len(valid_center_positions) < num_samples:
        print(f"Warning: Only {len(valid_center_positions)} valid center positions found, but {num_samples} requested")
        return valid_center_positions
    
    return random.sample(valid_center_positions, num_samples)


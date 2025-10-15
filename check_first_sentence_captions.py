#!/usr/bin/env python3
"""
Script to check captions processed with first_sentence_only mode.
This will help identify if there are any empty or problematic captions.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from olmo.data.pixmo_datasets import PixMoCap
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_captions():
    """Check captions in first_sentence_only mode."""
    
    print("Loading dataset with first_sentence_only=True...")
    dataset = PixMoCap(
        split="train", 
        mode="captions", 
        first_sentence_only=True,
        keep_in_memory=False
    )
    
    print(f"Dataset loaded. Total examples: {len(dataset)}")
    
    # Statistics
    empty_captions = 0
    very_short_captions = 0  # Less than 5 characters
    short_captions = 0       # Less than 20 characters
    total_captions = 0
    
    # Sample some examples to print
    sample_size = min(100, len(dataset))
    print(f"\nChecking first {sample_size} examples...")
    
    for i in range(sample_size):
        try:
            example = dataset.get(i, rng=None)
            
            # Get the caption from the message list
            caption = None
            for msg in example["message_list"]:
                if msg.get("style") == "long_caption":
                    caption = msg.get("text", "")
                    break
            
            if caption is None:
                print(f"Example {i}: NO CAPTION FOUND")
                continue
                
            total_captions += 1
            
            # Check for empty or very short captions
            if not caption.strip():
                empty_captions += 1
                print(f"Example {i}: EMPTY CAPTION")
            elif len(caption.strip()) < 5:
                very_short_captions += 1
                print(f"Example {i}: VERY SHORT CAPTION: '{caption}'")
            elif len(caption.strip()) < 20:
                short_captions += 1
                print(f"Example {i}: SHORT CAPTION: '{caption}'")
            
            # Print first 10 examples for inspection
            if i < 10:
                print(f"Example {i}: '{caption}'")
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
    
    # Print statistics
    print(f"\n=== STATISTICS (out of {total_captions} captions checked) ===")
    print(f"Empty captions: {empty_captions} ({empty_captions/total_captions*100:.2f}%)")
    print(f"Very short captions (<5 chars): {very_short_captions} ({very_short_captions/total_captions*100:.2f}%)")
    print(f"Short captions (<20 chars): {short_captions} ({short_captions/total_captions*100:.2f}%)")
    
    # Check a few more examples if we found issues
    if empty_captions > 0 or very_short_captions > 0:
        print(f"\nChecking more examples to find problematic ones...")
        for i in range(sample_size, min(sample_size + 50, len(dataset))):
            try:
                example = dataset.get(i, rng=None)
                caption = None
                for msg in example["message_list"]:
                    if msg.get("style") == "long_caption":
                        caption = msg.get("text", "")
                        break
                
                if caption and (not caption.strip() or len(caption.strip()) < 5):
                    print(f"Example {i}: '{caption}'")
                    
            except Exception as e:
                print(f"Error processing example {i}: {e}")

if __name__ == "__main__":
    check_captions()

#!/usr/bin/env python3
"""
Script to analyze caption lengths in the PixMoCap dataset.
Compares captions with and without first_sentence_only processing.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from olmo.data.pixmo_datasets import PixMoCap
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_caption_lengths():
    """Analyze caption lengths with and without first_sentence_only processing."""
    
    print("Loading datasets...")
    
    # Load dataset without first_sentence_only processing
    dataset_full = PixMoCap(
        split="train", 
        mode="captions", 
        first_sentence_only=False,
        keep_in_memory=False
    )
    
    # Load dataset with first_sentence_only processing
    dataset_first_sentence = PixMoCap(
        split="train", 
        mode="captions", 
        first_sentence_only=True,
        keep_in_memory=False
    )
    
    print(f"Dataset loaded. Total examples: {len(dataset_full)}")
    
    # Analyze full captions
    print("\n=== ANALYZING FULL CAPTIONS (first_sentence_only=False) ===")
    full_caption_lengths = []
    full_sentence_counts = []
    full_word_counts = []
    
    sample_size = min(1000, len(dataset_full))
    print(f"Analyzing first {sample_size} examples...")
    
    for i in range(sample_size):
        try:
            example = dataset_full.get(i, rng=None)
            
            # Get the caption from the message list
            caption = None
            for msg in example["message_list"]:
                if msg.get("style") == "long_caption":
                    caption = msg.get("text", "")
                    break
            
            if caption is None:
                continue
                
            # Count sentences (split by periods)
            sentences = [s.strip() for s in caption.split(".") if s.strip()]
            sentence_count = len(sentences)
            
            # Count words
            word_count = len(caption.split())
            
            # Store lengths
            full_caption_lengths.append(len(caption))
            full_sentence_counts.append(sentence_count)
            full_word_counts.append(word_count)
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
    
    # Analyze first-sentence-only captions
    print("\n=== ANALYZING FIRST-SENTENCE-ONLY CAPTIONS (first_sentence_only=True) ===")
    first_sentence_lengths = []
    first_sentence_sentence_counts = []
    first_sentence_word_counts = []
    
    for i in range(sample_size):
        try:
            example = dataset_first_sentence.get(i, rng=None)
            
            # Get the caption from the message list
            caption = None
            for msg in example["message_list"]:
                if msg.get("style") == "long_caption":
                    caption = msg.get("text", "")
                    break
            
            if caption is None:
                continue
                
            # Count sentences (split by periods)
            sentences = [s.strip() for s in caption.split(".") if s.strip()]
            sentence_count = len(sentences)
            
            # Count words
            word_count = len(caption.split())
            
            # Store lengths
            first_sentence_lengths.append(len(caption))
            first_sentence_sentence_counts.append(sentence_count)
            first_sentence_word_counts.append(word_count)
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
    
    # Print statistics
    print(f"\n=== STATISTICS (out of {len(full_caption_lengths)} captions analyzed) ===")
    
    print(f"\nFULL CAPTIONS:")
    print(f"  Average character length: {np.mean(full_caption_lengths):.1f} ± {np.std(full_caption_lengths):.1f}")
    print(f"  Average word count: {np.mean(full_word_counts):.1f} ± {np.std(full_word_counts):.1f}")
    print(f"  Average sentence count: {np.mean(full_sentence_counts):.1f} ± {np.std(full_sentence_counts):.1f}")
    print(f"  Median sentence count: {np.median(full_sentence_counts):.1f}")
    print(f"  Min sentence count: {np.min(full_sentence_counts)}")
    print(f"  Max sentence count: {np.max(full_sentence_counts)}")
    
    print(f"\nFIRST-SENTENCE-ONLY CAPTIONS:")
    print(f"  Average character length: {np.mean(first_sentence_lengths):.1f} ± {np.std(first_sentence_lengths):.1f}")
    print(f"  Average word count: {np.mean(first_sentence_word_counts):.1f} ± {np.std(first_sentence_word_counts):.1f}")
    print(f"  Average sentence count: {np.mean(first_sentence_sentence_counts):.1f} ± {np.std(first_sentence_sentence_counts):.1f}")
    print(f"  Median sentence count: {np.median(first_sentence_sentence_counts):.1f}")
    print(f"  Min sentence count: {np.min(first_sentence_sentence_counts)}")
    print(f"  Max sentence count: {np.max(first_sentence_sentence_counts)}")
    
    # Show distribution of sentence counts
    print(f"\n=== SENTENCE COUNT DISTRIBUTION ===")
    print(f"FULL CAPTIONS:")
    unique_counts, counts = np.unique(full_sentence_counts, return_counts=True)
    for count, freq in zip(unique_counts, counts):
        percentage = (freq / len(full_sentence_counts)) * 100
        print(f"  {count} sentences: {freq} captions ({percentage:.1f}%)")
    
    print(f"\nFIRST-SENTENCE-ONLY CAPTIONS:")
    unique_counts, counts = np.unique(first_sentence_sentence_counts, return_counts=True)
    for count, freq in zip(unique_counts, counts):
        percentage = (freq / len(first_sentence_sentence_counts)) * 100
        print(f"  {count} sentences: {freq} captions ({percentage:.1f}%)")
    
    # Show some examples
    print(f"\n=== EXAMPLE CAPTIONS ===")
    print(f"FULL CAPTIONS (showing first 5):")
    for i in range(min(5, len(full_caption_lengths))):
        example = dataset_full.get(i, rng=None)
        caption = None
        for msg in example["message_list"]:
            if msg.get("style") == "long_caption":
                caption = msg.get("text", "")
                break
        if caption:
            sentences = [s.strip() for s in caption.split(".") if s.strip()]
            print(f"  Example {i}: {len(sentences)} sentences, {len(caption.split())} words")
            print(f"    Text: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    
    print(f"\nFIRST-SENTENCE-ONLY CAPTIONS (showing first 5):")
    for i in range(min(5, len(first_sentence_sentence_counts))):
        example = dataset_first_sentence.get(i, rng=None)
        caption = None
        for msg in example["message_list"]:
            if msg.get("style") == "long_caption":
                caption = msg.get("text", "")
                break
        if caption:
            sentences = [s.strip() for s in caption.split(".") if s.strip()]
            print(f"  Example {i}: {len(sentences)} sentences, {len(caption.split())} words")
            print(f"    Text: {caption[:100]}{'...' if len(caption) > 100 else ''}")

if __name__ == "__main__":
    analyze_caption_lengths()

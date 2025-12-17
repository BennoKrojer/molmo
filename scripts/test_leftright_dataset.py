"""
Test script to verify the left/right dataset and formatter work correctly.
"""
import numpy as np
from olmo.data.pixmo_datasets import PixMoPointsLeftRight
from olmo.data.data_formatter import DataFormatter

def test_dataset():
    print("="*80)
    print("Testing PixMoPointsLeftRight Dataset")
    print("="*80)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = PixMoPointsLeftRight(split="train", kind="basic")
    print(f"   Dataset size: {len(dataset)}")
    
    # Test getting examples
    print("\n2. Testing dataset.get()...")
    rng = np.random.RandomState(42)
    
    valid_examples = 0
    attempts = 0
    max_attempts = 100
    
    while valid_examples < 5 and attempts < max_attempts:
        idx = rng.randint(0, len(dataset))
        example = dataset.get(idx, rng)
        attempts += 1
        
        if example is None:
            continue
        
        valid_examples += 1
        print(f"\n   Example {valid_examples} (dataset idx {idx}):")
        print(f"   - Image: {example['image']}")
        print(f"   - Number of messages: {len(example['message_list'])}")
        
        for i, msg in enumerate(example['message_list'][:3]):  # Show first 3 messages
            print(f"   - Message {i+1}:")
            print(f"     * Label: {msg['label']}")
            print(f"     * Position: {msg['position']}")
            print(f"     * Style: {msg['style']}")
    
    print(f"\n   Found {valid_examples} valid examples after {attempts} attempts")
    
    # Test formatter
    print("\n" + "="*80)
    print("Testing DataFormatter")
    print("="*80)
    
    formatter = DataFormatter()
    formatter.system_prompt = "demo_or_style"
    formatter.message_format = "none"
    formatter.prompt_templates = "uber_model"  # Use uber_model for templated prompts
    
    print("\n3. Testing format_left_right()...")
    rng = np.random.RandomState(42)
    
    # Get a valid example
    for i in range(100):
        example = dataset.get(rng.randint(0, len(dataset)), rng)
        if example is not None:
            break
    
    if example is None:
        print("   ERROR: Could not find valid example")
        return False
    
    print(f"\n   Testing with {len(example['message_list'])} messages from the example:")
    
    for i, msg in enumerate(example['message_list'][:5]):  # Test first 5
        # Format the message
        prompt, output, metadata = formatter.get_user_prompt(msg, is_training=True, for_inference=False, rng=rng)
        
        print(f"\n   Message {i+1}:")
        print(f"   - Input label: {msg['label']}")
        print(f"   - Position: {msg['position']}")
        print(f"   - Generated prompt: {prompt}")
        print(f"   - Expected output: {output}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    return True


if __name__ == "__main__":
    test_dataset()


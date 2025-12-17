"""
Inspect Qwen2-VL model structure to find vision encoder and understand architecture.

This script loads the model and prints its structure so we can figure out how to extract vision features.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

def inspect_model_structure(model_name="Qwen/Qwen2-VL-7B-Instruct"):
    """Inspect the model structure to find vision encoder."""
    
    print(f"Loading model: {model_name}")
    print("="*80)
    
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU first for inspection
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    print("\n1. MODEL TOP-LEVEL ATTRIBUTES:")
    print("-" * 80)
    for attr in dir(model):
        if not attr.startswith('_'):
            try:
                obj = getattr(model, attr)
                if not callable(obj) or isinstance(obj, torch.nn.Module):
                    print(f"  {attr}: {type(obj)}")
            except:
                pass
    
    print("\n2. MODEL CONFIG:")
    print("-" * 80)
    print(f"  Config type: {type(model.config)}")
    print(f"  Config attributes: {dir(model.config)}")
    
    # Check for vision-related config
    if hasattr(model.config, 'vision_config'):
        print(f"\n  Vision config: {model.config.vision_config}")
        print(f"  Vision config type: {type(model.config.vision_config)}")
        if hasattr(model.config.vision_config, '__dict__'):
            print(f"  Vision config attributes: {model.config.vision_config.__dict__}")
    
    # Check for model.model structure
    print("\n3. MODEL.MODEL STRUCTURE (if exists):")
    print("-" * 80)
    if hasattr(model, 'model'):
        print(f"  model.model type: {type(model.model)}")
        print(f"  model.model attributes:")
        for attr in dir(model.model):
            if not attr.startswith('_'):
                try:
                    obj = getattr(model.model, attr)
                    if not callable(obj) or isinstance(obj, torch.nn.Module):
                        print(f"    {attr}: {type(obj)}")
                except:
                    pass
    
    # Check for vision_model
    print("\n4. VISION MODEL ACCESS:")
    print("-" * 80)
    if hasattr(model, 'vision_model'):
        print(f"  ✓ model.vision_model exists: {type(model.vision_model)}")
        vision_model = model.vision_model
        print(f"    Attributes: {[a for a in dir(vision_model) if not a.startswith('_')]}")
    else:
        print("  ✗ model.vision_model does NOT exist")
    
    if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        print(f"  ✓ model.model.vision_model exists: {type(model.model.vision_model)}")
        vision_model = model.model.vision_model
        print(f"    Attributes: {[a for a in dir(vision_model) if not a.startswith('_')]}")
    else:
        print("  ✗ model.model.vision_model does NOT exist")
    
    # Try to find vision-related modules recursively
    print("\n5. SEARCHING FOR VISION-RELATED MODULES:")
    print("-" * 80)
    def find_vision_modules(module, prefix="", depth=0, max_depth=3):
        if depth > max_depth:
            return
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if 'vision' in name.lower() or 'image' in name.lower() or 'visual' in name.lower():
                print(f"  Found: {full_name} ({type(child)})")
            find_vision_modules(child, full_name, depth+1, max_depth)
    
    find_vision_modules(model)
    
    # Try a forward pass to see what happens
    print("\n6. TESTING FORWARD PASS:")
    print("-" * 80)
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    try:
        # Prepare inputs
        model_inputs = processor(images=[dummy_image], text="test", return_tensors="pt")
        print(f"  Processor output keys: {model_inputs.keys()}")
        
        # Move model to CPU and set to eval
        model.eval()
        
        # Try forward pass with output_hidden_states
        print("\n  Attempting forward pass with output_hidden_states=True...")
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
        
        print(f"  Output type: {type(outputs)}")
        print(f"  Output attributes: {[a for a in dir(outputs) if not a.startswith('_')]}")
        
        if hasattr(outputs, 'hidden_states'):
            print(f"  ✓ hidden_states available: {len(outputs.hidden_states)} layers")
            if len(outputs.hidden_states) > 0:
                print(f"    First hidden state shape: {outputs.hidden_states[0].shape}")
                print(f"    Last hidden state shape: {outputs.hidden_states[-1].shape}")
        
        # Check if we can access vision encoder outputs
        print("\n  Checking for vision encoder outputs...")
        
        # Try accessing vision model directly if it exists
        if hasattr(model, 'vision_model'):
            print("  Trying model.vision_model...")
            pixel_values = model_inputs.get('pixel_values', None)
            if pixel_values is not None:
                vision_outputs = model.vision_model(pixel_values=pixel_values)
                print(f"    Vision outputs type: {type(vision_outputs)}")
                print(f"    Vision outputs attributes: {[a for a in dir(vision_outputs) if not a.startswith('_')]}")
                if hasattr(vision_outputs, 'last_hidden_state'):
                    print(f"    ✓ last_hidden_state shape: {vision_outputs.last_hidden_state.shape}")
        
        if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
            print("  Trying model.model.vision_model...")
            pixel_values = model_inputs.get('pixel_values', None)
            if pixel_values is not None:
                vision_outputs = model.model.vision_model(pixel_values=pixel_values)
                print(f"    Vision outputs type: {type(vision_outputs)}")
                print(f"    Vision outputs attributes: {[a for a in dir(vision_outputs) if not a.startswith('_')]}")
                if hasattr(vision_outputs, 'last_hidden_state'):
                    print(f"    ✓ last_hidden_state shape: {vision_outputs.last_hidden_state.shape}")
        
        # Check input_ids to see if we can identify visual token positions
        if 'input_ids' in model_inputs:
            input_ids = model_inputs['input_ids']
            print(f"\n  Input IDs shape: {input_ids.shape}")
            print(f"  Input IDs (first 50): {input_ids[0][:50].tolist()}")
            
            # Try to decode to see what tokens are at the beginning
            tokenizer = processor.tokenizer
            first_tokens = input_ids[0][:20]
            decoded = tokenizer.decode(first_tokens, skip_special_tokens=False)
            print(f"  First 20 tokens decoded: {decoded}")
        
    except Exception as e:
        print(f"  ✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="Model name to inspect")
    args = parser.parse_args()
    
    inspect_model_structure(args.model_name)


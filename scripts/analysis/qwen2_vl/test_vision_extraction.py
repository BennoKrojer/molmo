"""
Test script to verify what model.visual returns and if it exposes hidden states.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def test_vision_encoder():
    """Test what model.visual returns."""
    
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Prepare inputs
    model_inputs = processor(images=[dummy_image], text="test", return_tensors="pt")
    pixel_values = model_inputs.get('pixel_values', None)
    
    print(f"\nPixel values shape: {pixel_values.shape if pixel_values is not None else None}")
    print(f"Model.visual type: {type(model.visual)}")
    print(f"Model.visual attributes: {[a for a in dir(model.visual) if not a.startswith('_') and not callable(getattr(model.visual, a, None))]}")
    
    # Test 1: Call model.visual directly
    print("\n" + "="*80)
    print("TEST 1: Calling model.visual(pixel_values=...)")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        try:
            vision_outputs = model.visual(pixel_values=pixel_values)
            print(f"✓ Success! Output type: {type(vision_outputs)}")
            print(f"  Output attributes: {[a for a in dir(vision_outputs) if not a.startswith('_')]}")
            
            if hasattr(vision_outputs, 'last_hidden_state'):
                print(f"  ✓ Has last_hidden_state: {vision_outputs.last_hidden_state.shape}")
            if hasattr(vision_outputs, 'hidden_states'):
                print(f"  ✓ Has hidden_states: {len(vision_outputs.hidden_states) if vision_outputs.hidden_states else 0} layers")
                if vision_outputs.hidden_states:
                    print(f"    First layer shape: {vision_outputs.hidden_states[0].shape}")
                    print(f"    Last layer shape: {vision_outputs.hidden_states[-1].shape}")
            if isinstance(vision_outputs, torch.Tensor):
                print(f"  ✓ Returns tensor directly: {vision_outputs.shape}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 2: Try with output_hidden_states parameter
    print("\n" + "="*80)
    print("TEST 2: Calling model.visual with output_hidden_states=True")
    print("="*80)
    
    with torch.no_grad():
        try:
            # Check if visual accepts output_hidden_states
            if hasattr(model.visual, 'forward'):
                # Try to inspect forward signature
                import inspect
                sig = inspect.signature(model.visual.forward)
                print(f"  Forward signature parameters: {list(sig.parameters.keys())}")
            
            vision_outputs = model.visual(pixel_values=pixel_values, output_hidden_states=True)
            print(f"✓ Success with output_hidden_states=True!")
            print(f"  Output type: {type(vision_outputs)}")
            
            if hasattr(vision_outputs, 'hidden_states'):
                print(f"  ✓ Has hidden_states: {len(vision_outputs.hidden_states) if vision_outputs.hidden_states else 0} layers")
            
        except TypeError as e:
            print(f"  Note: output_hidden_states parameter not accepted (this is OK)")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Check if we can access intermediate layers
    print("\n" + "="*80)
    print("TEST 3: Inspecting model.visual structure for layer access")
    print("="*80)
    
    if hasattr(model.visual, 'encoder'):
        print(f"  ✓ model.visual.encoder exists: {type(model.visual.encoder)}")
        if hasattr(model.visual.encoder, 'layers'):
            print(f"    ✓ Has layers: {len(model.visual.encoder.layers)} layers")
    if hasattr(model.visual, 'layers'):
        print(f"  ✓ model.visual.layers exists: {len(model.visual.layers)} layers")
    
    # Test 4: Try hook-based extraction
    print("\n" + "="*80)
    print("TEST 4: Testing hook-based extraction (if direct access fails)")
    print("="*80)
    
    captured_features = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            captured_features.append(output)
        elif hasattr(output, 'last_hidden_state'):
            captured_features.append(output.last_hidden_state)
        elif hasattr(output, 'hidden_states') and output.hidden_states:
            captured_features.append(output.hidden_states[-1])
    
    # Register hook on the last layer of vision encoder
    if hasattr(model.visual, 'encoder') and hasattr(model.visual.encoder, 'layers'):
        hook_handle = model.visual.encoder.layers[-1].register_forward_hook(hook_fn)
        print("  Registered hook on last encoder layer")
        
        with torch.no_grad():
            _ = model.visual(pixel_values=pixel_values)
        
        if captured_features:
            print(f"  ✓ Hook captured features: {captured_features[0].shape}")
        else:
            print("  ✗ Hook did not capture features")
        
        hook_handle.remove()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Based on these tests, we can determine:")
    print("1. What model.visual returns")
    print("2. Whether hidden states are accessible")
    print("3. Best method to extract vision features")


if __name__ == "__main__":
    test_vision_encoder()


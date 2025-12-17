"""
Find the actual vision encoder that processes pixel_values.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def find_vision_encoder():
    """Find what actually processes pixel_values."""
    
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    
    dummy_image = Image.new('RGB', (224, 224), color='red')
    model_inputs = processor(images=[dummy_image], text="test", return_tensors="pt")
    pixel_values = model_inputs.get('pixel_values', None)
    
    print(f"\nPixel values shape: {pixel_values.shape}")
    print(f"Model input keys: {model_inputs.keys()}")
    
    # Search for modules that might process pixel_values
    print("\n" + "="*80)
    print("SEARCHING FOR VISION ENCODER")
    print("="*80)
    
    def search_for_vision_modules(module, prefix="", depth=0, max_depth=4):
        """Recursively search for vision-related modules."""
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module might process images
            if any(keyword in name.lower() for keyword in ['vision', 'image', 'visual', 'patch', 'embed', 'conv']):
                print(f"\n  Found: {full_name}")
                print(f"    Type: {type(child)}")
                
                # Try to see forward signature
                if hasattr(child, 'forward'):
                    try:
                        import inspect
                        sig = inspect.signature(child.forward)
                        params = list(sig.parameters.keys())
                        print(f"    Forward params: {params}")
                        
                        # Check if it accepts pixel_values
                        if 'pixel_values' in params:
                            print(f"    *** ACCEPTS pixel_values! ***")
                    except:
                        pass
            
            # Recurse
            search_for_vision_modules(child, full_name, depth+1, max_depth)
    
    print("\nSearching model structure...")
    search_for_vision_modules(model)
    
    # Check model.model structure more carefully
    print("\n" + "="*80)
    print("CHECKING model.model STRUCTURE")
    print("="*80)
    
    if hasattr(model, 'model'):
        print(f"model.model type: {type(model.model)}")
        print(f"model.model children:")
        for name, child in model.model.named_children():
            print(f"  {name}: {type(child)}")
    
    # Try to trace through the forward pass
    print("\n" + "="*80)
    print("ATTEMPTING TO TRACE FORWARD PASS")
    print("="*80)
    
    # Register hooks to see what gets called
    called_modules = []
    
    def trace_hook(name):
        def hook(module, input, output):
            called_modules.append(name)
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    print(f"  {name}: input shape {input[0].shape}")
            if isinstance(output, torch.Tensor):
                print(f"  {name}: output shape {output.shape}")
            elif hasattr(output, 'last_hidden_state'):
                print(f"  {name}: output.last_hidden_state shape {output.last_hidden_state.shape}")
        return hook
    
    # Register hooks on potential vision modules
    hooks = []
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['vision', 'image', 'visual', 'patch', 'embed']):
            hook = module.register_forward_hook(trace_hook(name))
            hooks.append(hook)
            print(f"Registered hook on: {name}")
    
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(**model_inputs, output_hidden_states=True)
            print(f"\nForward pass completed!")
            print(f"Output type: {type(outputs)}")
            if hasattr(outputs, 'hidden_states'):
                print(f"Hidden states: {len(outputs.hidden_states)} layers")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*80)
    print("MODULES CALLED DURING FORWARD PASS")
    print("="*80)
    for name in called_modules[:20]:  # Show first 20
        print(f"  {name}")


if __name__ == "__main__":
    find_vision_encoder()


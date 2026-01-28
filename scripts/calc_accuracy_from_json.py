"""Quick script to calculate accuracy from existing generated_captions.json"""
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/calc_accuracy_from_json.py <path_to_generated_captions.json>")
    sys.exit(1)

json_path = Path(sys.argv[1])

with open(json_path) as f:
    data = json.load(f)

task_type = data["task_type"]
outputs = data["outputs"]

print(f"\nTask type: {task_type}")
print(f"Total outputs: {len(outputs)}")

# For top_bottom: manually extract position from generated text
if task_type == "top_bottom":
    correct = 0
    total = 0
    top_count = 0
    bottom_count = 0
    
    for entry in outputs:
        generated = entry["generated_output"].lower()
        total += 1
        
        # Heuristic: check if "top" or "bottom" appears in generated text
        has_top = "top" in generated
        has_bottom = "bottom" in generated
        
        # Count what was generated (for distribution)
        if has_top and not has_bottom:
            top_count += 1
        elif has_bottom and not has_top:
            bottom_count += 1
        
        # Since we don't have ground truth, we can't calculate accuracy
        # But we can show what the model is predicting
    
    print(f"\n{'='*60}")
    print(f"Generated predictions:")
    print(f"  Top: {top_count}/{total} ({top_count/total*100:.1f}%)")
    print(f"  Bottom: {bottom_count}/{total} ({bottom_count/total*100:.1f}%)")
    print(f"  Ambiguous/Neither: {total - top_count - bottom_count}/{total}")
    print(f"{'='*60}")
    
    print("\n⚠️  Cannot calculate accuracy - JSON missing 'position' field for ground truth!")
    print("Need to re-run with fixed script to get position metadata.\n")
    
    # Show a few examples
    print("Sample outputs:")
    for i, entry in enumerate(outputs[:10]):
        print(f"  {i}: {entry['generated_output']}")

else:
    print(f"Task type '{task_type}' not supported yet in this quick script")


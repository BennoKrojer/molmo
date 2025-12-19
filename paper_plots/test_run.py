#!/usr/bin/env python3
import sys
print("Python is working!")
print(f"Python version: {sys.version}")

# Test importing the module
try:
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).parent
    OUTPUT_DIR = SCRIPT_DIR / "paper_figures_output" / "interpretability"
    print(f"Output dir would be: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Created output directory!")
    
    # Write a test file
    test_file = OUTPUT_DIR / "test.txt"
    test_file.write_text("Test successful!")
    print(f"Wrote test file to: {test_file}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()



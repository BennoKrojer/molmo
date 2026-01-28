# Lessons from Qwen2-VL Preprocessing Bug

## What Went Wrong
- Updated 3 scripts to use shared preprocessing, missed 1 script
- `run_llm_judge.py` had duplicate preprocessing code that wasn't updated
- No type system to catch missing parameters
- Silent failures (model_name=None defaulted to wrong preprocessing)
- Poor logging (skip reasons hidden behind --debug flag)

## Root Causes
1. **Code duplication** across 4 scripts
2. **No type checking** (Python dynamic typing)
3. **Silent failures** (no validation)
4. **Hidden skip reasons** (debug logging)
5. **No tests** (no automation to catch mismatches)

## Preventive Measures Going Forward

### 1. Documentation
- ✅ Created `scripts/analysis/qwen2_vl/preprocessing.py` (SINGLE SOURCE OF TRUTH)
- ✅ Added clear comments marking it as canonical
- ⚠️ TODO: Add docstring explaining why it exists and what calls it

### 2. Code Practices
- ⚠️ TODO: Add validation in utils.py:
  ```python
  def process_image_with_mask(image_path, model_name=None):
      if model_name is None:
          warnings.warn("model_name not provided, using default preprocessing")
  ```

- ⚠️ TODO: Add assertion in shared preprocessing:
  ```python
  def preprocess_image_qwen2vl(image, target_size=448, force_square=True):
      result = ...
      # Validate output
      assert result.size == (target_size, target_size), "Preprocessing failed!"
      return result
  ```

### 3. Logging Improvements
- ⚠️ TODO: Always print skip statistics (not just in debug mode)
- ⚠️ TODO: Add warning when images are skipped
- ⚠️ TODO: Summary at end showing:  
  - Images processed
  - Images skipped (with reasons)
  - Patches evaluated
  - Patches skipped (with reasons)

### 4. Error Throwing
- ⚠️ TODO: Fail loudly when model_name missing for Qwen2-VL:
  ```python
  if "qwen2" in checkpoint_path.lower() and model_name is None:
      raise ValueError("model_name required for Qwen2-VL!")
  ```

### 5. Testing (Ideal but time-consuming)
- ⚠️ TODO: Unit test that preprocessing.py produces expected output
- ⚠️ TODO: Integration test that all scripts use same preprocessing
- ⚠️ TODO: Validation script that checks:
  - All scripts import from preprocessing.py (not duplicate code)
  - Sample image produces same result across all scripts

### 6. Refactoring (Long-term)
- ⚠️ TODO: Consolidate 4 duplicate scripts into 1
- ⚠️ TODO: Extract image processing to shared module
- ⚠️ TODO: Use dataclasses/TypedDict for configs (catch missing fields)

## Quick Wins (Do These Now)

1. Add skip statistics logging (remove debug flag requirement)
2. Add validation in preprocessing module
3. Add warning when model_name is None
4. Document that preprocessing.py is canonical source

## Why We Haven't Done This Yet
- Research code prioritizes speed over robustness
- Refactoring takes time away from experiments
- No automated testing infrastructure
- Python's dynamic typing enables quick prototyping but hides errors

## Key Takeaway
**Trade-off between velocity and reliability**: Research code is optimized for quick iteration, 
but this leads to tech debt. Each "quick fix" adds complexity. We need systematic refactoring 
sessions to clean up, not just patch symptoms.

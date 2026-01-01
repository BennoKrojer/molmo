# Schema Standardization Plan

## Executive Summary

This document defines a unified schema for all interpretability analysis outputs and identifies changes needed to achieve consistency across the codebase.

---

## Current State (Inconsistencies)

### 1. Folder Structure

| Lens | Current Folder | Issues |
|------|---------------|--------|
| Embedding Matrix | `nearest_neighbors/` | OK |
| LogitLens | `logit_lens/` | OK |
| LN-Lens | `contextual_nearest_neighbors/` | Was also `_vg/`, `_cc/`, `_visual0/`, `_visualN/` |

### 2. File Naming

| Lens | Current Pattern | Issues |
|------|----------------|--------|
| Embedding Matrix (Molmo) | `nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{N}.json` | Long, includes dataset name |
| Embedding Matrix (Qwen2-VL) | `nearest_neighbors_layer{N}_topk5.json` | Different pattern! |
| LogitLens | `logit_lens_layer{N}_topk5_multi-gpu.json` | OK |
| LN-Lens | `contextual_neighbors_visual{V}_allLayers.json` | Different structure (all ctx layers in one file) |

### 3. JSON Structure

#### Embedding Matrix (Molmo checkpoints)
```json
{
  "checkpoint": "...",
  "splits": {
    "validation": {
      "images": [
        {
          "image_idx": 0,
          "chunks": [{"patches": [...]}]  // nested in chunks!
        }
      ]
    }
  }
}
```

#### Embedding Matrix (Qwen2-VL)
```json
{
  "model_name": "...",
  "results": [
    {
      "image_idx": 0,
      "patches": [...]  // flat, no chunks!
    }
  ]
}
```

#### LogitLens
```json
{
  "checkpoint": "...",
  "results": [
    {
      "image_idx": 0,
      "chunks": [{"patches": [...]}]
    }
  ]
}
```

#### LN-Lens
```json
{
  "checkpoint": "...",
  "visual_layer": 0,
  "contextual_layers_used": [1, 2, 4, 8, ...],
  "results": [
    {
      "image_idx": 0,
      "chunks": [{"patches": [...]}]
    }
  ]
}
```

### 4. Patch Key Names

| Lens | Current Key | Contains |
|------|------------|----------|
| Embedding Matrix (Molmo) | `nearest_neighbors` | `[{token, similarity, token_id}]` |
| Embedding Matrix (Qwen2-VL) | `top_neighbors` | `[{token, similarity, token_id}]` |
| LogitLens | `top_predictions` | `[{token, logit, token_id}]` |
| LN-Lens | `nearest_contextual_neighbors` | `[{token_str, similarity, caption, contextual_layer}]` |

### 5. Neighbor/Prediction Fields

| Lens | Token Key | Score Key | Extra Fields |
|------|-----------|-----------|--------------|
| Embedding Matrix | `token` | `similarity` | `token_id` |
| LogitLens | `token` | `logit` | `token_id` |
| LN-Lens | `token_str` | `similarity` | `caption`, `position`, `contextual_layer` |

---

## Target Schema (Unified)

### Folder Structure
```
analysis_results/
├── embedding_matrix/           # Renamed from nearest_neighbors
│   └── {model}/
│       └── layer_{N}.json
├── logitlens/                  # Keep as is
│   └── {model}/
│       └── layer_{N}.json
└── ln_lens/                    # Renamed from contextual_nearest_neighbors
    └── {model}/
        └── layer_{N}.json      # One file per layer
```

### File Naming
```
layer_{N}.json                  # Simple, consistent
```

### JSON Structure (Unified)
```json
{
  "metadata": {
    "model": "train_mlp-only_pixmo_cap_resize_llama3-8b_vit-l-14-336",
    "layer": 8,
    "analysis_type": "embedding_matrix",  // or "logitlens" or "ln_lens"
    "created_at": "2024-01-02T12:00:00Z",
    "num_images": 100
  },
  "results": [
    {
      "image_idx": 0,
      "ground_truth_caption": "A photo of...",
      "grid_size": [24, 24],
      "patches": [
        {
          "patch_idx": 0,
          "row": 0,
          "col": 0,
          "neighbors": [
            {
              "rank": 1,
              "token": "sky",
              "score": 0.85,
              "token_id": 1234,
              // LN-Lens specific (optional):
              "caption": "blue sky with clouds",
              "position": 3
            }
          ]
        }
      ]
    }
  ]
}
```

### Key Changes
1. **Unified key**: `neighbors` (not `nearest_neighbors`, `top_neighbors`, `top_predictions`, etc.)
2. **Unified score key**: `score` (not `similarity`, `logit`)
3. **Flat patches**: No `chunks` wrapper (simplifies processing)
4. **Explicit metadata**: `analysis_type`, `layer`, `model` always present
5. **Consistent grid info**: `grid_size` as `[rows, cols]`

---

## Migration Plan

### Phase 1: Viewer Adapter (DONE partially)
Already have translation in `create_unified_viewer.py`. Complete the adapter to handle all formats.

### Phase 2: Update Generation Scripts

| Script | Output | Changes Needed |
|--------|--------|----------------|
| `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` | Embedding Matrix (Molmo) | Use unified schema |
| `scripts/analysis/qwen2_vl/nearest_neighbors.py` | Embedding Matrix (Qwen2-VL) | Use unified schema |
| `logitlens.py` | LogitLens (Molmo) | Use unified schema |
| `scripts/analysis/qwen2_vl/logitlens.py` | LogitLens (Qwen2-VL) | Use unified schema |
| `contextual_nearest_neighbors_allLayers_singleGPU.py` | LN-Lens (Molmo) | Use unified schema |
| `scripts/analysis/qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py` | LN-Lens (Qwen2-VL) | Use unified schema |

### Phase 3: Regenerate Results
After updating scripts, regenerate all results with the new schema.

### Phase 4: Simplify Viewer
With unified schema, remove all adapter/translation code.

---

## Scripts Inventory

### Active (Main Pipeline)

| Script | Purpose | Status |
|--------|---------|--------|
| `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` | Embedding Matrix for Molmo | ACTIVE - needs update |
| `logitlens.py` | LogitLens for Molmo | ACTIVE - needs update |
| `contextual_nearest_neighbors_allLayers_singleGPU.py` | LN-Lens for Molmo | ACTIVE - needs update |
| `create_unified_viewer.py` | HTML viewer generation | ACTIVE - has adapters |
| `generate_ablation_viewers.py` | Ablation HTML viewers | ACTIVE - imports from unified |

### Active (Qwen2-VL)

| Script | Purpose | Status |
|--------|---------|--------|
| `qwen2_vl/nearest_neighbors.py` | Embedding Matrix | ACTIVE - needs update |
| `qwen2_vl/logitlens.py` | LogitLens | ACTIVE - needs update |
| `qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py` | LN-Lens | ACTIVE - needs update |

### Legacy (Consider Deprecating)

| Script | Purpose | Recommendation |
|--------|---------|----------------|
| `general_and_nearest_neighbors.py` | Old NN script | DEPRECATE |
| `general_and_nearest_neighbors_pixmo_cap.py` | Single-GPU version | DEPRECATE (use multi-gpu) |
| `contextual_nearest_neighbors.py` | Old contextual NN | DEPRECATE |
| `interactive_*_viewer.py` (all) | Old interactive viewers | DEPRECATE (use unified) |

---

## Implementation Priority

1. **HIGH**: Fix Qwen2-VL `top_neighbors` → `nearest_neighbors` (or unified `neighbors`)
2. **MEDIUM**: Flatten `chunks` → `patches` in all outputs
3. **MEDIUM**: Rename folders to simpler names
4. **LOW**: Rename files to `layer_{N}.json`

---

## Backward Compatibility

During transition, the viewer should support both old and new formats:

```python
def get_neighbors(patch):
    """Get neighbors from patch, handling all format variants."""
    for key in ['neighbors', 'nearest_neighbors', 'top_neighbors', 
                'top_predictions', 'nearest_contextual_neighbors']:
        if key in patch:
            return patch[key]
    return []

def get_score(neighbor):
    """Get score from neighbor, handling all format variants."""
    for key in ['score', 'similarity', 'logit']:
        if key in neighbor:
            return neighbor[key]
    return 0.0
```

---

## Appendix: Display Names

| Internal Key | Display Name (UI) |
|--------------|-------------------|
| `embedding_matrix` / `nn` | Embedding Matrix |
| `logitlens` | LogitLens |
| `ln_lens` / `contextual` | LN-Lens |


# V-Lens Schema & Naming Standardization

Single source of truth for data formats, naming conventions, and migration plans.

---

## Quick Reference

### Display Names (Paper/UI)
| Internal Code | Display Name |
|---------------|--------------|
| `nn` / `embedding_matrix` | **Embedding Matrix** |
| `logitlens` | **LogitLens** |
| `contextual` / `ln_lens` | **LN-Lens** |

### Primary Scripts
| Purpose | Script |
|---------|--------|
| Embedding Matrix (Molmo) | `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` |
| Embedding Matrix (Qwen2-VL) | `qwen2_vl/nearest_neighbors.py` |
| LogitLens (Molmo) | `logitlens.py` |
| LogitLens (Qwen2-VL) | `qwen2_vl/logitlens.py` |
| LN-Lens (Molmo) | `contextual_nearest_neighbors_allLayers_singleGPU.py` |
| LN-Lens (Qwen2-VL) | `qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py` |
| Viewer Generation | `create_unified_viewer.py` |
| Ablation Viewer | `generate_ablation_viewers.py` |

### Legacy Scripts (Deprecate)
- `general_and_nearest_neighbors.py`, `general_and_nearest_neighbors_pixmo_cap.py`
- `contextual_nearest_neighbors.py`
- All `interactive_*_viewer.py`

---

## Current Inconsistencies

### 1. Folder Names
| Lens | Current |
|------|---------|
| Embedding Matrix | `nearest_neighbors/` |
| LogitLens | `logit_lens/` |
| LN-Lens | `contextual_nearest_neighbors/` (was `_vg/`, `_cc/`) |

### 2. File Names
| Lens | Pattern |
|------|---------|
| Embedding Matrix (Molmo) | `nearest_neighbors_analysis_pixmo_cap_multi-gpu_layer{N}.json` |
| Embedding Matrix (Qwen2-VL) | `nearest_neighbors_layer{N}_topk5.json` |
| LogitLens | `logit_lens_layer{N}_topk5_multi-gpu.json` |
| LN-Lens | `contextual_neighbors_visual{V}_allLayers.json` |

### 3. JSON Structure
| Model | Top-level | Image data |
|-------|-----------|------------|
| Molmo NN | `splits.validation.images` | `chunks.patches` |
| Qwen2-VL NN | `results` | `patches` (no chunks!) |
| LogitLens | `results` | `chunks.patches` |
| LN-Lens | `results` | `chunks.patches` |

### 4. Patch Keys (THE MAIN ISSUE)
| Lens | Key |
|------|-----|
| Embedding Matrix (Molmo) | `nearest_neighbors` |
| Embedding Matrix (Qwen2-VL) | `top_neighbors` ← **Different!** |
| LogitLens | `top_predictions` |
| LN-Lens | `nearest_contextual_neighbors` |

### 5. Neighbor Fields
| Lens | Token | Score |
|------|-------|-------|
| Embedding Matrix | `token` | `similarity` |
| LogitLens | `token` | `logit` |
| LN-Lens | `token_str` | `similarity` |

---

## Target Schema (Unified)

```json
{
  "metadata": {
    "model": "...",
    "layer": 8,
    "analysis_type": "embedding_matrix|logitlens|ln_lens"
  },
  "results": [
    {
      "image_idx": 0,
      "grid_size": [24, 24],
      "patches": [
        {
          "patch_idx": 0, "row": 0, "col": 0,
          "neighbors": [{"rank": 1, "token": "sky", "score": 0.85}]
        }
      ]
    }
  ]
}
```

Key principles:
- Unified `neighbors` key
- Unified `score` key  
- Flat `patches` (no `chunks`)
- Explicit `metadata`

---

## Migration Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Viewer display names (Embedding Matrix, LN-Lens) | ✅ Done |
| 2 | Drop `_vg` suffix from folders | ✅ Done |
| 3 | Fix Qwen2-VL `top_neighbors` → `nearest_neighbors` | 🔲 TODO |
| 4 | Legacy script cleanup | 🔲 Optional |
| 5 | Unified schema across all scripts | 🔲 Future |

---

## Viewer Adapters

The viewer handles format variations:

```python
def get_neighbors(patch):
    for key in ['neighbors', 'nearest_neighbors', 'top_neighbors', 
                'top_predictions', 'nearest_contextual_neighbors']:
        if key in patch:
            return patch[key]
    return []
```

---

## Notes

- **VG only**: Conceptual Captions (CC) corpus deprecated
- **Viewer = adapter**: Translates between JSON formats and HTML template
- **Don't break things**: Test each phase before moving to next

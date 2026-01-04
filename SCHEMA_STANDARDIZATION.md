# V-Lens Data Schema Reference

Documentation for JSON data formats across all interpretability analysis scripts.

---

## Display Names
| Internal Code | Display Name (Paper/UI) |
|---------------|-------------------------|
| `nn` | **Embedding Matrix** |
| `logitlens` | **LogitLens** |
| `contextual` | **LN-Lens** |

---

## Primary Scripts

| Purpose | Script |
|---------|--------|
| Embedding Matrix (Molmo) | `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` |
| Embedding Matrix (Qwen2-VL) | `qwen2_vl/nearest_neighbors.py` |
| LogitLens | `logitlens.py`, `qwen2_vl/logitlens.py` |
| LN-Lens | `contextual_nearest_neighbors_allLayers_singleGPU.py` |
| Viewer | `create_unified_viewer.py`, `generate_ablation_viewers.py` |

---

## Folder Structure

```
analysis_results/
├── nearest_neighbors/           # Embedding Matrix results
├── logit_lens/                  # LogitLens results
├── contextual_nearest_neighbors/  # LN-Lens results (VG corpus)
└── unified_viewer_lite_final/   # HTML viewer output
```

---

## JSON Formats

### Embedding Matrix (Molmo)
```json
{
  "splits": {"validation": {"images": [
    {"image_idx": 0, "chunks": [{"patches": [
      {"patch_idx": 0, "nearest_neighbors": [{"token": "sky", "similarity": 0.85}]}
    ]}]}
  ]}}
}
```

### Embedding Matrix (Qwen2-VL)
```json
{
  "results": [
    {"image_idx": 0, "patches": [
      {"patch_idx": 0, "nearest_neighbors": [{"token": "sky", "similarity": 0.85}]}
    ]}
  ]
}
```

### LogitLens
```json
{
  "results": [
    {"image_idx": 0, "chunks": [{"patches": [
      {"patch_idx": 0, "top_predictions": [{"token": "sky", "logit": 5.2}]}
    ]}]}
  ]
}
```

### LN-Lens (allLayers format)
```json
{
  "visual_layer": 0,
  "contextual_layers_used": [1, 2, 4, 8, 16, 24, 30, 31],
  "results": [
    {"image_idx": 0, "chunks": [{"patches": [
      {"patch_idx": 0, "nearest_contextual_neighbors": [
        {"token_str": "sky", "similarity": 0.85, "caption": "...", "contextual_layer": 8}
      ]}
    ]}]}
  ]
}
```

---

## Key Names Summary

| Analysis Type | Patch Key | Neighbor Fields |
|--------------|-----------|-----------------|
| Embedding Matrix | `nearest_neighbors` | `token`, `similarity` |
| LogitLens | `top_predictions` | `token`, `logit`, `token_id` |
| LN-Lens | `nearest_contextual_neighbors` | `token_str`, `similarity`, `caption`, `contextual_layer` |

---

## Viewer Architecture: Visual vs Contextual Layers

**CRITICAL: LN-Lens has TWO separate layer concepts - DO NOT CONFUSE THEM!**

### Visual Layer (UI dropdown)
- **What it is**: Selects which visual representation to analyze (layer 0, 1, 2, 4, 8, 16, 24, 30, 31)
- **Data source**: Corresponds to `visual_layer_N.json` files
- **What it means**: This is the layer of the vision encoder output being analyzed

### Contextual Layer (UI badges: L1, L2, L8, etc.)
- **What it is**: Shows which LLM layer the contextual neighbor embedding came from
- **Data source**: Stored in each neighbor's `contextual_layer` field
- **Critical**: ALL top-5 neighbors are shown together with their badges - DO NOT filter by contextual_layer!

### Example User Flow
1. User selects **"Layer 0"** from dropdown → analyzing visual layer 0 (input to LLM)
2. User clicks on **patch 5** in the image grid
3. Viewer shows **ALL top-5 contextual neighbors** with their badges:
   ```
   Rank 1: "sky"      (L8,  similarity=0.85)
   Rank 2: "blue"     (L2,  similarity=0.83)
   Rank 3: "cloud"    (L8,  similarity=0.81)
   Rank 4: "weather"  (L16, similarity=0.79)
   Rank 5: "ceiling"  (L24, similarity=0.77)
   ```
4. The badges **(L8, L2, L8, L16, L24)** show which contextual layer each neighbor came from

### Common Mistake to Avoid
**WRONG**: Filtering contextual neighbors by layer
```python
# ❌ WRONG - destroys badge functionality!
nearest_contextual = [n for n in all_neighbors if n.get("contextual_layer") == layer]
```

**RIGHT**: Show all top-5 neighbors with their contextual_layer preserved
```python
# ✅ CORRECT - preserves badges
all_neighbors = patch.get("nearest_contextual_neighbors", [])
for neighbor in all_neighbors[:5]:  # Show all top-5
    contextual_layer = neighbor.get("contextual_layer", -1)  # Use for badge display
```

---

## Consumer Scripts

Scripts that read these JSON files:
- `paper_plots/*.py` - Paper figures
- `llm_judge/*.py` - GPT-4o evaluations
- `create_unified_viewer.py` - HTML viewer
- `generate_ablation_viewers.py` - Ablation viewer

**Important:** Before changing any JSON key names, audit ALL consumer scripts.

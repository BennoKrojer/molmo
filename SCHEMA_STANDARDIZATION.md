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

## Consumer Scripts

Scripts that read these JSON files:
- `paper_plots/*.py` - Paper figures
- `llm_judge/*.py` - GPT-4o evaluations
- `create_unified_viewer.py` - HTML viewer
- `generate_ablation_viewers.py` - Ablation viewer

**Important:** Before changing any JSON key names, audit ALL consumer scripts.

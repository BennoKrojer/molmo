# V-Lens Naming Standardization Plan

## Current State (Inconsistent)

### Method Names
| Location | NN Method | LogitLens | Contextual Method |
|----------|-----------|-----------|-------------------|
| **Paper/Demo (TARGET)** | Embedding Matrix | LogitLens | LN-Lens |
| Folder names | `nearest_neighbors/` | `logit_lens/` | `contextual_nearest_neighbors_vg/` |
| JSON output keys | `nearest_neighbors` | `top_predictions` | `nearest_contextual_neighbors` |
| Viewer data keys | `neighbors` | `predictions` | `contextual_neighbors` |
| Wrapper keys | `nn` | `logitlens` | `contextual_vg` / `contextual_cc` |

### JSON Key Inconsistencies
| Script | NN Key | LogitLens Key | Contextual Key |
|--------|--------|---------------|----------------|
| `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` | `nearest_neighbors` | - | - |
| `qwen2_vl/nearest_neighbors.py` | `top_neighbors` | - | - |
| `logitlens.py` | - | `top_predictions` | - |
| `qwen2_vl/logitlens.py` | - | `top_predictions` | - |
| `contextual_nearest_neighbors_allLayers_singleGPU.py` | - | - | `nearest_contextual_neighbors` |
| **Viewer expects** | `neighbors` | `predictions` | `contextual_neighbors` |

---

## Proposed Standardization

### 1. Display Names (Paper/Demo UI)
| Internal Code | Display Name |
|---------------|--------------|
| `nn` | **Embedding Matrix** |
| `logitlens` | **LogitLens** |
| `contextual` | **LN-Lens** |

### 2. Folder Names (Keep as-is for now)
- `nearest_neighbors/` → keep (renaming would break paths)
- `logit_lens/` → keep
- `contextual_nearest_neighbors_vg/` → rename to `contextual_nearest_neighbors/` (drop `_vg`)

### 3. JSON Output Keys (Standardize)
| What | Current (varies) | Standard |
|------|------------------|----------|
| NN results | `nearest_neighbors`, `top_neighbors` | `nearest_neighbors` |
| LogitLens results | `top_predictions`, `top_tokens` | `top_predictions` |
| Contextual results | `nearest_contextual_neighbors` | `nearest_contextual_neighbors` |

### 4. Viewer Internal Keys (Keep as-is)
The viewer converts to these internally - no change needed:
- `neighbors`, `predictions`, `contextual_neighbors`

### 5. Wrapper Keys (Simplify)
| Current | New |
|---------|-----|
| `contextual_vg` | `contextual` |
| `contextual_cc` | (remove - not used) |

---

## Files to Update

### Priority 1: Drop `_vg` suffix (VG is now the only corpus)

**Scripts:**
- [ ] `scripts/analysis/create_unified_viewer.py`
  - Change `contextual_vg` → `contextual`
  - Remove `contextual_cc` handling (or keep as fallback)
- [ ] `scripts/analysis/generate_ablation_viewers.py`
  - Change `contextual_vg` → `contextual`

**Folders:**
- [ ] `analysis_results/contextual_nearest_neighbors_vg/` → `contextual_nearest_neighbors/`
- [ ] `molmo_data/contextual_llm_embeddings_vg/` → `contextual_llm_embeddings/`

### Priority 2: Display Names in Viewer

**Update column headers:**
- [ ] `scripts/analysis/create_unified_viewer.py` - HTML template section
  - "Static NN" or "NN" → "Embedding Matrix"
  - Keep "LogitLens"
  - "Contextual NN" → "LN-Lens"

### Priority 3: Standardize JSON Output Keys

**NN scripts:**
- [ ] `scripts/analysis/qwen2_vl/nearest_neighbors.py`
  - Change `top_neighbors` → `nearest_neighbors`

**LogitLens scripts:** (already consistent - `top_predictions`)
- No change needed

**Contextual scripts:** (already consistent - `nearest_contextual_neighbors`)
- No change needed

### Priority 4: Legacy Script Cleanup (Later)

Mark as deprecated or remove:
- `contextual_nearest_neighbors.py` (use `_allLayers_singleGPU` instead)
- `contextual_nearest_neighbors_allLayers_slower.py`
- `contextual_nearest_neighbors_allLayers.py` (multi-GPU version, complex)
- `general_and_nearest_neighbors.py`
- `general_and_nearest_neighbors_pixmo_cap.py` (single-GPU)
- `input_embedding_nearest_neighbors_fast.py`

---

## Primary Scripts (Source of Truth)

| Purpose | Primary Script |
|---------|----------------|
| Embedding Matrix (NN) | `general_and_nearest_neighbors_pixmo_cap_multi-gpu.py` |
| LogitLens | `logitlens.py` |
| LN-Lens (Contextual) | `contextual_nearest_neighbors_allLayers_singleGPU.py` |
| Qwen2-VL Embedding Matrix | `qwen2_vl/nearest_neighbors.py` |
| Qwen2-VL LogitLens | `qwen2_vl/logitlens.py` |
| Qwen2-VL LN-Lens | `qwen2_vl/contextual_nearest_neighbors_allLayers_singleGPU.py` |
| Viewer Generation | `create_unified_viewer.py` |
| Ablation Viewer | `generate_ablation_viewers.py` |

---

## Implementation Order

1. **Phase 1: Viewer display names** (safe, UI only)
   - Update column headers in HTML template
   
2. **Phase 2: Drop `_vg` suffix** (requires folder rename + script updates)
   - Rename folders
   - Update all scripts that reference `_vg`
   - Update `run_all_missing.sh`
   
3. **Phase 3: Standardize Qwen2-VL key** (requires data regeneration)
   - Change `top_neighbors` → `nearest_neighbors` in `qwen2_vl/nearest_neighbors.py`
   - Regenerate Qwen2-VL NN data

4. **Phase 4: Legacy cleanup** (optional, for cleanliness)
   - Add deprecation notices to legacy scripts
   - Update README

---

## Notes

- **Don't break things**: Each phase should be tested before moving to the next
- **VG only**: Conceptual Captions (CC) corpus is no longer used
- **Viewer adapter**: The viewer scripts (`create_unified_viewer.py`, `generate_ablation_viewers.py`) act as adapters between JSON output and HTML template - they handle key translation


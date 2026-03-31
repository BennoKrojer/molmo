# Dynamic LatentLens Study

Small-scale study validating dynamic phrase generation for LatentLens.

## Concept

Instead of searching a fixed corpus (Visual Genome), we:
1. Start with current LatentLens top phrases
2. Use GPT-4o to generate variations/elaborations
3. Compute contextual embeddings of generated phrases using OLMo
4. Score by cosine similarity to visual token
5. Keep top-3, iterate (evolutionary search)

## Usage

```bash
# Activate environment
source ../../env/bin/activate

# Run with defaults (5 examples, layer 16, 3 rounds)
python run_dynamic_study.py

# Custom settings
python run_dynamic_study.py --num-examples 5 --visual-layer 16 --num-rounds 3

# Output to custom directory
python run_dynamic_study.py --output-dir analysis_results/my_dynamic_study
```

## Requirements

- OpenAI API key set as `OPENAI_API_KEY` environment variable
- GPU with ~24GB VRAM (loads both VLM and LLM)
- Existing LatentLens results for OLMo + CLIP model

## Output

```
analysis_results/dynamic_latentlens_study/
├── results.json                    # All results and metrics
├── example0_visual_embedding.npy   # Visual token embeddings
├── example0_img0_r13_c9.png        # Visualization for manual eval
├── example1_visual_embedding.npy
├── example1_img1_r3_c16.png
└── ...
```

## Visualization Format

Each PNG shows:
- Original image with red bounding box on patch
- ORIGINAL: Top-5 phrases from VG corpus with similarities
- EVOLVED: Top-5 phrases from evolutionary search with similarities
- Improvement score (evolved best - original best)

## Manual Evaluation

Review each PNG and assess:
1. Are evolved phrases more descriptive/accurate?
2. Do they capture nuances the original missed?
3. Are they appropriate for the visual content?

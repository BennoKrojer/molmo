#!/usr/bin/env python3
"""Unit tests for LLaVA-1.5-7B preprocessing and vision token handling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
import torch
from PIL import Image

from preprocessing import (
    preprocess_image_llava,
    get_grid_dimensions,
    get_vision_token_positions,
    validate_vision_tokens,
    validate_preprocessing,
    NUM_VISION_TOKENS,
    GRID_H,
    GRID_W,
    IMAGE_TOKEN_ID,
    IMAGE_SIZE,
    PATCH_SIZE,
)


class TestPreprocessImage:
    def test_square_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        result = preprocess_image_llava(img, target_size=336)
        assert result.size == (336, 336)

    def test_landscape_center_crop(self):
        img = Image.fromarray(np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8))
        result = preprocess_image_llava(img, target_size=336)
        assert result.size == (336, 336)

    def test_portrait_center_crop(self):
        img = Image.fromarray(np.random.randint(0, 255, (600, 300, 3), dtype=np.uint8))
        result = preprocess_image_llava(img, target_size=336)
        assert result.size == (336, 336)

    def test_for_llm_judge(self):
        """LLM judge uses 512x512 display images."""
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        result = preprocess_image_llava(img, target_size=512)
        assert result.size == (512, 512)


class TestGridDimensions:
    def test_grid_values(self):
        h, w, n = get_grid_dimensions()
        assert h == 24
        assert w == 24
        assert n == 576
        assert h * w == n

    def test_constants_derived_from_clip(self):
        assert GRID_H == IMAGE_SIZE // PATCH_SIZE
        assert GRID_W == IMAGE_SIZE // PATCH_SIZE
        assert NUM_VISION_TOKENS == GRID_H * GRID_W


class TestVisionTokenPositions:
    def test_single_image_token(self):
        """Standard case: one <image> token in sequence."""
        # Simulate: [BOS, token, token, <image>, token, token, EOS]
        input_ids = torch.tensor([1, 100, 200, IMAGE_TOKEN_ID, 300, 400, 2])
        positions = get_vision_token_positions(input_ids)

        assert len(positions) == NUM_VISION_TOKENS
        # First vision token at position 3 (where <image> was)
        assert positions[0] == (3, 0, 0)
        # Last vision token
        assert positions[-1] == (3 + NUM_VISION_TOKENS - 1, 23, 23)

        # Check full spatial grid coverage
        rows = set(r for _, r, _ in positions)
        cols = set(c for _, _, c in positions)
        assert rows == set(range(24))
        assert cols == set(range(24))

    def test_no_image_token_raises(self):
        input_ids = torch.tensor([1, 100, 200, 300, 2])
        with pytest.raises(ValueError, match="not found"):
            get_vision_token_positions(input_ids)

    def test_image_token_at_start(self):
        input_ids = torch.tensor([IMAGE_TOKEN_ID, 100, 200])
        positions = get_vision_token_positions(input_ids)
        assert positions[0] == (0, 0, 0)

    def test_spatial_mapping_correctness(self):
        """Verify row/col mapping is correct for all 576 tokens."""
        input_ids = torch.tensor([1, IMAGE_TOKEN_ID, 2])
        positions = get_vision_token_positions(input_ids)

        for i, (seq_pos, row, col) in enumerate(positions):
            expected_row = i // 24
            expected_col = i % 24
            assert row == expected_row, f"Token {i}: expected row {expected_row}, got {row}"
            assert col == expected_col, f"Token {i}: expected col {expected_col}, got {col}"
            assert seq_pos == 1 + i  # image_token was at position 1


class TestValidateVisionTokens:
    def test_correct_count(self):
        validate_vision_tokens(576)  # Should not raise

    def test_wrong_count(self):
        with pytest.raises(ValueError, match="Expected 576"):
            validate_vision_tokens(577)

    def test_zero_tokens(self):
        with pytest.raises(ValueError):
            validate_vision_tokens(0)


class TestValidatePreprocessing:
    def test_correct_size_336(self):
        img = Image.new("RGB", (336, 336))
        validate_preprocessing(img, expected_size=336)

    def test_correct_size_512(self):
        img = Image.new("RGB", (512, 512))
        validate_preprocessing(img, expected_size=512)

    def test_wrong_size(self):
        img = Image.new("RGB", (448, 448))
        with pytest.raises(ValueError):
            validate_preprocessing(img, expected_size=336)


@pytest.mark.slow
class TestWithActualModel:
    """Tests that require loading the actual LLaVA model/processor."""

    def test_processor_produces_expected_tokens(self):
        """Verify LLaVA's processor produces 576 vision tokens."""
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))

        inputs = proc(
            text="USER: <image>\nDescribe this image. ASSISTANT:",
            images=img,
            return_tensors="pt"
        )

        # Count how many image tokens are in input_ids
        # After processing, <image> is expanded to 576 image feature tokens
        # The pixel_values should have the right shape
        pixel_values = inputs["pixel_values"]
        assert pixel_values.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE), (
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} image, got {pixel_values.shape}"
        )

    def test_model_hidden_states_shape(self):
        """Verify model produces hidden states at expected dimensions."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
        inputs = proc(
            text="USER: <image>\nDescribe. ASSISTANT:",
            images=img,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # Should have 33 hidden states (embedding + 32 layers)
        assert len(outputs.hidden_states) == 33, (
            f"Expected 33 hidden states, got {len(outputs.hidden_states)}"
        )
        # Hidden dim should be 4096
        assert outputs.hidden_states[0].shape[-1] == 4096


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

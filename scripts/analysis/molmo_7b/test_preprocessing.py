#!/usr/bin/env python3
"""Unit tests for Molmo-7B-D preprocessing and vision token handling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from PIL import Image

from preprocessing import (
    preprocess_image_molmo,
    get_base_crop_grid,
    get_base_crop_token_positions,
    validate_base_crop,
    validate_preprocessing,
    TOKENS_PER_CROP,
    TOKENS_PER_CROP_H,
    TOKENS_PER_CROP_W,
    BASE_CROP_INDEX,
)


class TestPreprocessImage:
    def test_square_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        result = preprocess_image_molmo(img, target_size=512)
        assert result.size == (512, 512)

    def test_landscape_image_center_crop(self):
        img = Image.fromarray(np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8))
        result = preprocess_image_molmo(img, target_size=512)
        assert result.size == (512, 512)

    def test_portrait_image_center_crop(self):
        img = Image.fromarray(np.random.randint(0, 255, (600, 300, 3), dtype=np.uint8))
        result = preprocess_image_molmo(img, target_size=512)
        assert result.size == (512, 512)

    def test_no_force_square(self):
        img = Image.fromarray(np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8))
        result = preprocess_image_molmo(img, target_size=512, force_square=False)
        assert result.size == (512, 512)  # Still resizes, just doesn't crop first

    def test_already_correct_size(self):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        result = preprocess_image_molmo(img, target_size=512)
        assert result.size == (512, 512)


class TestGridDimensions:
    def test_base_crop_grid(self):
        h, w, n = get_base_crop_grid()
        assert h == 12
        assert w == 12
        assert n == 144
        assert h * w == n

    def test_constants_consistent(self):
        assert TOKENS_PER_CROP == TOKENS_PER_CROP_H * TOKENS_PER_CROP_W
        assert TOKENS_PER_CROP == 144


class TestBaseCropTokenPositions:
    def test_all_valid_positions(self):
        """Base crop with all tokens valid (typical case)."""
        import torch
        # Simulate: 2 crops, 144 tokens each. Base crop has seq positions 2-145
        image_input_idx = torch.zeros(2, TOKENS_PER_CROP, dtype=torch.int32)
        image_input_idx[0] = torch.arange(2, 2 + TOKENS_PER_CROP)  # Base crop
        image_input_idx[1] = torch.arange(200, 200 + TOKENS_PER_CROP)  # High-res crop

        positions = get_base_crop_token_positions(image_input_idx)
        assert len(positions) == TOKENS_PER_CROP

        # Check spatial mapping
        for seq_pos, row, col in positions:
            assert 0 <= row < TOKENS_PER_CROP_H
            assert 0 <= col < TOKENS_PER_CROP_W
            assert seq_pos >= 2

        # First token should be (2, 0, 0)
        assert positions[0] == (2, 0, 0)
        # Last token should be (145, 11, 11)
        assert positions[-1] == (2 + TOKENS_PER_CROP - 1, 11, 11)

    def test_with_padding(self):
        """Base crop with some padding (-100)."""
        import torch
        image_input_idx = torch.full((1, TOKENS_PER_CROP), -100, dtype=torch.int32)
        # Only first 100 are valid
        image_input_idx[0, :100] = torch.arange(5, 105)

        positions = get_base_crop_token_positions(image_input_idx)
        assert len(positions) == 100


class TestValidateBaseCrop:
    def test_valid_base_crop(self):
        import torch
        idx = torch.arange(2, 2 + TOKENS_PER_CROP).unsqueeze(0)  # (1, 144)
        num_valid = validate_base_crop(idx)
        assert num_valid == TOKENS_PER_CROP

    def test_no_crops_raises(self):
        import torch
        idx = torch.zeros(0, TOKENS_PER_CROP, dtype=torch.int32)
        with pytest.raises(ValueError, match="No crops found"):
            validate_base_crop(idx)

    def test_incomplete_base_crop_raises(self):
        import torch
        idx = torch.full((1, TOKENS_PER_CROP), -100, dtype=torch.int32)
        idx[0, :100] = torch.arange(100)
        with pytest.raises(ValueError, match="100 valid tokens"):
            validate_base_crop(idx)


class TestValidatePreprocessing:
    def test_correct_size(self):
        img = Image.new("RGB", (512, 512))
        validate_preprocessing(img, expected_size=512)  # Should not raise

    def test_wrong_size(self):
        img = Image.new("RGB", (336, 336))
        with pytest.raises(ValueError):
            validate_preprocessing(img, expected_size=512)


@pytest.mark.slow
class TestWithActualModel:
    """Tests that require loading the actual Molmo model/processor."""

    def test_processor_produces_expected_base_crop(self):
        """Verify Molmo's processor produces a base crop with 144 tokens."""
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924", trust_remote_code=True
        )
        img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
        inputs = proc.process(images=[img], text="Describe this image.")

        image_input_idx = inputs["image_input_idx"]
        assert image_input_idx.shape[1] == TOKENS_PER_CROP, (
            f"Expected {TOKENS_PER_CROP} tokens per crop, got {image_input_idx.shape[1]}"
        )

        # Base crop should have all 144 tokens valid
        base_valid = (image_input_idx[BASE_CROP_INDEX] >= 0).sum().item()
        assert base_valid == TOKENS_PER_CROP, (
            f"Base crop has {base_valid} valid tokens, expected {TOKENS_PER_CROP}"
        )

        # Verify spatial mapping
        positions = get_base_crop_token_positions(image_input_idx)
        assert len(positions) == TOKENS_PER_CROP
        rows = set(r for _, r, _ in positions)
        cols = set(c for _, _, c in positions)
        assert rows == set(range(12)), f"Expected rows 0-11, got {rows}"
        assert cols == set(range(12)), f"Expected cols 0-11, got {cols}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

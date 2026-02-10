#!/usr/bin/env python3
"""
Comprehensive unit tests for the LLM judge preprocessing and crop pipeline.

Tests verify correctness of:
- process_image_with_mask() for all encoder types
- crop_image_region() from both run_single_model_with_viz.py and run_single_model_with_viz_logitlens.py
- SigLIP 27x27 grid edge cases (the exact bug we hit)
- Full pipeline smoke test
- No hardcoded grid_size=24 in crop functions

Run with: pytest test_preprocessing_pipeline.py -v
"""

import sys
import os
import io
import re
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

import pytest

# Ensure llm_judge directory is importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    process_image_with_mask,
    resize_and_pad,
    load_image,
    calculate_square_bbox_from_patch,
    draw_bbox_on_image,
    sample_valid_patch_positions,
    clip_bbox_to_image,
)

# Import crop_image_region from BOTH scripts
from run_single_model_with_viz import crop_image_region as crop_image_region_viz
from run_single_model_with_viz_logitlens import crop_image_region as crop_image_region_logitlens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_image_640x480(tmp_path):
    """Create a 640x480 RGB test image with a gradient pattern."""
    img = Image.new("RGB", (640, 480))
    pixels = img.load()
    for x in range(640):
        for y in range(480):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    path = tmp_path / "test_640x480.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def test_image_512x512():
    """Create a 512x512 solid-color PIL image."""
    return Image.new("RGB", (512, 512), color=(128, 64, 200))


@pytest.fixture
def test_image_1024x768(tmp_path):
    """Create a 1024x768 RGB test image (wide landscape)."""
    img = Image.new("RGB", (1024, 768), color=(50, 100, 150))
    path = tmp_path / "test_1024x768.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def test_image_480x640(tmp_path):
    """Create a 480x640 RGB test image (tall portrait)."""
    img = Image.new("RGB", (480, 640), color=(200, 100, 50))
    path = tmp_path / "test_480x640.png"
    img.save(str(path))
    return str(path)


# ===========================================================================
# Test 1: process_image_with_mask returns correct format for all encoder types
# ===========================================================================

class TestProcessImageWithMask:
    """Test process_image_with_mask() for all encoder types."""

    MODEL_NAMES = [
        "olmo-7b_vit-l-14-336",
        "olmo-7b_siglip",
        "olmo-7b_dinov2-large-336",
        # Skipping qwen2vl because it requires a special preprocessing module
        # that depends on qwen2-vl-specific code. Tested separately if available.
    ]

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_output_format(self, model_name, test_image_640x480):
        """Returned image is PIL 512x512; mask is bool numpy (512,512)."""
        processed_image, image_mask = process_image_with_mask(
            test_image_640x480, model_name=model_name
        )

        # Image checks
        assert isinstance(processed_image, Image.Image), (
            f"Expected PIL Image, got {type(processed_image)}"
        )
        assert processed_image.size == (512, 512), (
            f"Expected (512,512), got {processed_image.size}"
        )

        # Mask checks
        assert isinstance(image_mask, np.ndarray), (
            f"Expected numpy array, got {type(image_mask)}"
        )
        assert image_mask.shape == (512, 512), (
            f"Expected shape (512,512), got {image_mask.shape}"
        )
        assert image_mask.dtype == bool, (
            f"Expected dtype bool, got {image_mask.dtype}"
        )

    @pytest.mark.parametrize("model_name", ["olmo-7b_siglip", "llama3-8b_siglip"])
    def test_siglip_mask_all_true(self, model_name, test_image_640x480):
        """SigLIP squash-resizes, so mask must be ALL True (no padding)."""
        _, mask = process_image_with_mask(test_image_640x480, model_name=model_name)
        assert mask.all(), (
            f"SigLIP mask should be all True, but has {(~mask).sum()} False pixels"
        )

    @pytest.mark.parametrize("model_name", ["olmo-7b_dinov2-large-336", "llama3-8b_dinov2-large-336"])
    def test_dinov2_mask_all_true(self, model_name, test_image_640x480):
        """DINOv2 squash-resizes, so mask must be ALL True (no padding)."""
        _, mask = process_image_with_mask(test_image_640x480, model_name=model_name)
        assert mask.all(), (
            f"DINOv2 mask should be all True, but has {(~mask).sum()} False pixels"
        )

    def test_clip_mask_has_padding_for_nonsquare(self, test_image_640x480):
        """CLIP (vit-l-14) uses resize+pad, so non-square images should have padding."""
        _, mask = process_image_with_mask(
            test_image_640x480, model_name="olmo-7b_vit-l-14-336"
        )
        # 640x480 is wider than tall, so top/bottom will be padded
        # Mask should have some False values (padded pixels)
        assert not mask.all(), (
            "CLIP mask for a non-square image should have some False (padded) pixels"
        )
        # But it should have plenty of True pixels
        true_fraction = mask.sum() / mask.size
        assert true_fraction > 0.5, (
            f"CLIP mask True fraction is only {true_fraction:.3f}, expected > 0.5"
        )

    def test_clip_mask_all_true_for_square(self, tmp_path):
        """CLIP with a square image should have no padding."""
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        path = tmp_path / "square.png"
        img.save(str(path))
        _, mask = process_image_with_mask(str(path), model_name="olmo-7b_vit-l-14-336")
        assert mask.all(), "CLIP mask for a square image should be all True"

    def test_portrait_image(self, test_image_480x640):
        """CLIP with a portrait (tall) image: left/right should be padded."""
        _, mask = process_image_with_mask(
            test_image_480x640, model_name="olmo-7b_vit-l-14-336"
        )
        # Portrait: image is narrower, so left/right sides get padding
        assert not mask.all(), "CLIP mask for portrait image should have padding"

    def test_landscape_image(self, test_image_1024x768):
        """CLIP with a wide landscape image: top/bottom should be padded."""
        _, mask = process_image_with_mask(
            test_image_1024x768, model_name="olmo-7b_vit-l-14-336"
        )
        assert not mask.all(), "CLIP mask for wide landscape image should have padding"

    def test_default_model_name_none(self, test_image_640x480):
        """model_name=None should fall through to default CLIP path."""
        processed_image, image_mask = process_image_with_mask(
            test_image_640x480, model_name=None
        )
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (512, 512)
        assert isinstance(image_mask, np.ndarray)
        assert image_mask.shape == (512, 512)


# ===========================================================================
# Test 2: crop_image_region works with all grid sizes
# ===========================================================================

class TestCropImageRegion:
    """Test crop_image_region from both scripts across grid sizes."""

    CROP_FUNCTIONS = [
        ("viz", crop_image_region_viz),
        ("logitlens", crop_image_region_logitlens),
    ]

    GRID_SIZES = [24, 27, 16]

    @pytest.mark.parametrize("fname, crop_fn", CROP_FUNCTIONS, ids=lambda x: x if isinstance(x, str) else "")
    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_top_left_corner(self, fname, crop_fn, grid_size, test_image_512x512):
        """Crop at (0,0) with bbox_size=3 -- top-left corner."""
        cropped = crop_fn(test_image_512x512, 0, 0, 3, grid_size=grid_size)
        assert isinstance(cropped, Image.Image)
        assert cropped.width > 0 and cropped.height > 0
        # Must be saveable as PNG
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        assert len(buf.getvalue()) > 0

    @pytest.mark.parametrize("fname, crop_fn", CROP_FUNCTIONS, ids=lambda x: x if isinstance(x, str) else "")
    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_bottom_right_corner(self, fname, crop_fn, grid_size, test_image_512x512):
        """CRITICAL: Crop at (grid_size-3, grid_size-3) -- bottom-right corner.
        This is exactly where the old bug crashed."""
        row = grid_size - 3
        col = grid_size - 3
        cropped = crop_fn(test_image_512x512, row, col, 3, grid_size=grid_size)
        assert isinstance(cropped, Image.Image)
        assert cropped.width > 0, f"Cropped width is 0 for grid_size={grid_size} at ({row},{col})"
        assert cropped.height > 0, f"Cropped height is 0 for grid_size={grid_size} at ({row},{col})"
        # Must be saveable as PNG without error
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        assert len(buf.getvalue()) > 0

    @pytest.mark.parametrize("fname, crop_fn", CROP_FUNCTIONS, ids=lambda x: x if isinstance(x, str) else "")
    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_center(self, fname, crop_fn, grid_size, test_image_512x512):
        """Crop at center of grid."""
        row = grid_size // 2
        col = grid_size // 2
        cropped = crop_fn(test_image_512x512, row, col, 3, grid_size=grid_size)
        assert isinstance(cropped, Image.Image)
        assert cropped.width > 0 and cropped.height > 0
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        assert len(buf.getvalue()) > 0

    @pytest.mark.parametrize("fname, crop_fn", CROP_FUNCTIONS, ids=lambda x: x if isinstance(x, str) else "")
    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_crop_within_image_bounds(self, fname, crop_fn, grid_size, test_image_512x512):
        """Verify crop coordinates stay within [0, 512] for all corners."""
        positions = [
            (0, 0),
            (grid_size - 3, grid_size - 3),
            (grid_size // 2, grid_size // 2),
        ]
        for row, col in positions:
            cropped = crop_fn(test_image_512x512, row, col, 3, grid_size=grid_size)
            # The cropped image should come from within the 512x512 image
            # Verify non-zero dimensions
            assert cropped.width > 0 and cropped.height > 0, (
                f"Invalid crop at ({row},{col}) grid_size={grid_size}: "
                f"size=({cropped.width},{cropped.height})"
            )
            # The crop region should be at most 512x512
            assert cropped.width <= 512 and cropped.height <= 512

    @pytest.mark.parametrize("fname, crop_fn", CROP_FUNCTIONS, ids=lambda x: x if isinstance(x, str) else "")
    def test_crop_at_exact_boundary(self, fname, crop_fn, test_image_512x512):
        """For grid_size=24, patch at (21,21) with bbox_size=3: right/bottom = 24*512/24 = 512."""
        cropped = crop_fn(test_image_512x512, 21, 21, 3, grid_size=24)
        assert isinstance(cropped, Image.Image)
        assert cropped.width > 0 and cropped.height > 0
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        assert len(buf.getvalue()) > 0


# ===========================================================================
# Test 3: SigLIP 27x27 grid edge case (the exact bug we hit)
# ===========================================================================

class TestSigLIP27x27EdgeCase:
    """Reproduce and verify the SigLIP 27x27 grid edge case that caused crashes."""

    @pytest.mark.parametrize("crop_fn", [crop_image_region_viz, crop_image_region_logitlens],
                             ids=["viz", "logitlens"])
    def test_patch_24_24_bbox3_grid27(self, crop_fn, test_image_512x512):
        """
        grid_size=27, patch at (24,24) with bbox_size=3.
        patch_size = 512/27 ~ 18.963
        bottom = (24+3) * 18.963 = 512.0 -- right at the edge.
        This must NOT crash and must produce a valid PNG.
        """
        cropped = crop_fn(test_image_512x512, 24, 24, 3, grid_size=27)
        assert isinstance(cropped, Image.Image)
        assert cropped.width > 0, "Cropped image has zero width at SigLIP edge case"
        assert cropped.height > 0, "Cropped image has zero height at SigLIP edge case"
        # Must produce a valid PNG
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        png_data = buf.getvalue()
        assert len(png_data) > 0, "PNG output is empty"
        # Verify it can be read back
        buf.seek(0)
        reloaded = Image.open(buf)
        assert reloaded.size == cropped.size

    @pytest.mark.parametrize("crop_fn", [crop_image_region_viz, crop_image_region_logitlens],
                             ids=["viz", "logitlens"])
    def test_pixel_coordinates_at_edge(self, crop_fn, test_image_512x512):
        """Verify the pixel math: patch_size=512/27, so (24+3)*patch_size ~ 512.0."""
        grid_size = 27
        patch_size = 512.0 / grid_size  # ~18.963
        row, col = 24, 24
        bbox_size = 3

        expected_left = int(col * patch_size)
        expected_top = int(row * patch_size)
        expected_right = min(int((col + bbox_size) * patch_size), 512)
        expected_bottom = min(int((row + bbox_size) * patch_size), 512)

        # Verify our expectations
        assert expected_right <= 512, f"Expected right <= 512, got {expected_right}"
        assert expected_bottom <= 512, f"Expected bottom <= 512, got {expected_bottom}"
        assert expected_right > expected_left, "Right must be > left"
        assert expected_bottom > expected_top, "Bottom must be > top"

        # Verify crop succeeds
        cropped = crop_fn(test_image_512x512, row, col, bbox_size, grid_size=grid_size)
        assert cropped.width == expected_right - expected_left
        assert cropped.height == expected_bottom - expected_top

    @pytest.mark.parametrize("crop_fn", [crop_image_region_viz, crop_image_region_logitlens],
                             ids=["viz", "logitlens"])
    def test_all_valid_positions_27x27(self, crop_fn, test_image_512x512):
        """Every valid position on a 27x27 grid with bbox_size=3 should produce valid crops."""
        grid_size = 27
        bbox_size = 3
        for row in range(grid_size - bbox_size + 1):
            for col in range(grid_size - bbox_size + 1):
                cropped = crop_fn(test_image_512x512, row, col, bbox_size, grid_size=grid_size)
                assert cropped.width > 0 and cropped.height > 0, (
                    f"Invalid crop at ({row},{col}): size=({cropped.width},{cropped.height})"
                )


# ===========================================================================
# Test 4: Full pipeline smoke test
# ===========================================================================

class TestFullPipelineSmokeTest:
    """
    End-to-end pipeline test for SigLIP (grid=27):
    process_image -> sample patches -> crop -> draw bbox -> save PNG
    """

    def test_siglip_full_pipeline(self, test_image_640x480):
        """
        For model_name "olmo-7b_siglip" (SigLIP, grid=27):
        1. Process image with process_image_with_mask
        2. Sample patch positions with sample_valid_patch_positions(mask, 3, 5, 27)
        3. For each position, crop with crop_image_region
        4. Draw bbox with draw_bbox_on_image
        5. Save both to BytesIO as PNG
        All operations must succeed without error.
        """
        model_name = "olmo-7b_siglip"
        grid_size = 27
        bbox_size = 3

        # Step 1: Process
        processed_image, mask = process_image_with_mask(
            test_image_640x480, model_name=model_name
        )
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (512, 512)
        assert mask.all(), "SigLIP mask should be all True"

        # Step 2: Sample
        positions = sample_valid_patch_positions(
            mask, bbox_size=bbox_size, num_samples=5, grid_size=grid_size
        )
        assert len(positions) == 5, f"Expected 5 positions, got {len(positions)}"

        for row, col in positions:
            # Validate position ranges
            assert 0 <= row <= grid_size - bbox_size, f"row {row} out of range"
            assert 0 <= col <= grid_size - bbox_size, f"col {col} out of range"

            # Step 3: Crop
            cropped = crop_image_region_viz(
                processed_image, row, col, bbox_size, grid_size=grid_size
            )
            assert isinstance(cropped, Image.Image)
            assert cropped.width > 0 and cropped.height > 0

            # Step 4: Draw bbox
            actual_patch_size = 512.0 / grid_size
            bbox = calculate_square_bbox_from_patch(
                row, col, patch_size=actual_patch_size, size=bbox_size
            )
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            assert isinstance(image_with_bbox, Image.Image)
            assert image_with_bbox.size == (512, 512)

            # Step 5: Save both as PNG
            buf_crop = io.BytesIO()
            cropped.save(buf_crop, format="PNG")
            assert len(buf_crop.getvalue()) > 0, "Cropped PNG is empty"

            buf_bbox = io.BytesIO()
            image_with_bbox.save(buf_bbox, format="PNG")
            assert len(buf_bbox.getvalue()) > 0, "Bbox PNG is empty"

    def test_clip_full_pipeline(self, test_image_640x480):
        """Same pipeline test for CLIP (grid=24)."""
        model_name = "olmo-7b_vit-l-14-336"
        grid_size = 24
        bbox_size = 3

        processed_image, mask = process_image_with_mask(
            test_image_640x480, model_name=model_name
        )
        assert processed_image.size == (512, 512)

        positions = sample_valid_patch_positions(
            mask, bbox_size=bbox_size, num_samples=5, grid_size=grid_size
        )
        # May get fewer than 5 if non-square image has lots of padding
        assert len(positions) > 0, "Should get at least one valid position"

        for row, col in positions:
            cropped = crop_image_region_viz(
                processed_image, row, col, bbox_size, grid_size=grid_size
            )
            assert isinstance(cropped, Image.Image)
            assert cropped.width > 0 and cropped.height > 0

            actual_patch_size = 512.0 / grid_size
            bbox = calculate_square_bbox_from_patch(
                row, col, patch_size=actual_patch_size, size=bbox_size
            )
            image_with_bbox = draw_bbox_on_image(processed_image, bbox)
            assert image_with_bbox.size == (512, 512)

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            assert len(buf.getvalue()) > 0

    def test_dinov2_full_pipeline(self, test_image_640x480):
        """Same pipeline test for DINOv2 (grid=24)."""
        model_name = "olmo-7b_dinov2-large-336"
        grid_size = 24
        bbox_size = 3

        processed_image, mask = process_image_with_mask(
            test_image_640x480, model_name=model_name
        )
        assert processed_image.size == (512, 512)
        assert mask.all(), "DINOv2 mask should be all True"

        positions = sample_valid_patch_positions(
            mask, bbox_size=bbox_size, num_samples=5, grid_size=grid_size
        )
        assert len(positions) == 5

        for row, col in positions:
            cropped = crop_image_region_logitlens(
                processed_image, row, col, bbox_size, grid_size=grid_size
            )
            assert isinstance(cropped, Image.Image)
            assert cropped.width > 0 and cropped.height > 0

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            assert len(buf.getvalue()) > 0


# ===========================================================================
# Test 5: Verify grid_size is NOT hardcoded anywhere in crop functions
# ===========================================================================

class TestNoHardcodedGridSize:
    """Verify that crop_image_region does NOT hardcode grid_size=24."""

    def _read_crop_function_source(self, filepath):
        """Read the crop_image_region function body from a source file."""
        with open(filepath, "r") as f:
            source = f.read()

        # Extract the crop_image_region function body
        match = re.search(
            r"def crop_image_region\(.*?\):\s*\n(.*?)(?=\ndef |\nclass |\Z)",
            source,
            re.DOTALL,
        )
        assert match is not None, f"Could not find crop_image_region in {filepath}"
        return match.group(0)

    def test_no_hardcoded_24_in_viz_script(self):
        """run_single_model_with_viz.py crop_image_region must not use /24 or / 24."""
        filepath = Path(__file__).parent / "run_single_model_with_viz.py"
        func_source = self._read_crop_function_source(str(filepath))

        # Check for hardcoded division by 24 (the old bug pattern)
        # Pattern: img_width / 24 or similar
        bad_patterns = [
            r"(?<!\w)img_width\s*/\s*24(?!\d)",
            r"(?<!\w)img_height\s*/\s*24(?!\d)",
            r"(?<!\w)512\s*/\s*24(?!\d)",
            r"(?<!\w)/\s*24(?!\d)(?!\.)",  # bare / 24 (not in a comment or default param)
        ]
        for pattern in bad_patterns:
            # Only match outside of comments and the default parameter
            lines = func_source.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue
                # Skip the function signature default parameter grid_size=24
                if "def crop_image_region" in line:
                    continue
                if re.search(pattern, line):
                    pytest.fail(
                        f"Found hardcoded /24 pattern in run_single_model_with_viz.py: {line.strip()}"
                    )

    def test_no_hardcoded_24_in_logitlens_script(self):
        """run_single_model_with_viz_logitlens.py crop_image_region must not use /24 or / 24."""
        filepath = Path(__file__).parent / "run_single_model_with_viz_logitlens.py"
        func_source = self._read_crop_function_source(str(filepath))

        bad_patterns = [
            r"(?<!\w)img_width\s*/\s*24(?!\d)",
            r"(?<!\w)img_height\s*/\s*24(?!\d)",
            r"(?<!\w)512\s*/\s*24(?!\d)",
            r"(?<!\w)/\s*24(?!\d)(?!\.)",
        ]
        for pattern in bad_patterns:
            lines = func_source.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue
                if "def crop_image_region" in line:
                    continue
                if re.search(pattern, line):
                    pytest.fail(
                        f"Found hardcoded /24 pattern in run_single_model_with_viz_logitlens.py: {line.strip()}"
                    )

    def test_crop_uses_grid_size_parameter(self):
        """Verify that different grid_size values produce different crop sizes."""
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))

        # Same patch position, different grid sizes should yield different crop sizes
        crop_24 = crop_image_region_viz(img, 0, 0, 3, grid_size=24)
        crop_27 = crop_image_region_viz(img, 0, 0, 3, grid_size=27)
        crop_16 = crop_image_region_viz(img, 0, 0, 3, grid_size=16)

        # grid_size=24: 3 patches * (512/24) = 64px
        # grid_size=27: 3 patches * (512/27) ~ 56.9px = 56px
        # grid_size=16: 3 patches * (512/16) = 96px
        assert crop_24.width != crop_27.width, (
            f"grid_size=24 and 27 produced same width: {crop_24.width}"
        )
        assert crop_24.width != crop_16.width, (
            f"grid_size=24 and 16 produced same width: {crop_24.width}"
        )
        assert crop_27.width != crop_16.width, (
            f"grid_size=27 and 16 produced same width: {crop_27.width}"
        )


# ===========================================================================
# Additional helper function tests
# ===========================================================================

class TestHelperFunctions:
    """Tests for utility functions in utils.py."""

    def test_clip_bbox_to_image(self):
        """clip_bbox_to_image should clamp to [0, width] x [0, height]."""
        assert clip_bbox_to_image((-10, -10, 600, 600), 512, 512) == (0, 0, 512, 512)
        assert clip_bbox_to_image((10, 10, 100, 100), 512, 512) == (10, 10, 100, 100)
        assert clip_bbox_to_image((500, 500, 520, 520), 512, 512) == (500, 500, 512, 512)

    def test_calculate_square_bbox_from_patch(self):
        """Verify bbox calculation with integer and float patch sizes."""
        # Integer patch_size (CLIP grid=24 on 576px image)
        bbox = calculate_square_bbox_from_patch(0, 0, patch_size=24, size=3)
        assert bbox == (0, 0, 72, 72)

        # Float patch_size (SigLIP: 512/27 ~ 18.963)
        ps = 512.0 / 27
        bbox = calculate_square_bbox_from_patch(0, 0, patch_size=ps, size=3)
        assert bbox[0] == 0 and bbox[1] == 0
        # right = 3 * 18.963 ~ 56.89
        assert abs(bbox[2] - 3 * ps) < 1e-6
        assert abs(bbox[3] - 3 * ps) < 1e-6

    def test_draw_bbox_on_image_returns_rgb(self, test_image_512x512):
        """draw_bbox_on_image should return an RGB image even with overlay."""
        bbox = (10, 10, 100, 100)
        result = draw_bbox_on_image(test_image_512x512, bbox)
        assert result.mode == "RGB"
        assert result.size == (512, 512)

    def test_draw_bbox_on_image_edge_bbox(self, test_image_512x512):
        """draw_bbox_on_image at image edge should not crash."""
        bbox = (500, 500, 520, 520)  # Extends beyond image
        result = draw_bbox_on_image(test_image_512x512, bbox)
        assert isinstance(result, Image.Image)

    def test_draw_bbox_on_image_invalid_bbox(self, test_image_512x512):
        """draw_bbox_on_image with completely out-of-bounds bbox returns original."""
        bbox = (600, 600, 700, 700)  # Completely outside
        result = draw_bbox_on_image(test_image_512x512, bbox)
        # Should return original image (clipped bbox becomes invalid)
        assert isinstance(result, Image.Image)

    def test_sample_valid_patch_positions_all_true_mask(self):
        """All-true mask (SigLIP/DINOv2) should allow sampling anywhere."""
        mask = np.ones((512, 512), dtype=bool)
        positions = sample_valid_patch_positions(mask, bbox_size=3, num_samples=10, grid_size=24)
        assert len(positions) == 10
        for row, col in positions:
            assert 0 <= row <= 24 - 3
            assert 0 <= col <= 24 - 3

    def test_sample_valid_patch_positions_grid27(self):
        """Grid size 27 with all-true mask."""
        mask = np.ones((512, 512), dtype=bool)
        positions = sample_valid_patch_positions(mask, bbox_size=3, num_samples=10, grid_size=27)
        assert len(positions) == 10
        for row, col in positions:
            assert 0 <= row <= 27 - 3
            assert 0 <= col <= 27 - 3

    def test_sample_valid_patch_positions_with_padding(self):
        """Mask with padding should exclude padded areas."""
        mask = np.ones((512, 512), dtype=bool)
        # Simulate horizontal padding (portrait image): left 64px and right 64px are padded
        mask[:, :64] = False
        mask[:, 448:] = False

        positions = sample_valid_patch_positions(mask, bbox_size=3, num_samples=5, grid_size=24)
        patch_size = 512 // 24  # ~21

        for row, col in positions:
            # Each patch in the 3x3 block should be in the valid area
            for dr in range(3):
                for dc in range(3):
                    pr = row + dr
                    pc = col + dc
                    start_col_px = pc * patch_size
                    end_col_px = min((pc + 1) * patch_size, 512)
                    # All pixels in this patch should be True
                    patch_area = mask[:, start_col_px:end_col_px]
                    # At minimum, the patch row range should be valid
                    assert patch_area.any(), (
                        f"Position ({row},{col}) dr={dr} dc={dc} lands in padded area"
                    )

    def test_load_image(self, test_image_640x480):
        """load_image should return numpy uint8 RGB array."""
        arr = load_image(test_image_640x480)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (480, 640, 3)
        assert arr.dtype == np.uint8

    def test_resize_and_pad_output_shape(self, test_image_640x480):
        """resize_and_pad returns (512,512,3) float array and (512,512) bool mask."""
        arr = load_image(test_image_640x480)
        image, mask = resize_and_pad(arr, (512, 512), normalize=False)
        assert image.shape == (512, 512, 3)
        assert mask.shape == (512, 512)
        assert mask.dtype == bool


# ===========================================================================
# Test both crop functions produce identical results
# ===========================================================================

class TestCropFunctionConsistency:
    """Verify that crop_image_region from both scripts produces identical results."""

    @pytest.mark.parametrize("grid_size", [24, 27, 16])
    def test_both_scripts_produce_same_output(self, grid_size, test_image_512x512):
        """crop_image_region from viz and logitlens scripts should be identical."""
        positions = [
            (0, 0),
            (grid_size - 3, grid_size - 3),
            (grid_size // 2, grid_size // 2),
        ]
        for row, col in positions:
            crop_viz = crop_image_region_viz(
                test_image_512x512, row, col, 3, grid_size=grid_size
            )
            crop_ll = crop_image_region_logitlens(
                test_image_512x512, row, col, 3, grid_size=grid_size
            )
            assert crop_viz.size == crop_ll.size, (
                f"Size mismatch at ({row},{col}) grid={grid_size}: "
                f"viz={crop_viz.size} vs logitlens={crop_ll.size}"
            )
            # Pixel-level comparison
            arr_viz = np.array(crop_viz)
            arr_ll = np.array(crop_ll)
            assert np.array_equal(arr_viz, arr_ll), (
                f"Pixel mismatch at ({row},{col}) grid={grid_size}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

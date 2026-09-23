"""
File: test_dead_pixel_correction.py
Description: Unit tests for DPC border handling (CFA-aware padding).
             Run from the repo root:  python -m pytest tests/test_dead_pixel_correction.py
------------------------------------------------------------
"""
import sys

import numpy as np
import pytest

sys.path.append(".")
from modules.dead_pixel_correction import DeadPixelCorrection  # pylint: disable=C0413
from util.utils import pad_cfa  # pylint: disable=C0413


HEIGHT, WIDTH, BPP = 32, 32, 12
THRESHOLD = 80
BASE_LEVEL = 512

# Every row/col index a 5x5 window touches the border with: 0, 1, 2 and N-3, N-2, N-1
BORDER_ROWS = [0, 1, 2, HEIGHT - 3, HEIGHT - 2, HEIGHT - 1]
BORDER_COLS = [0, 1, 2, WIDTH - 3, WIDTH - 2, WIDTH - 1]
BORDER_POSITIONS = [(r, c) for r in BORDER_ROWS for c in BORDER_COLS]


def make_dpc(img):
    """Build a DPC object with minimal config dicts."""
    sensor_info = {"height": img.shape[0], "width": img.shape[1], "bit_depth": BPP}
    parm_dpc = {
        "is_enable": True,
        "is_save": False,
        "dp_threshold": THRESHOLD,
        "is_debug": False,
    }
    platform = {
        "in_file": "unit_test",
        "disable_progress_bar": True,
        "leave_pbar_string": False,
    }
    return DeadPixelCorrection(np.float32(img), platform, sensor_info, parm_dpc, None)


def make_clean_image(seed=0):
    """Flat image with mild noise, far below the DPC threshold."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 8, (HEIGHT, WIDTH))
    return (BASE_LEVEL + noise).astype(np.uint16)


@pytest.mark.parametrize("defect_value", [(2**BPP) - 1, 0], ids=["hot", "dead"])
@pytest.mark.parametrize("pos", BORDER_POSITIONS)
def test_isolated_border_defect_is_corrected(pos, defect_value):
    """A single defect at any border position must be detected and corrected."""
    img = make_clean_image()
    img[pos] = defect_value

    out = make_dpc(img).apply_fast_dead_pixel_correction()

    assert abs(int(out[pos]) - BASE_LEVEL) < THRESHOLD, (
        f"defect at {pos} not corrected: {out[pos]}"
    )


def test_fast_and_loop_versions_match():
    """Vectorised and loop implementations must be bit-exact."""
    img = make_clean_image()
    rng = np.random.default_rng(1)
    rows = rng.choice(np.arange(0, HEIGHT, 4), 6, replace=False)
    cols = rng.choice(np.arange(1, WIDTH, 4), 6, replace=False)
    img[rows, cols] = (2**BPP) - 1

    fast = make_dpc(img).apply_fast_dead_pixel_correction()
    loop = make_dpc(img).apply_dead_pixel_correction()

    np.testing.assert_array_equal(fast, loop)


def test_clean_image_is_untouched():
    """No defects -> output equals input, and shape/dtype are preserved."""
    img = make_clean_image()

    out = make_dpc(img).apply_fast_dead_pixel_correction()

    assert out.shape == img.shape
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, img)


def test_pad_cfa_never_mirrors_pixel_onto_itself():
    """Same-color neighbours of row/col 1 must not be the pixel itself."""
    img = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)

    padded = pad_cfa(img)

    assert padded.shape == (HEIGHT + 4, WIDTH + 4)
    np.testing.assert_array_equal(padded[2:-2, 2:-2], img)
    # original (1, 1) -> padded (3, 3); its same-color neighbours are 2 px away
    center = padded[3, 3]
    neighbours = padded[1:6:2, 1:6:2].ravel()
    neighbours = np.delete(neighbours, 4)  # drop the center itself
    assert np.all(neighbours != center)
    # every padded pixel keeps the Bayer phase of the original image
    for row in (0, 1):
        for col in (0, 1):
            np.testing.assert_array_equal(
                padded[row::2, col::2],
                np.pad(img[row::2, col::2], 1, mode="reflect"),
            )

"""
File: test_auto_exposure_stats.py
Description: Unit tests for the AE statistics block against a pixel-by-pixel model of
             the RTL (axis_ae_stat): every pixel is binned by comparing its (row, col)
             with the grid registers, green is decoded from the Bayer phase as the RTL
             does (sns_bayer XOR {row & 1, col & 1}: Gr / Gb), and the green sum is a
             per-pixel saturating add.
             Run from the repo root:  python -m pytest tests/test_auto_exposure_stats.py
------------------------------------------------------------
"""
import sys

import numpy as np
import pytest

sys.path.append(".")
import modules.auto_exposure_stats as aes_mod  # pylint: disable=C0413
from modules.auto_exposure_stats import AutoExposureStats  # pylint: disable=C0413

BPP = 10
FULL = (1 << BPP) - 1
SNS_BAYER = {"rggb": 0, "grbg": 1, "gbrg": 2, "bggr": 3}   # RTL sns_bayer encoding


def make_aes(img, bayer, cols, rows, shadow=50, highlight=950):
    """AES object with minimal config dicts."""
    sensor_info = {"bit_depth": BPP, "bayer_pattern": bayer,
                   "width": img.shape[1], "height": img.shape[0]}
    parm = {"is_enable": True, "is_save": False, "is_debug": False, "col_starts": cols,
            "row_starts": rows, "shadow_threshold": shadow, "highlight_threshold": highlight}
    return AutoExposureStats(img, {"in_file": "unit_test"}, sensor_info, parm, None)


def rtl_model(img, bayer, cols, rows, shadow, highlight, mean_bits=32):
    """Pixel-by-pixel model of the RTL accumulators."""
    out = {k: [0] * 9 for k in aes_mod.STAT_KEYS}
    mean_max = (1 << mean_bits) - 1
    for row in range(img.shape[0]):
        r_band = next((b for b in range(3) if rows[b] <= row < rows[b + 1]), None)
        for col in range(img.shape[1]):
            c_band = next((b for b in range(3) if cols[b] <= col < cols[b + 1]), None)
            if r_band is None or c_band is None:
                continue
            cell, val = r_band * 3 + c_band, int(img[row, col])
            fmt = SNS_BAYER[bayer] ^ (((row & 1) << 1) | (col & 1))
            if (fmt >> 1) ^ (fmt & 1):                       # Gr (01) / Gb (10)
                out["shadow_count"][cell] += val < shadow
                out["highlight_count"][cell] += val > highlight
                out["grid_mean_sum"][cell] = min(out["grid_mean_sum"][cell] + val, mean_max)
            out["black_clip_count"][cell] += val == 0
            out["white_clip_count"][cell] += val == FULL
    return out


def random_frame(height, width, seed):
    """Random raw with extra pixels at 0, full scale and on the thresholds."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, FULL + 1, size=(height, width)).astype(np.uint16)
    for value in (0, FULL, 50, 950):
        img[rng.random((height, width)) < 0.04] = value
    return img


@pytest.mark.parametrize("bayer", ["rggb", "grbg", "gbrg", "bggr"])
@pytest.mark.parametrize("height, width, cols, rows", [
    (24, 36, [0, 12, 24, 36], [0, 8, 16, 24]),          # grid = frame, even cells
    (23, 37, [0, 11, 24, 37], [0, 7, 15, 23]),          # odd frame, odd cells
    (30, 40, [3, 10, 21, 33], [5, 9, 20, 26]),          # grid inside the frame
])
def test_stats_match_rtl_model(bayer, height, width, cols, rows):
    """Every count of every cell equals the RTL model."""
    img = random_frame(height, width, seed=height * width)
    stats = make_aes(img, bayer, cols, rows).compute_grid_stats()
    assert stats == rtl_model(img, bayer, cols, rows, 50, 950)


def test_thresholds_are_strict():
    """A green pixel equal to a threshold is neither shadow nor highlight."""
    img = np.full((6, 6), 50, np.uint16)
    stats = make_aes(img, "rggb", [0, 2, 4, 6], [0, 2, 4, 6], 50, 50).compute_grid_stats()
    assert stats["shadow_count"] == [0] * 9 and stats["highlight_count"] == [0] * 9


def test_green_sum_saturates(monkeypatch):
    """The green sum saturates at the accumulator width instead of wrapping."""
    monkeypatch.setattr(aes_mod, "MEAN_BITS", 12)
    img = random_frame(24, 36, seed=7)
    cols, rows = [0, 12, 24, 36], [0, 8, 16, 24]
    stats = make_aes(img, "grbg", cols, rows).compute_grid_stats()
    ref = rtl_model(img, "grbg", cols, rows, 50, 950, mean_bits=12)
    assert stats["grid_mean_sum"] == ref["grid_mean_sum"]
    assert max(ref["grid_mean_sum"]) == (1 << 12) - 1        # the case really saturates


@pytest.mark.parametrize("cols, rows, shadow", [
    ([0, 12, 24, 37], [0, 8, 16, 24], 50),               # grid past the frame
    ([0, 24, 12, 36], [0, 8, 16, 24], 50),               # not ascending
    ([0, 12, 36], [0, 8, 16, 24], 50),                   # 3 boundaries
    ([0, 12, 24, 36], [0, 8, 16, 24], 1024),             # threshold > 10-bit register
])
def test_bad_registers_are_refused(cols, rows, shadow):
    """A grid off the frame or a value the register cannot hold is refused."""
    with pytest.raises(ValueError, match="auto_exposure_stats"):
        make_aes(np.zeros((24, 36), np.uint16), "rggb", cols, rows, shadow=shadow)

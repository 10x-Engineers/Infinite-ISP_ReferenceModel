"""
File: test_saturation_enhancement.py
Description: Unit tests for the saturation block against the golden of HDR-ISP's own
             RTL testbench (tb_axis_sat.sv):
                 chroma = clip( floor((c - offset) * gain / 256) + offset, 0, 2^BITS - 1 )
             for every 8-bit chroma value and SAT_GAIN register values across the clean
             range, plus the register quantisation and the refusals.
             Run from the repo root:  python -m pytest tests/test_saturation_enhancement.py
------------------------------------------------------------
"""
import sys

import numpy as np
import pytest

sys.path.append(".")
from modules import saturation_enhancement as se_mod  # pylint: disable=C0413


def chroma_gold(chroma, gain, offset=128):
    """tb_axis_sat.sv chroma_gold (integer, floor division)."""
    return min(max(((chroma - offset) * gain) // 256 + offset, 0), 255)


def make_se(img, gain, algorithm="global"):
    """SE object with minimal config dicts."""
    parm = {"is_enable": True, "is_save": False, "algorithm": algorithm,
            "saturation_gain": gain}
    return se_mod.SaturationEnhancement(img, {"in_file": "unit_test"}, {}, parm, None)


def all_chroma_image():
    """Every (Cb, Cr) value once each, Y a ramp."""
    chroma = np.arange(256, dtype=np.uint8)
    return np.stack([chroma[::-1], chroma, chroma[::-1]], axis=-1).reshape(16, 16, 3)


@pytest.mark.parametrize("register", [0, 1, 128, 255, 256, 300, 384, 511, 1024, 2000,
                                      3000, 3869, 3870])
def test_chroma_matches_tb_golden(register):
    """Every chroma value, Y untouched, at register values over the clean range."""
    img = all_chroma_image()
    out = make_se(img, register / 256).execute()
    assert np.array_equal(out[:, :, 0], img[:, :, 0])
    for plane in (1, 2):
        gold = [chroma_gold(int(c), register) for c in img[:, :, plane].ravel()]
        assert out[:, :, plane].ravel().tolist() == gold


@pytest.mark.parametrize("gain, register", [(1.0, 256), (1.5, 384), (1.3, 332), (0.0, 0),
                                            (15.12, 3870)])
def test_register_truncates_to_q4_8(gain, register):
    """saturation_gain becomes the 12-bit Q4.8 register, truncated."""
    assert make_se(all_chroma_image(), gain).sat_gain == register


def test_unity_gain_and_neutrals_are_identity():
    """Gain 1.0 changes nothing; a neutral pixel (Cb = Cr = 128) never changes."""
    img = all_chroma_image()
    assert np.array_equal(make_se(img, 1.0).execute(), img)
    neutral = np.full((4, 4, 3), 128, np.uint8)
    for gain in (0.0, 0.5, 1.5, 4.0, 15.0):
        assert np.array_equal(make_se(neutral, gain).execute(), neutral)


def test_wrap_above_the_clean_range():
    """Above the clean range the RTL's 12-bit add stage wraps: full chroma -> 0.
    That is why such a gain is refused."""
    limit = se_mod.max_clean_gain()
    full = np.array([255])
    assert se_mod.chroma_rtl(full, limit)[0] == 255
    assert se_mod.chroma_rtl(full, limit + 1)[0] == 0
    with pytest.raises(ValueError, match="clean range"):
        make_se(all_chroma_image(), (limit + 1) / 256)


def test_hue_rolloff_is_refused():
    """Only the RTL algorithm has a fixed-point design; the refusal names it."""
    with pytest.raises(ValueError, match="'global'"):
        make_se(all_chroma_image(), 1.5, algorithm="hue_rolloff")

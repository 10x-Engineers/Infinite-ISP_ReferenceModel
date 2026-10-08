"""
File: test_auto_exposure.py
Description: Unit tests for the AE control (HDR-ISP firmware EV law): the Q8 log2,
             the per-cell pixel counts from the grid, hand-worked meter and decision
             cases, and the gain-ladder bracket on a synthetic video sequence.
             Run from the repo root:  python -m pytest tests/test_auto_exposure.py
------------------------------------------------------------
"""
import contextlib
import io
import math
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.append(".")
from modules import auto_exposure as ae_mod  # pylint: disable=C0413
from modules.auto_exposure_stats import AutoExposureStats  # pylint: disable=C0413

BPP = 10
PARM_AE = {"is_enable": True, "is_debug": False, "target_mean": 200, "valid_min_pct": 10,
           "grid_weights": [1, 2, 1, 2, 4, 2, 1, 2, 1], "ev_deadband": 38, "ev_hyst": 90,
           "ev_damp": 50, "ev_damp_fast": 80, "ev_slew_max": 256, "ev_slew_fast": 512,
           "scene_change_ev": 512, "vm_iir_a": 40, "hi_frac_pm": 50, "hi_k": 100,
           "clip_frac_pm": 30, "clip_k": 150, "dark_frac_pm": 800, "ev_max_pull": 256}
GRID_4X4 = {"col_starts": [0, 4, 8, 12], "row_starts": [0, 4, 8, 12]}   # 16 px, 8 greens


def make_ae(stats, parm_aes=None, gains=(1, 2, 4, 8), index=0, state=None,
            temporal=False, **overrides):
    """AE object with minimal config dicts."""
    return ae_mod.AutoExposure(
        stats, {"bit_depth": BPP, "bayer_pattern": "grbg"}, dict(PARM_AE, **overrides),
        parm_aes or GRID_4X4, {"gain_array": list(gains), "current_gain": index},
        state if state is not None else ae_mod.new_ae_state(), temporal)


def stats_of(sums, white=None, high=None):
    """Statistics dict for the 4x4-cell grid from per-cell green sums."""
    zero = [0] * 9
    return {"shadow_count": zero, "highlight_count": high or zero, "grid_mean_sum": sums,
            "black_clip_count": zero, "white_clip_count": white or zero}


def test_log2_q8_matches_log2():
    """Exact on powers of two, never decreasing, and within the firmware's error of
    log2: the mantissa is truncated to 6 bits (up to log2(65/64)) and interpolated on
    an 8-entry table (chord + rounding + truncating division: under 2/256 EV)."""
    values = list(range(2, 70000))
    got = [ae_mod.log2_q8(v) for v in values]
    bound = 256 * math.log2(65 / 64) + 2
    assert all(ae_mod.log2_q8(1 << k) == 256 * k for k in range(1, 31))
    assert max(abs(g - 256 * math.log2(v)) for g, v in zip(got, values)) <= bound
    assert all(a <= b for a, b in zip(got, got[1:]))


@pytest.mark.parametrize("bayer", ["rggb", "grbg", "gbrg", "bggr"])
def test_cell_counts_match_the_pixels(bayer):
    """Green / total counts from the grid equal a count of the green positions."""
    parm = {"col_starts": [1, 6, 13, 20], "row_starts": [0, 3, 10, 17]}
    rows, cols = np.indices((17, 20))
    green = (rows + cols) % 2 == (1 if bayer in ("rggb", "bggr") else 0)
    greens, totals = ae_mod.cell_pixel_counts(parm, bayer)
    for cell in range(9):
        r_0, r_1 = parm["row_starts"][cell // 3], parm["row_starts"][cell // 3 + 1]
        c_0, c_1 = parm["col_starts"][cell % 3], parm["col_starts"][cell % 3 + 1]
        assert greens[cell] == int(green[r_0:r_1, c_0:c_1].sum())
        assert totals[cell] == (r_1 - r_0) * (c_1 - c_0)


def test_meter_centre_weighted_mean():
    """Cells at 100, centre at 500: (100 * 12 + 500 * 4) / 16 = 200."""
    sums = [800] * 9
    sums[4] = 4000
    assert make_ae(stats_of(sums)).meter()["vm"] == 200


def test_meter_removes_clipped_greens():
    """Centre: 6 greens at 400 + 2 clipped at 1023 (4 clipped pixels, all channels).
    The 2 clipped greens leave the sum and the count: 2400 / 6 = 400."""
    sums, white, high = [800] * 9, [0] * 9, [0] * 9
    sums[4], white[4], high[4] = 6 * 400 + 2 * 1023, 4, 2
    mtr = make_ae(stats_of(sums, white, high)).meter()
    assert mtr["vm"] == (100 * 12 + 400 * 4) // 16
    assert mtr["clip_frac_pm"] == 4 * 1000 // 144 and mtr["hi_frac_pm"] == 2 * 1000 // 72


@pytest.mark.parametrize("mtr, want, move", [
    ({"vm": 200, "hi_frac_pm": 0, "dark_frac_pm": 0, "clip_frac_pm": 0}, 0, 0),
    # 2 EV under: ev 512 (not above scene_change_ev), damp 50 % -> 1 EV move
    ({"vm": 50, "hi_frac_pm": 0, "dark_frac_pm": 0, "clip_frac_pm": 0}, 512, 256),
    # 1 EV over: -256, damp 50 % -> -128
    ({"vm": 400, "hi_frac_pm": 0, "dark_frac_pm": 0, "clip_frac_pm": 0}, -256, -128),
    # 1 EV over but 90 % dark: never darker -> 0, converged
    ({"vm": 400, "hi_frac_pm": 0, "dark_frac_pm": 900, "clip_frac_pm": 0}, 0, 0),
    # 2 EV under, clip fraction 2x its budget: the cap (-1.5 EV) is limited to
    # ev_max_pull (1 EV) below the mean's request -> 256, move 128
    ({"vm": 50, "hi_frac_pm": 0, "dark_frac_pm": 0, "clip_frac_pm": 64}, 256, 128),
])
def test_decide(mtr, want, move):
    """Hand-worked decisions (target 256 so that log2 is exact on powers of two)."""
    mtr = dict(mtr, valid=1, vm=mtr["vm"] * 256 // 200)
    ae_ctrl = make_ae(stats_of([800] * 9), target_mean=256, clip_frac_pm=32)
    assert ae_ctrl.decide(mtr) == (want, move)


def firmware_meter(stats, greens, totals, weights, pct, fsc):
    """The firmware's ae_meter as written in C (floor divisions), as the reference."""
    acc = wsum = 0
    for i, weight in enumerate(weights):
        wgc = min(stats["white_clip_count"][i] >> 1, stats["highlight_count"][i])
        bgc = min(stats["black_clip_count"][i] >> 1, stats["shadow_count"][i])
        vcnt = greens[i] - (wgc + bgc)
        if vcnt <= 0 or vcnt < greens[i] * pct // 100:
            continue
        acc += max(0, stats["grid_mean_sum"][i] - wgc * fsc) // vcnt * weight
        wsum += weight
    g_tot, p_tot = max(1, sum(greens)), max(1, sum(totals))
    return {"valid": 1 if wsum else 0, "vm": acc // wsum if wsum else 0,
            "hi_frac_pm": sum(stats["highlight_count"]) * 1000 // g_tot,
            "dark_frac_pm": sum(stats["shadow_count"]) * 1000 // g_tot,
            "clip_frac_pm": sum(stats["white_clip_count"]) * 1000 // p_tot}


def consistent_stats(rng, greens, totals, fsc):
    """Random statistics a statistics block could produce for these cells."""
    stats = {k: [] for k in ("shadow_count", "highlight_count", "grid_mean_sum",
                             "black_clip_count", "white_clip_count")}
    for gpc, total in zip(greens, totals):
        high = int(rng.integers(0, gpc + 1))
        stats["highlight_count"].append(high)
        stats["shadow_count"].append(int(rng.integers(0, gpc - high + 1)))
        stats["white_clip_count"].append(int(rng.integers(0, total + 1)))
        stats["black_clip_count"].append(int(rng.integers(0, total + 1)))
        stats["grid_mean_sum"].append(min(int(rng.integers(0, fsc * gpc + 1)), (1 << 32) - 1))
    return stats


def test_meter_hardware_form_equals_firmware():
    """The block's meter (multiply-compare valid test, one divider) equals the
    firmware's C meter on tiny random grids, where the valid-green boundary
    (vcnt + 1) * 100 == gpc * pct is hit often."""
    rng = np.random.default_rng(7)
    for _ in range(3000):
        cuts = [sorted(int(v) for v in rng.integers(0, 9, 2)) for _ in range(2)]
        parm_aes = {"col_starts": [0, cuts[0][0], cuts[0][1], 8],
                    "row_starts": [0, cuts[1][0], cuts[1][1], 8]}
        weights = [int(w) for w in rng.integers(0, 6, 9)]
        pct = int(rng.choice([0, 10, 20, 25, 50, 75, 100, int(rng.integers(0, 101))]))
        greens, totals = ae_mod.cell_pixel_counts(parm_aes, "grbg")
        stats = consistent_stats(rng, greens, totals, (1 << BPP) - 1)
        ae_ctrl = make_ae(stats, parm_aes, grid_weights=weights, valid_min_pct=pct)
        assert ae_ctrl.meter() == firmware_meter(stats, greens, totals, weights, pct,
                                                 (1 << BPP) - 1)


def test_divider_truncates_toward_zero():
    """Signed quotients truncate toward zero, as C does (Python's // floors)."""
    div = ae_mod.Divider()
    for num in range(-1000, 1001, 7):
        for den in (1, 3, 7, 100, 999):
            assert div.sdiv(num, den) == int(Fraction(num, den))


def extreme_block(rng, bit_depth, grid, weights, prm, stats_kind):
    """A block configured at the edges of its registers, on extreme legal statistics."""
    greens, totals = ae_mod.cell_pixel_counts(grid, "rggb")
    fsc, top = (1 << bit_depth) - 1, (1 << 32) - 1
    if stats_kind == "bright":
        stats = {"highlight_count": greens, "shadow_count": [0] * 9,
                 "white_clip_count": totals, "black_clip_count": [0] * 9,
                 "grid_mean_sum": [top] * 9}
    elif stats_kind == "dark":
        stats = {"highlight_count": [0] * 9, "shadow_count": greens,
                 "white_clip_count": [0] * 9, "black_clip_count": totals,
                 "grid_mean_sum": [0] * 9}
    elif stats_kind == "worst_mean":
        # one valid green per cell (the rest black-clipped) under a saturated sum:
        # every cell mean ~2^32, the largest weighted sum the guarantees allow
        stats = {"highlight_count": [0] * 9, "shadow_count": [g - 1 for g in greens],
                 "white_clip_count": [0] * 9,
                 "black_clip_count": [2 * (g - 1) for g in greens],
                 "grid_mean_sum": [top] * 9}
    else:
        stats = consistent_stats(rng, greens, totals, fsc)
    gains = [k / 256 for k in np.linspace(1, 65535, 128).astype(int)]
    return ae_mod.AutoExposure(
        stats, {"bit_depth": bit_depth, "bayer_pattern": "rggb"},
        dict(PARM_AE, grid_weights=weights, **prm), grid,
        {"gain_array": gains, "current_gain": int(rng.integers(0, 128))},
        ae_mod.new_ae_state(), bool(rng.integers(0, 2)))


def test_widths_hold_at_the_extremes():
    """No datapath value overflows its declared width - 16-bit sensor, 65535-pixel
    grid, every register at a limit, saturated / empty / random statistics - and a
    frame never needs more than 17 divisions."""
    rng = np.random.default_rng(3)
    big = {"col_starts": [0, 21845, 43690, 65535], "row_starts": [0, 21845, 43690, 65535]}
    ranges = ae_mod.register_ranges(16)
    worst = 0
    for trial in range(400):
        pick = "low" if trial % 3 == 0 else "high" if trial % 3 == 1 else "rand"
        prm = {name: (low if pick == "low" else high if pick == "high"
                      else int(rng.integers(low, high + 1)))
               for name, (_, low, high) in ranges.items()}
        weights = [255] * 9 if trial % 2 else [int(w) for w in rng.integers(0, 256, 9)]
        kind = ("bright", "dark", "random", "worst_mean")[trial % 4]
        if kind == "worst_mean":
            weights, prm["valid_min_pct"] = [255] * 9, 0
        state = None
        for _ in range(3):                           # a few frames: IIR, ladder state
            ae_ctrl = extreme_block(rng, 16, big, weights, prm, kind)
            if state is not None:
                ae_ctrl.state = state
            with contextlib.redirect_stdout(io.StringIO()):
                result = ae_ctrl.execute()
            state = ae_ctrl.state
            worst = max(worst, result["divisions"])
    assert worst <= 17


@pytest.mark.parametrize("override, match", [
    ({"target_mean": 0}, "target_mean"),
    ({"ev_damp": 101}, "ev_damp"),
    ({"hi_frac_pm": 1001}, "hi_frac_pm"),
    ({"vm_iir_a": 0}, "vm_iir_a"),
    ({"grid_weights": [1, 2, 1, 2, 256, 2, 1, 2, 1]}, "grid_weights"),
])
def test_out_of_range_registers_are_refused(override, match):
    """A value a register cannot hold is refused, naming the register and its range."""
    with pytest.raises(ValueError, match=match):
        make_ae(stats_of([800] * 9), **override)


@pytest.mark.parametrize("gains, index", [([1, 1.3, 2], 0), (list(range(1, 130)), 0),
                                          ([1, 2, 4], 3)])
def test_gain_table_limits_are_refused(gains, index):
    """Gains must be Q8 values (multiples of 1/256), at most 128 of them, and the
    current index must exist."""
    with pytest.raises(ValueError, match="digital_gain"):
        make_ae(stats_of([800] * 9), gains=gains, index=index)


def test_enabled_without_statistics_is_refused():
    """The control needs the statistics block, and says so."""
    with pytest.raises(ValueError, match="auto_exposure_stats.is_enable"):
        make_ae(None)


def actuate(index, came_from, age, mtr, ev_want, ev_move, hold=None):
    """One gain-ladder step on gains [1, 2, 4, 8] from a crafted controller state."""
    state = dict(ae_mod.new_ae_state(), came_from=came_from, age=age, hold=hold)
    ae_ctrl = make_ae(stats_of([800] * 9), index=index, state=state)
    mtr = dict({"hi_frac_pm": 0, "dark_frac_pm": 0, "clip_frac_pm": 0}, **mtr)
    return ae_ctrl.actuate(mtr, ev_want, ev_move), state


def test_bracket_after_dwell_holds_the_darker_end():
    """Came to 4x from 2x (wanting +200) three frames ago; now 4x breaks the clip
    budget and asks for 2x.  The bracket holds 2x - its reference the raw mean
    metered at 2x - and keeps it there while that mean stays; a scene change
    releases it."""
    raw_2x = ae_mod.log2_q8(150)
    best, state = actuate(2, (1, 200, False, raw_2x), 3,
                          {"vm": 300, "clip_frac_pm": 100}, -256, -128)
    assert best == 1 and state["hold"] == (1, 2, raw_2x)
    # at 2x, same scene: the loop asks for 4x again - held
    best, state = actuate(1, state["came_from"], state["age"], {"vm": 150}, 200, 100,
                          hold=state["hold"])
    assert best == 1
    # at 2x, the scene darkened (raw mean halved): released, the loop moves to 4x
    best, state = actuate(1, state["came_from"], state["age"], {"vm": 75}, 200, 100,
                          hold=state["hold"])
    assert best == 2 and state["hold"] is None


def test_ties_go_to_the_first_gain_and_the_darker_end():
    """Spec tie rules: the scan keeps the FIRST gain at the smallest distance (aim
    1.5 EV from 1x lies equally between 2x and 4x -> 2x); a bracket with equal errors
    at both ends settles on the darker end."""
    best, _ = actuate(0, None, 0, {"vm": 100}, 600, 384)
    assert best == 1
    best, state = actuate(2, (1, 200, False, ae_mod.log2_q8(150)), 1, {"vm": 290}, -200, -100)
    assert best == 1 and state["hold"][0] == 1


def test_bracket_found_after_a_dwell_frame_by_frame():
    """From a clean state: 2x asks for 4x, 4x dwells a frame (inside the deadband
    while the IIR catches up), then breaks the clip budget and asks for 2x - the
    reversal is still recognised and 2x is held."""
    raw_2x = ae_mod.log2_q8(150)
    best, state = actuate(1, None, 0, {"vm": 150}, 200, 100)
    assert best == 2
    best, state = actuate(2, state["came_from"], state["age"], {"vm": 300}, 20, 0)
    assert best == 2
    best, state = actuate(2, state["came_from"], state["age"],
                          {"vm": 300, "clip_frac_pm": 100}, -256, -128)
    assert best == 1 and state["hold"] == (1, 2, raw_2x)


@pytest.mark.parametrize("age, best", [(1, 2), (3, 1)])
def test_stay_needs_a_fresh_reading_of_the_other_end(age, best):
    """At 4x, asking for 2x; 2x (left wanting +300) had the larger error, nothing is
    over budget, so the bracket would stay at 4x.  Only on a reading of 2x from the
    previous frame (age 1); an older one may be stale, so the move goes ahead."""
    got, state = actuate(2, (1, 300, False, ae_mod.log2_q8(150)), age, {"vm": 290}, -200, -100)
    assert got == best
    assert (state["hold"] is not None) == (best == 2)


def scene(level, gain):
    """Synthetic frame at `level`, with 1 pixel in 8 (greens and non-greens) at 3x,
    times the gain, clipped like digital gain; its AE statistics."""
    img = np.full((24, 24), level, np.float64)
    img[::4, ::4] = level * 3
    img[1::4, 2::4] = level * 3
    img = np.uint16(np.clip(img * gain, 0, (1 << BPP) - 1))
    parm = {"is_enable": True, "is_save": False, "is_debug": False, "col_starts": [0, 8, 16, 24],
            "row_starts": [0, 8, 16, 24], "shadow_threshold": 50, "highlight_threshold": 950}
    return AutoExposureStats(img, {"in_file": "unit_test"},
                             {"bit_depth": BPP, "bayer_pattern": "grbg"}, parm, None
                             ).compute_grid_stats()


def test_gain_ladder_settles_between_two_gains():
    """Video, IIR on: the target lies between 2x (under) and 4x (the bright pixels
    clip, over the clip budget).  The first reversal back to 2x - after dwelling at
    4x while the IIR catches up - settles the bracket on 2x for good; when the scene
    darkens 1 EV it re-exposes."""
    gains, parm_aes = [1, 2, 4, 8], {"col_starts": [0, 8, 16, 24], "row_starts": [0, 8, 16, 24]}
    state, index, trail = ae_mod.new_ae_state(), 0, []
    for frame in range(36):
        level = 100 if frame < 24 else 50
        ae_ctrl = make_ae(scene(level, gains[index]), parm_aes, gains, index, state,
                          temporal=True, target_mean=400)
        index = ae_ctrl.execute()["gain_index"]
        trail.append(gains[index])
    assert trail[:3] == [2, 4, 4], trail
    assert trail[3:24] == [2] * 21, trail
    assert trail[-1] > 2, trail

"""
File: auto_exposure.py
Description: 3A-AE control block - the HDR-ISP auto-exposure control law (ae_ctrl.c,
             "EV law") as an RTL block driving Infinite-ISP's digital gain.  This file is
             the bit-accurate specification of that block.

             Timing: the statistics block (auto_exposure_stats) latches frame N's 45
             values at the start of frame N+1; this block then runs once, as a
             sequential state machine during vertical blanking, and its gain index is
             applied by digital gain from the next frame on (compute N, apply N+1).

             Datapath: ONE shared divider (unsigned DIV_NUM_BITS / DIV_DEN_BITS,
             restoring, one quotient bit per cycle; a signed dividend goes through it as
             sign-magnitude and the quotient truncates toward zero, like C), the Q8 log2
             unit (leading-one position + 9-entry table, linear interpolation), adders,
             multipliers and comparators.  At most 17 divisions per frame: 9 cell means,
             the weighted mean, 3 fractions, the IIR step, 2 caps, the damping.

             Per frame:
             1. Meter: per grid cell (sequentially, cell index = row * 3 + col), a
                clip-corrected green mean - saturated greens are removed from the sum
                (the all-pixel clip counts halved to green and bounded by the
                green-only counters), cells with too few valid greens are dropped -
                then a centre-weighted mean `vm` over the cells.  The green / total
                pixel count of each cell comes from the grid registers (the statistics
                block has no divider and no pixel counters).  Plus three FRAME-level
                fractions, in per mille: greens above the highlight threshold, pixels
                at full scale, greens below the shadow threshold.
             2. Decide: the exposure error in EV,
                    ev = log2(target_mean) - log2(vm)                (Q8: 256 = 1 EV)
                capped DOWN when the highlight or clip fraction is over its budget,
                floored at 0 when most of the frame is dark (a dark frame is never
                driven darker).  Inside the deadband (+ hysteresis once converged)
                nothing moves; otherwise a damped, slew-limited EV move is made.
             3. Actuate: a scan of the gain table (digital_gain.gain_array, Q8) for
                the gain nearest the requested exposure, in the same Q8 log2.  A target
                between two gains is settled on one of them (the smaller error, never
                a brighter gain that breaks the highlight / clip budget) instead of
                alternating; when the ladder cannot get closer the loop has converged
                as far as the gains allow.

             Registers (config, refused outside these ranges):
               target_mean      BITS  1 .. 2^BITS-1    valid_min_pct    7   0 .. 100
               grid_weights     9 x 8 0 .. 255         ev_deadband      10  0 .. 1023
               ev_hyst          10    0 .. 1023        ev_damp          7   0 .. 100
               ev_damp_fast     7     0 .. 100         ev_slew_max      12  1 .. 4095
               ev_slew_fast     12    1 .. 4095        scene_change_ev  12  0 .. 4095
               vm_iir_a         7     1 .. 100         hi_frac_pm       10  0 .. 1000
               hi_k             8     0 .. 255         clip_frac_pm     10  0 .. 1000
               clip_k           8     0 .. 255         dark_frac_pm     10  0 .. 1000
               ev_max_pull      12    0 .. 4095        iir bypass       1   (render_3a)
               gain table       up to 128 x 16-bit unsigned Q8.8 (gain_array, multiples
                                of 1/256 from 1/256 to 255.996); current index 7 bit
               grid             the statistics block's col_starts / row_starts (16 bit)

             Internal widths (unsigned unless signed; every value is checked against
             its width and the check raises OverflowError, so the tests prove them):
               cell pixel / green count  32      frame green / pixel total  36
               frame stat sums           36      vcnt (valid greens)        34 signed
               valid-green compare       41 signed (vcnt + 1) * 100 vs gpc * pct
               cell numerator            32      cell mean                  32
               weighted sum              44      weight sum                 12
               vm                        32      vm_filt (Q4 state)         36
               fractions (per mille)     10      log2 Q8 out                13
               EV error / move           16 signed  gain-table EV           12 signed
               held / came-from raw log2 13      came-from age              2 (saturating)
             The fraction widths rely on the statistics block's guarantees: a cell's
             shadow / highlight counts never exceed its greens, its clip counts never
             exceed its pixels.

             The arithmetic is the firmware's exactly (mirrored in HDR-ISP
             verify/ae_loop_sim.py) - where the hardware form differs, it is an exact
             rewrite (the valid-green test is a multiply-compare) - generalised from the
             firmware's fixed 10-bit, fixed-grid constants to this frame's bit depth
             and grid.

Code / Paper  Reference: HDR-ISP firmware ae_ctrl.c (AE_CTRL_V2) / verify/ae_loop_sim.py
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time

from modules.auto_exposure_stats import (GRID_COLS, GRID_REG_BITS, GRID_ROWS, MEAN_BITS,
                                         green_parity)

EV_ONE = 256                                  # Q8: 256 = 1 EV
_K_LOG2 = [0, 44, 82, 118, 150, 179, 207, 232, 256]
N_CELLS = GRID_ROWS * GRID_COLS

# internal widths (see the table above)
CELL_BITS = 2 * GRID_REG_BITS                 # pixels of a cell
TOT_BITS = CELL_BITS + 4                      # sum over 9 cells
VCNT_BITS = CELL_BITS + 2                     # signed
WEIGHT_BITS = 8
ACC_BITS = MEAN_BITS + WEIGHT_BITS + 4
WSUM_BITS = WEIGHT_BITS + 4
FRAC_BITS = 10
LOG2_IN_BITS, LOG2_OUT_BITS = 32, 13
EV_BITS = 16                                  # signed
GAIN_Q8_BITS, MAX_GAINS = 16, 128
GAIN_EV_BITS = 12                             # signed
AGE_MAX = 3                                   # 2-bit saturating counter (only age == 1 is used)
DIV_NUM_BITS, DIV_DEN_BITS = 48, 36


def register_ranges(bit_depth):
    """Register widths and legal ranges: name -> (bits, low, high)."""
    return {"target_mean": (bit_depth, 1, (1 << bit_depth) - 1),
            "valid_min_pct": (7, 0, 100), "ev_deadband": (10, 0, 1023),
            "ev_hyst": (10, 0, 1023), "ev_damp": (7, 0, 100), "ev_damp_fast": (7, 0, 100),
            "ev_slew_max": (12, 1, 4095), "ev_slew_fast": (12, 1, 4095),
            "scene_change_ev": (12, 0, 4095), "vm_iir_a": (7, 1, 100),
            "hi_frac_pm": (10, 0, 1000), "hi_k": (8, 0, 255),
            "clip_frac_pm": (10, 0, 1000), "clip_k": (8, 0, 255),
            "dark_frac_pm": (10, 0, 1000), "ev_max_pull": (12, 0, 4095)}


def fit(value, bits, signed=False):
    """A datapath value checked against its register width."""
    low, high = ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed
                 else (0, (1 << bits) - 1))
    if not low <= value <= high:
        raise OverflowError(f"{value} does not fit a {'signed ' if signed else ''}"
                            f"{bits}-bit register")
    return value


class Divider:
    """The block's one shared divider; counts its operations per frame."""

    def __init__(self):
        self.ops = 0

    def udiv(self, num, den):
        """Unsigned quotient (den > 0 by construction of every caller)."""
        fit(num, DIV_NUM_BITS)
        fit(den, DIV_DEN_BITS)
        self.ops += 1
        return num // den

    def sdiv(self, num, den):
        """Signed dividend, positive divisor: sign-magnitude, truncating toward zero."""
        quo = self.udiv(abs(num), den)
        return -quo if num < 0 else quo


def log2_q8(value):
    """Q8 log2 unit: exponent = leading-one position, 6 mantissa bits below it, the
    top 3 index the table, the low 3 interpolate linearly."""
    fit(value, LOG2_IN_BITS)
    if value < 2:
        return 0
    exp = value.bit_length() - 1
    frac = ((value >> (exp - 6)) & 63) if exp >= 6 else ((value << (6 - exp)) & 63)
    idx, rem = frac >> 3, frac & 7
    mant = _K_LOG2[idx] + (((_K_LOG2[idx + 1] - _K_LOG2[idx]) * rem) >> 3)
    return fit(exp * EV_ONE + mant, LOG2_OUT_BITS)


def cell_pixel_counts(parm_aes, bayer):
    """(green, total) pixel count of each grid cell, from the grid registers.  An
    odd-sized cell has one more pixel of the phase at its first corner."""
    cols, rows = parm_aes["col_starts"], parm_aes["row_starts"]
    greens, totals = [], []
    for r_idx in range(GRID_ROWS):
        for c_idx in range(GRID_COLS):
            r_0, c_0 = int(rows[r_idx]), int(cols[c_idx])
            total = fit((int(rows[r_idx + 1]) - r_0) * (int(cols[c_idx + 1]) - c_0), CELL_BITS)
            green = total >> 1
            if total % 2 and (r_0 + c_0) % 2 == green_parity(bayer):
                green += 1
            greens.append(green)
            totals.append(total)
    return greens, totals


def new_ae_state():
    """The block's state registers, kept from frame to frame: the filtered mean and
    convergence flag, plus the gain ladder's (gain it came from, how many frames ago,
    held bracket)."""
    return {"vm_filt_q4": 0, "converged": 0, "came_from": None, "age": 0, "hold": None}


class AutoExposure:
    """
    Auto Exposure control block (EV law) on the AE statistics
    """

    def __init__(self, ae_stats, sensor_info, parm_ae, parm_aes, parm_dga, state, temporal):
        self.stats = ae_stats
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.bit_depth = sensor_info["bit_depth"]
        self.full_scale = (1 << self.bit_depth) - 1
        self.ranges = register_ranges(self.bit_depth)
        self.prm = {k: int(parm_ae[k]) for k in self.ranges}
        self.weights = [int(w) for w in parm_ae["grid_weights"]]
        self.greens, self.totals = cell_pixel_counts(parm_aes, sensor_info["bayer_pattern"])
        self.g_tot = fit(max(1, sum(self.greens)), TOT_BITS)
        self.p_tot = fit(max(1, sum(self.totals)), TOT_BITS)
        self.gains = list(parm_dga["gain_array"])
        self.gain_index = int(parm_dga["current_gain"])
        self.state = state
        # temporal=False is the IIR-bypass register: the same frame is processed again
        # (render_3a), so there is no frame-to-frame noise for the IIR to filter
        self.temporal = temporal
        self.div = Divider()
        self.gains_q8 = []
        if self.enable:
            if self.stats is None:
                raise ValueError("auto_exposure needs the AE statistics: set "
                                 "auto_exposure_stats.is_enable to true.")
            self.check_registers()
            self.gains_q8 = [int(g * EV_ONE) for g in self.gains]

    def check_registers(self):
        """Refuse a configuration the block's registers cannot hold, naming the fix."""
        for name, (bits, low, high) in self.ranges.items():
            if not low <= self.prm[name] <= high:
                raise ValueError(f"auto_exposure.{name} = {self.prm[name]} does not fit its "
                                 f"{bits}-bit register: use {low} .. {high}.")
        if len(self.weights) != N_CELLS or \
                not all(0 <= w < (1 << WEIGHT_BITS) for w in self.weights):
            raise ValueError(f"auto_exposure.grid_weights needs {N_CELLS} weights of "
                             f"0 .. {(1 << WEIGHT_BITS) - 1} (cell index = row * 3 + col).")
        if not 1 <= len(self.gains) <= MAX_GAINS or not all(
                g * EV_ONE == int(g * EV_ONE) and 1 <= g * EV_ONE < (1 << GAIN_Q8_BITS)
                for g in self.gains):
            raise ValueError(f"digital_gain.gain_array must hold 1 .. {MAX_GAINS} gains, each "
                             f"a multiple of 1/256 from 1/256 to 255.996 (the AE block's "
                             f"16-bit Q8.8 gain table).")
        if not 0 <= self.gain_index < len(self.gains):
            raise ValueError(f"digital_gain.current_gain = {self.gain_index} is not an index "
                             f"of gain_array (0 .. {len(self.gains) - 1}).")

    # ------------------------------------------------------------------ 1. meter
    def meter(self):
        """Clip-corrected, centre-weighted green mean + frame fractions."""
        st, fsc, prm, div = self.stats, self.full_scale, self.prm, self.div
        acc = wsum = hi_cnt = wc_cnt = dk_cnt = 0
        for i in range(N_CELLS):
            gpc, weight = self.greens[i], self.weights[i]
            hi_cnt = fit(hi_cnt + st["highlight_count"][i], TOT_BITS)
            wc_cnt = fit(wc_cnt + st["white_clip_count"][i], TOT_BITS)
            dk_cnt = fit(dk_cnt + st["shadow_count"][i], TOT_BITS)
            # all-pixel clip counts halved to green, bounded by the green-only
            # counters (a green at full scale is necessarily a highlight, at 0 a shadow)
            wgc = min(st["white_clip_count"][i] >> 1, st["highlight_count"][i])
            bgc = min(st["black_clip_count"][i] >> 1, st["shadow_count"][i])
            vcnt = fit(gpc - (wgc + bgc), VCNT_BITS, signed=True)
            # firmware: skip if vcnt <= 0 or vcnt < floor(gpc * pct / 100); for an
            # integer vcnt the second test is exactly (vcnt + 1) * 100 <= gpc * pct
            if vcnt <= 0 or (vcnt + 1) * 100 <= gpc * prm["valid_min_pct"]:
                continue
            num = fit(max(0, st["grid_mean_sum"][i] - wgc * fsc), MEAN_BITS)
            cell_mean = fit(div.udiv(num, vcnt), MEAN_BITS)
            acc = fit(acc + cell_mean * weight, ACC_BITS)
            wsum = fit(wsum + weight, WSUM_BITS)
        return {"valid": 1 if wsum else 0,
                "vm": fit(div.udiv(acc, wsum), MEAN_BITS) if wsum else 0,
                "hi_frac_pm": fit(div.udiv(hi_cnt * 1000, self.g_tot), FRAC_BITS),
                "dark_frac_pm": fit(div.udiv(dk_cnt * 1000, self.g_tot), FRAC_BITS),
                "clip_frac_pm": fit(div.udiv(wc_cnt * 1000, self.p_tot), FRAC_BITS)}

    # ------------------------------------------------------------------ 2. decide
    def decide(self, mtr):
        """Decision: (ev_want, ev_move) in Q8 EV; ev_move 0 = hold."""
        prm, state, div = self.prm, self.state, self.div
        vm_q4 = fit((mtr["vm"] if mtr["vm"] else 1) << 4, MEAN_BITS + 4)
        iir_a = prm["vm_iir_a"] if self.temporal else 100
        if not state["vm_filt_q4"]:
            state["vm_filt_q4"] = vm_q4
        else:
            step = div.sdiv((vm_q4 - state["vm_filt_q4"]) * iir_a, 100)
            state["vm_filt_q4"] = fit(state["vm_filt_q4"] + step, MEAN_BITS + 4)
        vmf = max(1, state["vm_filt_q4"] >> 4)

        if not mtr["valid"]:
            ev_want = (-prm["ev_slew_max"] if mtr["clip_frac_pm"] > mtr["dark_frac_pm"]
                       else prm["ev_slew_max"])
        else:
            ev_mean = log2_q8(prm["target_mean"]) - log2_q8(vmf)
            ev_want = ev_mean
            if mtr["hi_frac_pm"]:                 # passes through 0 at the budget
                over = log2_q8(mtr["hi_frac_pm"]) - log2_q8(prm["hi_frac_pm"])
                ev_want = min(ev_want, div.sdiv(-(over * prm["hi_k"]), 100))
            if mtr["clip_frac_pm"]:
                over = log2_q8(mtr["clip_frac_pm"]) - log2_q8(prm["clip_frac_pm"])
                ev_want = min(ev_want, div.sdiv(-(over * prm["clip_k"]), 100))
            ev_want = max(ev_want, ev_mean - prm["ev_max_pull"])
            if mtr["dark_frac_pm"] > prm["dark_frac_pm"] and ev_want < 0:
                ev_want = 0
        fit(ev_want, EV_BITS, signed=True)

        band = prm["ev_deadband"] + (prm["ev_hyst"] if state["converged"] else 0)
        if abs(ev_want) <= band:
            state["converged"] = 1
            return ev_want, 0
        state["converged"] = 0
        damp, slew = prm["ev_damp"], prm["ev_slew_max"]
        if abs(ev_want) > prm["scene_change_ev"]:
            damp, slew = prm["ev_damp_fast"], prm["ev_slew_fast"]
        ev_move = max(-slew, min(slew, div.sdiv(ev_want * damp, 100)))
        if ev_move == 0:
            ev_move = 1 if ev_want > 0 else -1
        return ev_want, fit(ev_move, EV_BITS, signed=True)

    # ------------------------------------------------------------------ 3. actuate
    def gain_ev(self, index):
        """Q8 EV of a gain-table entry: log2 of its Q8 value, minus log2(256)."""
        return fit(log2_q8(self.gains_q8[index]) - 8 * EV_ONE, GAIN_EV_BITS, signed=True)

    def actuate(self, mtr, ev_want, ev_move):
        """Move along the gain table.

        The gain nearest the requested exposure (a scan of the table; the first
        entry wins a tie); if that is the current gain, still one step when it lands
        closer to the full target (the firmware's minimum move).  The ladder is
        coarse where the firmware's shutter is fine, so a target can fall BETWEEN two
        gains: the loop then asks to go back to the gain it came from (A -> B -> A),
        the error having changed sign - possibly after dwelling at B for a few frames
        while the mean's IIR catches up.  That bracket is settled on the end with the
        smaller error (the darker end on a tie) - never on the brighter end if it
        breaks the highlight or clip budget (HDR-ISP's default: protect highlights) -
        and held while the scene stays put: the RAW metered mean at the held gain
        within the deadband (the filtered error still drifts for frames after a move,
        so it cannot tell a scene change from the filter catching up).  The reference
        of the end held is the raw mean metered AT that gain: when it is the end being
        moved to, its reference is checked on arrival; staying at the current end is
        decided only when the other end was metered on the previous frame, so an old
        reading can never pin the gain."""
        cur, state, prm = self.gain_index, self.state, self.prm
        over = (mtr["hi_frac_pm"] > prm["hi_frac_pm"] or
                mtr["clip_frac_pm"] > prm["clip_frac_pm"])
        ev_raw = log2_q8(max(1, mtr["vm"]))
        best = cur
        if ev_move != 0:
            aim = self.gain_ev(cur) + ev_move
            best_dist = None
            for k in range(len(self.gains_q8)):
                dist = abs(self.gain_ev(k) - aim)
                if best_dist is None or dist < best_dist:
                    best, best_dist = k, dist
            if best == cur:
                step = cur + (1 if ev_move > 0 else -1)
                target = self.gain_ev(cur) + ev_want
                if 0 <= step < len(self.gains_q8) and \
                        abs(self.gain_ev(step) - target) < abs(self.gain_ev(cur) - target):
                    best = step

        # an end: (gain index, ev_want, over budget, log2 raw mean) metered at a gain
        entry = (cur, ev_want, over, ev_raw)
        hold, came = state["hold"], state["came_from"]
        if hold and hold[0] == cur and abs(ev_raw - hold[2]) > prm["ev_deadband"]:
            hold = came = None                            # the scene moved: release
        if hold and hold[0] == cur and best == hold[1]:
            best = cur                                    # still bracketed: stay
        elif came and best != cur and best == came[0] and (came[1] > 0) != (ev_want > 0):
            entry_darker = self.gains_q8[entry[0]] <= self.gains_q8[came[0]]
            darker, brighter = (entry, came) if entry_darker else (came, entry)
            pick = darker if brighter[2] or abs(darker[1]) <= abs(brighter[1]) else brighter
            if pick is came or state["age"] == 1:
                other = brighter if pick is darker else darker
                hold, best = (pick[0], other[0], pick[3]), pick[0]
        if best != cur:
            came, state["age"] = entry, 0
        state["age"] = min(state["age"] + 1, AGE_MAX)
        state["hold"], state["came_from"] = hold, came
        return best

    def execute(self):
        """
        Execute AE: returns the decision for the next frame, or None when disabled
        """
        print("Auto Exposure (EV law) = " + str(self.enable))
        if not self.enable:
            return None
        start = time.time()
        self.div.ops = 0
        mtr = self.meter()
        ev_want, ev_move = self.decide(mtr)
        index = self.actuate(mtr, ev_want, ev_move)
        result = dict(mtr, target_mean=self.prm["target_mean"], ev_err=ev_want,
                      ev_move=ev_move, converged=bool(self.state["converged"]),
                      gain_index=index, gain=self.gains[index],
                      moved=index != self.gain_index, divisions=self.div.ops)
        if self.is_debug:
            print(f"   - AE - metered mean / target   = {mtr['vm']} / {self.prm['target_mean']}")
            print(f"   - AE - highlight / clip / dark = {mtr['hi_frac_pm']} / "
                  f"{mtr['clip_frac_pm']} / {mtr['dark_frac_pm']} per mille")
            print(f"   - AE - EV error / move (Q8)    = {ev_want} / {ev_move}  "
                  f"({ev_want / EV_ONE:+.2f} / {ev_move / EV_ONE:+.2f} EV)")
            print(f"   - AE - digital gain            = {self.gains[self.gain_index]} -> "
                  f"{self.gains[index]}  ({self.div.ops} divisions)")
        print(f"  Execution time: {time.time() - start:.3f}s")
        return result

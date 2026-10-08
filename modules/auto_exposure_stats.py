"""
File: auto_exposure_stats.py
Description: 3A - AE statistics block (HDR-ISP axis_ae_stat).  A statistics sink on
             the raw Bayer frame: over a 3x3 grid of cells it accumulates, per cell
               shadow_count     : green pixels with value <  shadow_threshold
               highlight_count  : green pixels with value >  highlight_threshold
               grid_mean_sum    : sum of the green pixel values
               black_clip_count : pixels of any channel with value == 0
               white_clip_count : pixels of any channel with value == 2^bit_depth - 1
             Luma statistics are green-only (green is the luma proxy before white
             balance); clip counts take every channel.  Comparisons are strict, so a
             pixel equal to a threshold is in neither count.

             Bit-exact model of the RTL:
               - cells are half-open intervals [start, next start) of the col_starts /
                 row_starts registers (16 bit); pixels outside the grid are not counted
               - green = the Bayer phase of (row, col), as the RTL's bayer decode
               - the four counters are 32-bit (modulo 2^32 - no realistic frame wraps)
               - grid_mean_sum is a 32-bit SATURATING accumulator: a wrapped bright
                 cell would read dark and drive the AE the wrong way
               - the block has no divider: the AE firmware divides the sum by the
                 green count it knows from the grid geometry
             The RTL latches a frame's statistics at the next frame start, so they
             steer the frame after - as in this model, where the AE decision of a
             frame sets the digital gain of the next.

             Where it runs: on the digital-gain output (linear, black-level
             corrected, before noise reduction).  Digital gain stands in for the
             sensor exposure that the HDR-ISP firmware AE moves, so the statistics
             must see it.

Code / Paper  Reference: HDR-ISP RTL axis/rtl/axis_ae_stat.v
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

GRID_ROWS = GRID_COLS = 3
COUNT_BITS = 32                # RTL per-cell counters (wrap)
MEAN_BITS = 32                 # RTL per-cell green-sum accumulator (saturates)
GRID_REG_BITS = 16             # RTL grid boundary registers
STAT_KEYS = ("shadow_count", "highlight_count", "grid_mean_sum",
             "black_clip_count", "white_clip_count")


def green_parity(bayer):
    """(row + col) % 2 of the green pixels: 1 for RGGB / BGGR, 0 for GRBG / GBRG."""
    return 1 if bayer in ("rggb", "bggr") else 0


class AutoExposureStats:
    """
    AE statistics on a 3x3 grid of the Bayer raw (green-only luma, all-channel clips)
    """

    def __init__(self, img, platform, sensor_info, parm_aes, save_out_obj):
        self.img = img
        self.platform = platform
        self.sensor_info = sensor_info
        self.enable = parm_aes["is_enable"]
        self.is_save = parm_aes["is_save"]
        self.is_debug = parm_aes["is_debug"]
        self.bit_depth = sensor_info["bit_depth"]
        self.bayer = sensor_info["bayer_pattern"]
        self.save_out_obj = save_out_obj

        # registers: 4 column / 4 row boundaries (the last closes the grid), thresholds
        self.col_starts = [int(c) for c in parm_aes["col_starts"]]
        self.row_starts = [int(r) for r in parm_aes["row_starts"]]
        self.shadow_threshold = int(parm_aes["shadow_threshold"])
        self.highlight_threshold = int(parm_aes["highlight_threshold"])
        if self.enable:
            self.check_registers()

    def check_registers(self):
        """Refuse register values the RTL cannot hold or a grid off the frame,
        naming the fix."""
        height, width = self.img.shape[:2]
        for name, starts, size in (("col_starts", self.col_starts, width),
                                   ("row_starts", self.row_starts, height)):
            if len(starts) != GRID_COLS + 1 or starts != sorted(starts) or \
                    starts[0] < 0 or starts[-1] > min(size, 2**GRID_REG_BITS - 1):
                raise ValueError(
                    f"auto_exposure_stats.{name} = {starts} must be 4 ascending "
                    f"boundaries within 0..{size} (the frame after crop), e.g. "
                    f"[0, {size // 3}, {2 * size // 3}, {size}].")
        full_scale = (1 << self.bit_depth) - 1
        for name in ("shadow_threshold", "highlight_threshold"):
            if not 0 <= getattr(self, name) <= full_scale:
                raise ValueError(
                    f"auto_exposure_stats.{name} = {getattr(self, name)} does not fit "
                    f"the {self.bit_depth}-bit register: use 0..{full_scale}.")

    def compute_grid_stats(self):
        """The five statistics of each of the 9 cells, cell index = row * 3 + col."""
        img = self.img
        rows, cols = np.indices(img.shape[:2])
        green = (rows + cols) % 2 == green_parity(self.bayer)
        full_scale = (1 << self.bit_depth) - 1
        stats = {key: [0] * (GRID_ROWS * GRID_COLS) for key in STAT_KEYS}
        for r_idx in range(GRID_ROWS):
            r_0, r_1 = self.row_starts[r_idx], self.row_starts[r_idx + 1]
            for c_idx in range(GRID_COLS):
                c_0, c_1 = self.col_starts[c_idx], self.col_starts[c_idx + 1]
                cell = r_idx * GRID_COLS + c_idx
                roi = img[r_0:r_1, c_0:c_1]
                greens = roi[green[r_0:r_1, c_0:c_1]]
                counts = {
                    "shadow_count": np.sum(greens < self.shadow_threshold),
                    "highlight_count": np.sum(greens > self.highlight_threshold),
                    "black_clip_count": np.sum(roi == 0),
                    "white_clip_count": np.sum(roi == full_scale),
                }
                for key, count in counts.items():
                    stats[key][cell] = int(count) % (1 << COUNT_BITS)
                # adds are non-negative: once saturated the RTL stays at full scale
                stats["grid_mean_sum"][cell] = min(int(np.sum(greens, dtype=np.uint64)),
                                                   (1 << MEAN_BITS) - 1)
        return stats

    def save(self, stats):
        """
        Function to save module output
        """
        if self.is_save:
            self.save_out_obj.save_output_3a(
                self.platform["in_file"], stats, "Out_auto_exposure_stats_"
            )

    def execute(self):
        """
        Execute AE statistics: returns the statistics dict, or None when disabled
        """
        print("Auto Exposure Statistics = " + str(self.enable))
        if not self.enable:
            return None
        start = time.time()
        stats = self.compute_grid_stats()
        if self.is_debug:
            print(f"   - AES - shadow / highlight threshold = "
                  f"{self.shadow_threshold} / {self.highlight_threshold}")
            for key in STAT_KEYS:
                print(f"   - AES - {key:17s}= {stats[key]}")
        print(f"  Execution time: {time.time() - start:.3f}s")
        self.save(stats)
        return stats

"""
File: ldci.py
Description: Implements Local Dynamic Contrast Improvement (LDCI) using
             Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
             Y channel, in fixed point.
Code / Paper  Reference:
https://arxiv.org/ftp/arxiv/papers/2108/2108.12818.pdf
Ported from the Infinite-ISP (float) algorithm-design model to RM's
fixed-point register conventions.
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import math
import time
import numpy as np
from util.utils import get_approximate

# Histogram is always taken over the full 8-bit Y range -> 256 bins. This is
# also why the "+1 pixel per bin" bias and the clipped-pixel redistribution
# below divide by exactly 256: that division is a right-shift by FRAC_BITS,
# not a real divider.
NUM_BINS = 256
FRAC_BITS = 8  # log2(NUM_BINS); keeps the "/256" redistribution step exact

# Register width for the per-tile histogram bin counts. A tile holds wind*wind
# pixels, so wind is bounded so that wind**2 fits comfortably in this width.
HIST_REG_BITS = 16

# Register/fraction widths for the reciprocal-multiply that replaces the
# float "pdf = hist / hist.sum()" divide. hist.sum() after redistribution is
# always the compile-time constant (wind**2 + NUM_BINS), so its reciprocal
# can be precomputed once per module instance, the same way
# BlackLevelCorrection precomputes its linearization factor.
RECIP_REG_BITS = 32
RECIP_FRAC_BITS = 24

# Q0.10 interpolation weights (1024 = unity), matching the reference model.
WEIGHT_FRAC_BITS = 10
WEIGHT_UNITY = 1 << WEIGHT_FRAC_BITS


class LDCI:
    """
    Local Dynamic Contrast Improvement (fixed-point CLAHE on the Y plane)
    """

    def __init__(self, img, platform, sensor_info, parm_ldci, save_out_obj):
        self.img = img.copy()
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ldci = parm_ldci
        self.wind = parm_ldci["wind"]
        self.clip_limit = parm_ldci["clip_limit"]
        self.is_save = parm_ldci["is_save"]
        self.platform = platform
        self.save_out_obj = save_out_obj

        self.validate_registers()

        # Clip-limit fallback, mirrored from the float model: a clip limit
        # that is not tighter than the tile itself is clamped to 8% of wind.
        # This is a config-time (not per-pixel) constant, so it is rounded
        # once here rather than carried through the pipeline as a fraction.
        if self.clip_limit >= self.wind:
            self.clip_limit = int(round(0.08 * self.wind))

        # Reciprocal of hist.sum() == wind**2 + NUM_BINS, folded together with
        # the final "* 255" LUT scaling and the Q(FRAC_BITS) encoding of
        # hist_fp (see get_tile_lut) into a single register, precomputed once
        # per instance since it only depends on `wind`. get_approximate
        # quantizes this constant to RECIP_REG_BITS/RECIP_FRAC_BITS, the same
        # way BlackLevelCorrection quantizes its linearization factor.
        denom = self.wind**2 + NUM_BINS
        self.lut_scale, self.lut_scale_bin = get_approximate(
            255 / (denom * (1 << FRAC_BITS)), RECIP_REG_BITS, RECIP_FRAC_BITS
        )

    def validate_registers(self):
        """
        Validate LDCI register/config ranges, matching the bit widths the
        fixed-point datapath below assumes.
        """
        if self.wind < 1:
            raise ValueError(f"LDCI: wind must be >= 1, got {self.wind}")
        if self.wind**2 >= (1 << HIST_REG_BITS):
            raise ValueError(
                f"LDCI: wind={self.wind} overflows the U{HIST_REG_BITS} "
                f"histogram bin-count register (wind**2 must be < "
                f"{1 << HIST_REG_BITS})"
            )
        if self.clip_limit < 0:
            raise ValueError(
                f"LDCI: clip_limit must be >= 0, got {self.clip_limit}"
            )

    def pad_array(self, array, pads, mode="reflect"):
        """
        Pad an array with the given margins on left, right, top and bottom
        """
        if isinstance(pads, (list, tuple, np.ndarray)):
            if len(pads) == 2:
                pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (
                    array.ndim - 2
                )
            elif len(pads) == 4:
                pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (
                    array.ndim - 2
                )
            else:
                raise NotImplementedError
        return np.pad(array, pads, mode)

    def crop(self, array, crops):
        """
        Crop an array within the given margins
        """
        if isinstance(crops, (list, tuple, np.ndarray)):
            if len(crops) == 2:
                top_crop = bottom_crop = crops[0]
                left_crop = right_crop = crops[1]
            elif len(crops) == 4:
                top_crop, bottom_crop, left_crop, right_crop = crops
            else:
                raise NotImplementedError
        else:
            top_crop = bottom_crop = left_crop = right_crop = crops

        height, width = array.shape[:2]
        return array[
            top_crop : height - bottom_crop, left_crop : width - right_crop, ...
        ]

    def get_tile_lut(self, tiled_array):
        """
        Generate the 256-entry equalization LUT for one tile, in fixed point.

        hist.sum() after clip + redistribution is always the compile-time
        constant (wind**2 + NUM_BINS), so the float "pdf = hist / hist.sum()"
        divide is replaced by a single precomputed reciprocal-multiply
        (self.lut_scale), with the final "* 255" folded into the same
        constant.
        """
        hist, _ = np.histogram(tiled_array, bins=NUM_BINS, range=(0, 255))
        hist = hist.astype(np.int64)

        clipped_hist = np.clip(hist, 0, self.clip_limit)
        num_clipped = int((hist - clipped_hist).sum())

        # hist_fp[i] represents (clipped_hist[i] + num_clipped/NUM_BINS + 1)
        # in Q(*).FRAC_BITS fixed point. Because NUM_BINS == 2**FRAC_BITS,
        # "+ num_clipped/NUM_BINS" and "+ 1" are exact in this format --
        # num_clipped and (1 << FRAC_BITS) are simply placed directly into
        # the fractional bits, with no divider and no rounding error.
        hist_fp = (clipped_hist << FRAC_BITS) + num_clipped + (1 << FRAC_BITS)

        # self.lut_scale already folds in 255/(denom * 2**FRAC_BITS), so
        # weighted[i] == pdf[i] * 255 and cdf_scaled is cdf * 255 directly --
        # no further division or shift needed, matching the float reference.
        weighted = hist_fp * self.lut_scale
        cdf_scaled = np.cumsum(weighted)

        look_up_table = np.clip(np.round(cdf_scaled), 0, 255).astype(np.uint8)

        return look_up_table

    def interp_blocks(self, weights, block, first_block_lut, second_block_lut):
        """
        Interpolate two tile LUTs with Q0.10 alpha-blending weights.
        """
        first = weights * first_block_lut[block].astype(np.int32)
        second = (WEIGHT_UNITY - weights) * second_block_lut[block].astype(
            np.int32
        )
        return np.right_shift(first + second, WEIGHT_FRAC_BITS).astype(np.uint8)

    def interp_top_bottom_block(self, left_lut_weights, block, left_lut, current_lut):
        """
        Interpolate blocks on the top/bottom border with their left neighbour.
        """
        return self.interp_blocks(left_lut_weights, block, left_lut, current_lut)

    def interp_left_right_block(self, top_lut_weights, block, top_lut, current_lut):
        """
        Interpolate blocks on the left/right border with their top neighbour.
        """
        return self.interp_blocks(top_lut_weights, block, top_lut, current_lut)

    def interp_neighbor_block(
        self,
        left_lut_weights,
        top_lut_weights,
        block,
        tl_lut,
        top_lut,
        left_lut,
        current_lut,
    ):
        """
        Interpolate an interior block with its top-left, top and left
        neighbours (bilinear blend of the four surrounding tile LUTs).
        """
        interp_top_blocks = self.interp_blocks(left_lut_weights, block, tl_lut, top_lut)
        interp_current_blocks = self.interp_blocks(
            left_lut_weights, block, left_lut, current_lut
        )
        interp_final = np.right_shift(
            top_lut_weights * interp_top_blocks
            + (WEIGHT_UNITY - top_lut_weights) * interp_current_blocks,
            WEIGHT_FRAC_BITS,
        ).astype(np.uint8)
        return interp_final

    def is_corner_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Check if the current tile is at a corner of the tile grid.
        """
        return (
            (i_row == 0 and i_col == 0)
            or (i_row == 0 and i_col == x_tiles)
            or (i_row == y_tiles and i_col == 0)
            or (i_row == y_tiles and i_col == x_tiles)
        )

    def is_top_or_bottom_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Check if the current tile is on the top/bottom border (non-corner).
        """
        return (i_row == 0 or i_row == y_tiles) and not self.is_corner_block(
            x_tiles, y_tiles, i_col, i_row
        )

    def is_left_or_right_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Check if the current tile is on the left/right border (non-corner).
        """
        return (i_col == 0 or i_col == x_tiles) and not self.is_corner_block(
            x_tiles, y_tiles, i_col, i_row
        )

    def apply_ldci(self):
        """
        Apply fixed-point CLAHE contrast enhancement to the Y channel.
        """
        wind = self.wind
        in_yuv = self.img

        # LDCI only touches luminance; Cb/Cr pass through unchanged.
        yuv = in_yuv[:, :, 0].astype(np.uint8)
        img_height, img_width = yuv.shape

        out_ceh = np.empty(shape=(img_height, img_width, 3), dtype=np.uint8)

        vert_tiles = math.ceil(img_height / wind)
        horiz_tiles = math.ceil(img_width / wind)
        tile_height = wind
        tile_width = wind

        row_pads = tile_height * vert_tiles - img_height
        col_pads = tile_width * horiz_tiles - img_width
        pads = (
            row_pads // 2,
            row_pads - row_pads // 2,
            col_pads // 2,
            col_pads - col_pads // 2,
        )

        left_lut_weights = np.linspace(
            WEIGHT_UNITY, 0, tile_width, dtype=np.int32
        ).reshape((1, -1))
        top_lut_weights = np.linspace(
            WEIGHT_UNITY, 0, tile_height, dtype=np.int32
        ).reshape((-1, 1))

        luts = np.empty(shape=(vert_tiles, horiz_tiles, NUM_BINS), dtype=np.uint8)

        y_padded = self.pad_array(yuv, pads=pads)

        for rows in range(vert_tiles):
            for colm in range(horiz_tiles):
                start_row = rows * tile_height
                end_row = (rows + 1) * tile_height
                start_col = colm * tile_width
                end_col = (colm + 1) * tile_width
                y_tile = y_padded[start_row:end_row, start_col:end_col]
                luts[rows, colm] = self.get_tile_lut(y_tile)

        y_ceh = np.empty_like(y_padded)

        for i_row in range(vert_tiles + 1):
            for i_col in range(horiz_tiles + 1):
                start_row_index = i_row * tile_height - tile_height // 2
                end_row_index = min(start_row_index + tile_height, y_padded.shape[0])
                start_col_index = i_col * tile_width - tile_width // 2
                end_col_index = min(start_col_index + tile_width, y_padded.shape[1])
                start_row_index = max(start_row_index, 0)
                start_col_index = max(start_col_index, 0)

                y_block = y_padded[
                    start_row_index:end_row_index, start_col_index:end_col_index
                ].astype(np.uint8)

                if self.is_corner_block(horiz_tiles, vert_tiles, i_col, i_row):
                    lut_y_idx = 0 if i_row == 0 else vert_tiles - 1
                    lut_x_idx = 0 if i_col == 0 else horiz_tiles - 1
                    lut = luts[lut_y_idx, lut_x_idx]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = lut[y_block]

                elif self.is_top_or_bottom_block(horiz_tiles, vert_tiles, i_col, i_row):
                    lut_y_idx = 0 if i_row == 0 else vert_tiles - 1
                    left_lut = luts[lut_y_idx, i_col - 1]
                    current_lut = luts[lut_y_idx, i_col]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = self.interp_top_bottom_block(
                        left_lut_weights, y_block, left_lut, current_lut
                    )

                elif self.is_left_or_right_block(horiz_tiles, vert_tiles, i_col, i_row):
                    lut_x_idx = 0 if i_col == 0 else horiz_tiles - 1
                    top_lut = luts[i_row - 1, lut_x_idx]
                    current_lut = luts[i_row, lut_x_idx]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = self.interp_left_right_block(
                        top_lut_weights, y_block, top_lut, current_lut
                    )

                else:
                    tl_lut = luts[i_row - 1, i_col - 1]
                    top_lut = luts[i_row - 1, i_col]
                    left_lut = luts[i_row, i_col - 1]
                    current_lut = luts[i_row, i_col]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = self.interp_neighbor_block(
                        left_lut_weights,
                        top_lut_weights,
                        y_block,
                        tl_lut,
                        top_lut,
                        left_lut,
                        current_lut,
                    )

        y_out = self.crop(y_ceh, pads)

        out_ceh[:, :, 0] = y_out
        out_ceh[:, :, 1] = in_yuv[:, :, 1]
        out_ceh[:, :, 2] = in_yuv[:, :, 2]

        return out_ceh

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            self.save_out_obj.save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_ldci_",
                self.platform,
            )

    def execute(self):
        """
        Executing LDCI module according to user choice
        """
        print("LDCI = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            ldci_out = self.apply_ldci()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = ldci_out

        self.save()
        return self.img

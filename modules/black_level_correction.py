"""
File: black_level_correction.py
Description: Implements black level correction and image linearization based on config file params
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import get_approximate


class BlackLevelCorrection:
    """
    Black Level Correction
    """

    def __init__(self, img, platform, sensor_info, parm_blc, save_out_obj):
        self.img = img.copy()
        self.enable = parm_blc["is_enable"]
        self.is_save = parm_blc["is_save"]
        self.is_debug = parm_blc["is_debug"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.param_blc = parm_blc
        self.is_linearize = self.param_blc["is_linear"]
        self.save_out_obj = save_out_obj

    def apply_blc_parameters(self):
        """
        Apply BLC parameters provided in config file
        """

        # get config parm
        bayer = self.sensor_info["bayer_pattern"]
        bpp = self.sensor_info["bit_depth"]
        r_offset = self.param_blc["r_offset"]
        gb_offset = self.param_blc["gb_offset"]
        gr_offset = self.param_blc["gr_offset"]
        b_offset = self.param_blc["b_offset"]

        r_linfact = self.param_blc["linear_r"]
        gr_linfact = self.param_blc["linear_gr"]
        gb_linfact = self.param_blc["linear_gb"]
        b_linfact = self.param_blc["linear_b"]

        raw = np.float64(self.img)

        ## Get Approximates for Linearization - U16.14 precision
        # print("Approximated Linearization Factor")
        # r_linfact, r_linfact_bin = get_approximate(
        #     ((2**bpp) - 1) / (r_sat - r_offset), 16, 14
        # )
        # gr_linfact, gr_linfact_bin = get_approximate(
        #     ((2**bpp) - 1) / (gr_sat - gr_offset), 16, 14
        # )
        # gb_linfact, gb_linfact_bin = get_approximate(
        #     ((2**bpp) - 1) / (gb_sat - gb_offset), 16, 14
        # )
        # b_linfact, b_linfact_bin = get_approximate(
        #     ((2**bpp) - 1) / (b_sat - b_offset), 16, 14
        # )

        # if self.is_debug:
        #     print("   - BLC - R linearization factor (U16.14): " + r_linfact_bin)
        #     print("   - BLC - Gr linearization factor (U16.14): " + gr_linfact_bin)
        #     print("   - BLC - Gb linearization factor (U16.14): " + gb_linfact_bin)
        #     print("   - BLC - B linearization factor (U16.14): " + b_linfact_bin)

        if bayer == "rggb":
            # implementing this formula with condition
            # ((img - blc) / (sat_level-blc)) * bitRange

            raw[0::2, 0::2] = raw[0::2, 0::2] - r_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gr_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gb_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - b_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = raw[0::2, 0::2] * r_linfact
                raw[0::2, 1::2] = raw[0::2, 1::2] * gr_linfact
                raw[1::2, 0::2] = raw[1::2, 0::2] * gb_linfact
                raw[1::2, 1::2] = raw[1::2, 1::2] * b_linfact

        elif bayer == "bggr":
            raw[0::2, 0::2] = raw[0::2, 0::2] - b_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gb_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gr_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - r_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = raw[0::2, 0::2] * b_linfact
                raw[0::2, 1::2] = raw[0::2, 1::2] * gb_linfact
                raw[1::2, 0::2] = raw[1::2, 0::2] * gr_linfact
                raw[1::2, 1::2] = raw[1::2, 1::2] * r_linfact

        elif bayer == "grbg":
            raw[0::2, 0::2] = raw[0::2, 0::2] - gr_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - r_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - b_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gb_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = raw[0::2, 0::2] * gr_linfact
                raw[0::2, 1::2] = raw[0::2, 1::2] * r_linfact
                raw[1::2, 0::2] = raw[1::2, 0::2] * b_linfact
                raw[1::2, 1::2] = raw[1::2, 1::2] * gb_linfact

        elif bayer == "gbrg":
            raw[0::2, 0::2] = raw[0::2, 0::2] - gb_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - b_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - r_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gr_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = raw[0::2, 0::2] * gb_linfact
                raw[0::2, 1::2] = raw[0::2, 1::2] * b_linfact
                raw[1::2, 0::2] = raw[1::2, 0::2] * r_linfact
                raw[1::2, 1::2] = raw[1::2, 1::2] * gr_linfact

        raw = np.where(raw >= 0, np.floor(raw + 0.5), np.ceil(raw - 0.5))

        raw_blc = np.uint16(np.clip(raw, 0, (2**bpp) - 1))
        return raw_blc

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            self.save_out_obj.save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
            )

    def execute(self):
        """
        Black Level Correction
        """
        print("Black Level Correction = " + str(self.enable))

        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = blc_out
        self.save()
        return self.img

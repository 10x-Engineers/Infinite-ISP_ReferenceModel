"""
File: saturation_enhancement.py
Description: Saturation enhancement in YUV (HDR-ISP axis_sat).  Runs after 2D noise
             reduction, before RGB conversion.  Y passes through; each chroma sample
             is scaled about the chroma pedestal:

                 C' = clip( ((C - csc_offset) * sat_gain) >>> 8  + csc_offset )

             sat_gain   : the SAT_GAIN register, 12-bit unsigned Q4.8 (256 = 1.0) - the
                          config's saturation_gain truncated to it
             csc_offset : the SAT_CSC_OFFSET register, the chroma pedestal of the 8-bit
                          YUV that color_space_conversion makes and rgb_conversion
                          undoes: 2^(8-1) = 128
             >>> 8      : arithmetic shift - division by 256 rounded toward minus
                          infinity

             Bit-exact model of the RTL datapath at its register widths (BITS = 8):
               sub   = C - csc_offset              signed BITS+1      = 9 bit
               mult  = sub * sat_gain              signed 2*BITS+4    = 20 bit
               shift = mult >>> 8                  signed BITS+4      = 12 bit
               add   = shift + csc_offset          signed BITS+4      = 12 bit
               out   = 0 if add < 0, 2^BITS - 1 if add > 2^BITS - 1, else add
             The 12-bit add stage wraps for a gain above ~15.1: a fully saturated
             chroma sample would turn to 0.  Such a gain is refused (no realistic
             tuning comes near it), so every accepted configuration is clean.

             Only the "global" algorithm has a fixed-point / RTL design.  The
             hue-selective "hue_rolloff" of the algorithm-design model has none yet.

Code / Paper  Reference: HDR-ISP RTL axis/rtl/axis_sat.v (native hdr_isp_sat.v)
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import get_approximate

YUV_BITS = 8                      # color_space_conversion output depth (BITS)
GAIN_BITS, GAIN_FRAC = 12, 8      # SAT_GAIN register: unsigned Q4.8
CSC_OFFSET = 1 << (YUV_BITS - 1)  # chroma pedestal of the 8-bit YUV
SUB_BITS = YUV_BITS + 1
MULT_BITS = 2 * YUV_BITS + 4
ADD_BITS = YUV_BITS + 4


def wrap(value, bits):
    """Two's-complement wrap to a signed `bits`-wide register."""
    half = 1 << (bits - 1)
    return ((value + half) & ((1 << bits) - 1)) - half


def chroma_rtl(chroma, sat_gain, csc_offset=CSC_OFFSET):
    """One chroma plane through the RTL datapath (integer arrays in, uint8 out)."""
    sub = wrap(chroma.astype(np.int64) - csc_offset, SUB_BITS)
    mult = wrap(sub * sat_gain, MULT_BITS)
    shift = wrap(mult >> GAIN_FRAC, ADD_BITS)          # >> on int64 is arithmetic
    add = wrap(shift + csc_offset, ADD_BITS)
    return np.clip(add, 0, (1 << YUV_BITS) - 1).astype(np.uint8)


def max_clean_gain(csc_offset=CSC_OFFSET):
    """Largest SAT_GAIN register value for which no chroma sample wraps the add
    stage (the largest positive chroma offset is (2^BITS - 1) - csc_offset)."""
    top, limit = (1 << YUV_BITS) - 1 - csc_offset, (1 << (ADD_BITS - 1)) - 1
    reg = (1 << GAIN_BITS) - 1
    while reg and ((top * reg) >> GAIN_FRAC) + csc_offset > limit:
        reg -= 1
    return reg


class SaturationEnhancement:
    """
    Saturation Enhancement
    """

    def __init__(self, img, platform, sensor_info, parm_se, save_out_obj):
        self.img = img.copy()
        self.platform = platform
        self.sensor_info = sensor_info
        self.enable = parm_se["is_enable"]
        self.is_save = parm_se["is_save"]
        self.save_out_obj = save_out_obj
        self.algorithm = parm_se["algorithm"]
        self.saturation_gain = float(parm_se["saturation_gain"])
        self.sat_gain = None
        if self.enable:
            self.sat_gain = self.register_values()

    def register_values(self):
        """SAT_GAIN register from the config, refusing what the RTL cannot do."""
        if self.algorithm != "global":
            raise ValueError(
                f"saturation_enhancement.algorithm = '{self.algorithm}' has no "
                f"fixed-point / RTL design; set it to 'global' (hue_rolloff runs in "
                f"the Infinite-ISP algorithm-design model).")
        approx, _ = get_approximate(self.saturation_gain, GAIN_BITS, GAIN_FRAC)
        sat_gain = int(round(approx * (1 << GAIN_FRAC)))
        limit = max_clean_gain()
        if self.saturation_gain < 0 or sat_gain > limit:
            raise ValueError(
                f"saturation_enhancement.saturation_gain = {self.saturation_gain} is "
                f"outside the RTL's clean range: use 0 .. {limit / (1 << GAIN_FRAC):.4f} "
                f"(above it the 12-bit add stage wraps and saturated chroma turns to 0).")
        return sat_gain

    def apply_saturation(self):
        """Scale Cb / Cr about the pedestal; Y unchanged."""
        out = self.img.copy()
        out[:, :, 1] = chroma_rtl(self.img[:, :, 1], self.sat_gain)
        out[:, :, 2] = chroma_rtl(self.img[:, :, 2], self.sat_gain)
        return out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            self.save_out_obj.save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_saturation_enhancement_",
                self.platform,
            )

    def execute(self):
        """
        Execute Saturation Enhancement
        """
        print("Saturation Enhancement = " + str(self.enable))
        if self.enable:
            start = time.time()
            print(f"   - SE  - algorithm / gain = {self.algorithm} / {self.saturation_gain} "
                  f"(SAT_GAIN = {self.sat_gain}, Q4.8; SAT_CSC_OFFSET = {CSC_OFFSET})")
            self.img = self.apply_saturation()
            print(f"  Execution time: {time.time() - start:.3f}s")
        self.save()
        return self.img

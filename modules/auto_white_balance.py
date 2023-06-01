"""
File: auto_white_balance.py
Description: 3A - AWB Runs the AWB algorithm based on selection from config file
Code / Paper  Reference: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import get_approximate


class AutoWhiteBalance:
    """
    Auto White Balance Module
    """

    def __init__(self, raw, sensor_info, parm_awb):

        self.raw = raw

        self.sensor_info = sensor_info
        self.parm_awb = parm_awb
        self.enable = parm_awb["is_enable"]
        self.bit_depth = sensor_info["bit_depth"]
        self.is_debug = parm_awb["is_debug"]
        self.stats_window_offset = np.array(parm_awb["stats_window_offset"])
        self.underexposed_percentage = parm_awb["underexposed_pecentage"]
        self.overexposed_percentage = parm_awb["overexposed_pecentage"]
        self.flatten_raw = None
        self.bayer = self.sensor_info["bayer_pattern"]

    def apply_window_offset_crop(self):
        """
        Get AWB Stats window by cropping the offsets
        """
        offsets = np.ceil(self.stats_window_offset / 4) * 4
        top = int(offsets[0])
        bottom = None if offsets[1] == 0 else -int(offsets[1])
        left = int(offsets[2])
        right = None if offsets[3] == 0 else -int(offsets[3])
        return self.raw[top:bottom, left:right]

    def determine_white_balance_gain(self):
        """
        Determine white balance gains calculated using AWB Algorithms to Raw Image
        """

        self.raw = self.apply_window_offset_crop()

        max_pixel_value = 2**self.bit_depth
        appox_percenntage, _ = get_approximate(max_pixel_value / 100, 16, 8)
        # Removed overexposed and underexposed pixels for wb gain calculation
        overexposed_limit = (
            max_pixel_value - (self.overexposed_percentage) * appox_percenntage
        )
        underexposed_limit = (self.underexposed_percentage) * appox_percenntage

        if self.is_debug:
            print("   - AWB - Underexposed Pixel Limit = ", underexposed_limit)
            print("   - AWB - Overexposed Pixel Limit  = ", overexposed_limit)

        if self.bayer == "rggb":

            r_channel = self.raw[0::2, 0::2]
            gr_channel = self.raw[0::2, 1::2]
            gb_channel = self.raw[1::2, 0::2]
            b_channel = self.raw[1::2, 1::2]

        elif self.bayer == "bggr":
            b_channel = self.raw[0::2, 0::2]
            gb_channel = self.raw[0::2, 1::2]
            gr_channel = self.raw[1::2, 0::2]
            r_channel = self.raw[1::2, 1::2]

        elif self.bayer == "grbg":
            gr_channel = self.raw[0::2, 0::2]
            r_channel = self.raw[0::2, 1::2]
            b_channel = self.raw[1::2, 0::2]
            gb_channel = self.raw[1::2, 1::2]

        elif self.bayer == "gbrg":
            gb_channel = self.raw[0::2, 0::2]
            b_channel = self.raw[0::2, 1::2]
            r_channel = self.raw[1::2, 0::2]
            gr_channel = self.raw[1::2, 1::2]

        bayer_channels = np.dstack((r_channel, gr_channel, gb_channel, b_channel))
        # print(bayer_channels.shape)

        bad_pixels = np.sum(
            np.where(
                (bayer_channels < underexposed_limit)
                | (bayer_channels > overexposed_limit),
                1,
                0,
            ),
            axis=2,
        )
        self.flatten_raw = bayer_channels[bad_pixels == 0]
        # print(self.flatten_raw.shape)

        channels_mean = np.mean(self.flatten_raw, axis=0)
        # print(channels_mean)

        g_mean = np.mean(channels_mean[1:3], axis=0)

        rgain = np.nan_to_num(g_mean / channels_mean[0])
        bgain = np.nan_to_num(g_mean / channels_mean[3])

        # Check if r_gain and b_gain go out of bound
        rgain = 1 if rgain <= 0 else rgain
        bgain = 1 if bgain <= 0 else bgain

        rgain_approx, rgain_approx_bin = get_approximate(rgain, 16, 8)
        bgain_approx, bgain_approx_bin = get_approximate(bgain, 16, 8)

        if self.is_debug:
            print("   - AWB Actual Gains: ")
            print("   - AWB - RGain = ", rgain)
            print("   - AWB - Bgain = ", bgain)

            print("   - AWB Gains Aproximations: ")
            print("   - AWB - RGain = ", rgain_approx)
            print("   - AWB - Bgain = ", bgain_approx)

            print("   - AWB Gains Aproximation Binarys: ")
            print("   - AWB - RGain = ", rgain_approx_bin)
            print("   - AWB - Bgain = ", bgain_approx_bin)

        return rgain_approx, bgain_approx

    def execute(self):
        """
        Execute Auto White Balance Module
        """
        print("Auto White balancing = " + str(self.enable))

        # This module is enabled only when white balance 'enable' and 'auto' parameter both
        # are true.
        if self.enable is True:
            start = time.time()
            rgain, bgain = self.determine_white_balance_gain()
            print(f"  Execution time: {time.time() - start:.3f}s")
            return np.array([rgain, bgain])
        return None

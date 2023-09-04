"""
File: crop.py
Description: Crops the bayer image keeping cfa pattern intact
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import re
import time
from util.utils import save_output_array_yuv, save_output_array


class InvalidRegionCrop:
    """
    Parameters:
    ----------
    img: 3D array
    new_size:
    Return:
    ------
    cropped_img: cropped image of required size with same dtype
    as the input image.
    """

    def __init__(self, img, platform, sensor_info, parm_irc):
        self.img = img.copy()
        self.enable = parm_irc["is_enable"]
        self.platform = platform
        self.is_save = parm_irc["is_save"]
        self.is_debug = parm_irc["is_debug"]
        self.parm_irc = parm_irc
        self.bit_depth = sensor_info["bit_depth"]
        self.sensor_info = sensor_info
        self.is_valid = False

    def get_idx_for_rtl(self):
        """An offset is added to the indices to enbale exact array comparison."""

        dut = self.platform["dut"]
        offset = 0

        # offset is added according to the module under test
        if "dead_pixel_correction" in dut:
            offset += 2
        if "demosaic" in dut:
            offset += 2
        if "bayer_noise_reduction" in dut:
            offset += 6
        if "2d_noise_reduction" in dut:
            offset += 4

        # indices for RTL
        self.h_idx_rtl = self.h_strat_idx + offset
        self.w_idx_rtl = self.w_strat_idx

        # if the user-defined indices are within this defined range, generated TVs
        #  can be compared without any removal of rows or columns.
        # Input image size is assumed to be 2592x1536
        min_idx_h_gm, min_idx_w_gm = 14, 14
        max_idx_w_gm = 644
        if self.new_h == 1440:
            max_idx_h_gm = 68
        elif self.new_h == 1080:
            max_idx_h_gm = 428

        idx_valid_h = min_idx_h_gm <= self.h_strat_idx <= max_idx_h_gm
        idx_valid_w = min_idx_w_gm <= self.w_strat_idx <= max_idx_w_gm

        self.is_valid = idx_valid_h and idx_valid_w

    def crop_3d(self, img, strat_h, end_h, start_w, end_w):

        """This function performs cropping on a 3-channel image. The cropped
        region is determined by the given starting and ending indices for
        height and width"""

        if end_h > self.img.shape[0] or end_w > self.img.shape[1]:
            print("   - IRC - Invalid starting index.")
            cropped_img = img.copy()
        else:
            cropped_img = img[strat_h:end_h, start_w:end_w]

        return cropped_img

    def apply_cropping(self):
        """Crop Image"""
        # Get output size. It can be either 1920x1080 or 1920x1440
        self.crop_to_size = self.parm_irc["crop_to_size"]
        self.h_strat_idx = self.parm_irc["height_start_idx"]
        self.w_strat_idx = self.parm_irc["width_start_idx"]

        if self.crop_to_size == 1:
            self.new_h, self.new_w = 1080, 1920
        elif self.crop_to_size == 2:
            self.new_h, self.new_w = 1440, 1920
        else:
            print(
                f"   - IRC - Invalid key for output size {self.crop_to_size}. Select either 1 or 2."
            )
            return self.img

        # compute indices for cropping
        h_end_idx, w_end_idx = (
            self.h_strat_idx + self.new_h,
            self.w_strat_idx + self.new_w,
        )

        cropped_img = self.crop_3d(
            self.img, self.h_strat_idx, h_end_idx, self.w_strat_idx, w_end_idx
        )

        if self.is_debug:
            if cropped_img.shape == self.img.shape:
                print("   - IRC - Shape of cropped image = ", cropped_img.shape)

            else:
                if "dut" in self.platform.keys():
                    self.get_idx_for_rtl()
                else:
                    self.h_idx_rtl, self.w_idx_rtl = self.h_strat_idx, self.w_strat_idx

                crop_rows = self.img.shape[0] - self.new_h
                crop_cols = self.img.shape[1] - self.new_w
                print("   - IRC - Number of rows cropped = ", crop_rows)
                print("   - IRC - Number of columns cropped = ", crop_cols)
                print("   - IRC - Starting height index for RTL = ", self.h_idx_rtl)
                print("   - IRC - Starting width index for RTL = ", self.w_idx_rtl)
                print("   - IRC - Output width = ", self.new_w)
                print("   - IRC - Output height = ", self.new_h)
                if self.is_valid:
                    print(
                        "   - IRC - Indices for RTL can be used for TV comparison "
                        + "without removal of rows/border."
                    )

        return cropped_img

    def save(self):
        """
        Function to save module output
        """
        # update size of array in filename
        self.platform["in_file"] = re.sub(
            r"\d+x\d+",
            f"{self.img.shape[1]}x{self.img.shape[0]}",
            self.platform["in_file"],
        )
        if self.is_save:
            if self.platform["rgb_output"]:
                save_output_array(
                    self.platform["in_file"],
                    self.img,
                    "Out_invalid_region_crop_",
                    self.platform,
                    self.bit_depth,
                )
            else:
                save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_invalid_region_crop_",
                    self.platform,
                )

    def execute(self):
        """Execute cropping if enabled."""
        print("Invalid Region Crop = " + str(self.enable))
        if self.enable:
            start = time.time()
            cropped_img = self.apply_cropping()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = cropped_img
        self.save()
        return self.img


##########################################################################

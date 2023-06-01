"""
File: crop.py
Description: Crops the bayer image keeping cfa pattern intact
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import time


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

    def __init__(self, img, parm_irc):
        self.img = img
        self.enable = parm_irc["is_enable"]
        self.is_debug = parm_irc["is_debug"]
        self.parm_irc = parm_irc

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
                crop_rows = self.img.shape[0] - self.new_h
                crop_cols = self.img.shape[1] - self.new_w
                print("   - IRC - Number of rows cropped = ", crop_rows)
                print("   - IRC - Number of columns cropped = ", crop_cols)
                print("   - IRC - Starting index for height = ", self.h_strat_idx)
                print("   - IRC - Starting index for width = ", self.w_strat_idx)
                print("   - IRC - Output width = ", self.new_w)
                print("   - IRC - Output height = ", self.new_h)

        return cropped_img

    def execute(self):
        """Execute cropping if enabled."""
        print("Invalid Region Crop = " + str(self.enable))
        if self.enable:
            start = time.time()
            cropped_img = self.apply_cropping()
            print(f"  Execution time: {time.time() - start:.3f}s")
            return cropped_img
        return self.img


##########################################################################

"""
File: crop.py
Description: Crops the bayer image keeping cfa pattern intact
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import time
import re
import numpy as np


class Crop:
    """
    Parameters:
    ----------
    img: 2D array
    new_size: 2-tuple with required height and width.
    Return:
    ------
    cropped_img: Generated coloured image of required size with same dtype
    as the input image.
    """

    def __init__(self, img, platform, sensor_info, parm_cro, save_out_obj):
        self.img = img.copy()
        self.sensor_info = sensor_info
        self.platform = platform
        self.parm_cro = parm_cro
        self.enable = parm_cro["is_enable"]
        self.is_save = parm_cro["is_save"]
        self.save_out_obj = save_out_obj

    def crop(self, img, rows_to_crop=0, cols_to_crop=0):
        """
        Crop 2D array.
        Parameter:
        ---------
        img: image (2D array) to be cropped.
        rows_to_crop: Number of rows to crop. If it is an even integer,
                      equal number of rows are cropped from either side of the image.
                      Otherwise the image is cropped from the extreme right/bottom.
        cols_to_crop: Number of columns to crop. Works exactly as rows_to_crop.
        Output: cropped image
        """

        if rows_to_crop != 0 and rows_to_crop % 4 == 0:
            img = img[rows_to_crop // 2 : -rows_to_crop // 2, :]

        if cols_to_crop != 0 and cols_to_crop % 4 == 0:
            img = img[:, cols_to_crop // 2 : -cols_to_crop // 2]

        if rows_to_crop % 4 != 0 or cols_to_crop % 4 != 0:
            print(
                "   - Input/Output heights are not compatible. "
                "Bayer pattern will be disturbed if cropped!\n"
                f"   - Image size: {img.shape} "
            )
        return img

    def apply_cropping(self):
        """Crop Image"""

        # get crop parameters
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_cro["new_height"], self.parm_cro["new_width"])
        self.is_debug = self.parm_cro["is_debug"]

        if self.old_size == self.new_size:
            print("   - Output size is the same as input size.")
            return self.img

        if self.old_size[0] < self.new_size[0] or self.old_size[1] < self.new_size[1]:
            print(f"   - Invalid output size {self.new_size[0]}x{self.new_size[1]}")
            print(
                "   - Unable to crop! Make sure output size is smaller than input size."
            )
            return self.img

        crop_rows = self.old_size[0] - self.new_size[0]
        crop_cols = self.old_size[1] - self.new_size[1]
        cropped_img = np.empty(
            (self.new_size[0], self.new_size[1]), dtype=self.img.dtype
        )

        cropped_img = self.crop(self.img, crop_rows, crop_cols)

        if self.is_debug and cropped_img.shape != self.old_size:
            print("   - Crop - Number of rows cropped = ", crop_rows)
            print("   - Crop - Number of columns cropped = ", crop_cols)
            print("   - Crop - Shape of cropped image = ", cropped_img.shape)
        return cropped_img

    def save(self, filename_tag):
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
            self.save_out_obj.save_output_array(
                self.platform["in_file"],
                self.img,
                filename_tag,
                self.platform,
                self.sensor_info["bit_depth"],
            )

    def execute(self):
        """Execute cropping if enabled."""
        print("Crop = " + str(self.enable))

        # Save the input of crop module
        self.save("Inpipeline_crop_")

        # crop image if enabled
        if self.enable:
            start = time.time()
            cropped_img = self.apply_cropping()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = cropped_img

        # save the output of crop module
        self.save("Out_crop_")
        return self.img


##########################################################################

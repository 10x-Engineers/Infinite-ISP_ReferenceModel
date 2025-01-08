"""
File: sharpen.py
Description: Simple unsharp masking with frequency and strength control.
Code / Paper  Reference:
Author: 10xEngineers
"""

import time
import os
import numpy as np
from scipy import ndimage

class Sharpening:
    """
    Sharpening
    """

    def __init__(self, img, platform, sensor_info, parm_sha, save_out_obj):
        self.img = img.copy()
        self.enable = parm_sha["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sha = parm_sha
        self.is_save = parm_sha["is_save"]
        self.platform = platform
        self.save_out_obj = save_out_obj

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            self.save_out_obj.save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_sharpen_",
                self.platform,
            )

    def execute(self):
        """
        Applying sharpening to input image
        """

        print("Sharpen = " + str(self.enable))
        if self.enable is True:
            start = time.time()
            s_out = self.apply_sharpen()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img

    def apply_sharpen(self):
        return self.img 
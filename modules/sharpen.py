"""
File: sharpen.py
Description: Simple unsharp masking with frequency and strength control.
Code / Paper  Reference:
Author: xx-isp (ispinfinite@gmail.com)
"""

import time
import os
import numpy as np
from scipy import ndimage

from util.utils import create_coeff_file

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

    def gaussian_kernel(self, size_x, size_y=None, sigma_x=5, sigma_y=None):
        """
        Generate a Gaussian kernel for convolutions for Sharpening Algorithm
        """
        if size_y is None:
            size_y = size_x
        if sigma_y is None:
            sigma_y = sigma_x

        assert isinstance(size_x, int)
        assert isinstance(size_y, int)

        x_0 = size_x // 2
        y_0 = size_y // 2

        x_axis = np.arange(0, size_x, dtype=float)  # x_axis range [0:size_x]
        y_axis = np.arange(0, size_y, dtype=float)[:, np.newaxis]

        x_axis -= x_0
        y_axis -= y_0

        exp_part = x_axis**2 / (2 * sigma_x**2) + y_axis**2 / (2 * sigma_y**2)
        return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)

    def apply_sharpen(self):
        """Sharpens an image using the unsharp mask algorithm.

        Args:
            image: A numpy array of shape (height, width) representing the image to be sharpened.
            kernel_size: The size of the Gaussian kernel to use for blurring the image.
            sigma: The standard deviation of the Gaussian kernel.

        Returns:
            A numpy array of shape (height, width) representing the sharpened image.
        """

        sigma = self.parm_sha["sharpen_sigma"]
        kernel_size = 9

        kernel = self.gaussian_kernel(kernel_size, kernel_size, sigma, sigma)

        kernel = (kernel * (2**20)).astype(np.int32)

        folder_name = "coefficients/SHARP"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        create_coeff_file(kernel, "coefficients/SHARP/luma_kernel", 20)

        luma = (self.img[:, :, 0]).astype(np.int32)

        # Filter the luma component of the image with a Gaussian LPF
        # Smoothing magnitude can be controlled with the sharpen_sigma parameter
        correlation = ndimage.correlate(luma, kernel, mode='constant', cval=0.0, origin=0)

        smoothened = correlation >> 20

        # Sharpen the image with upsharp mask
        # Strength is tuneable with the sharpen_strength parameter
        sh_str = self.parm_sha["sharpen_strength"]
        print("   - Sharpen  - strength = ", sh_str)
        strength = int(sh_str * (2**10))

        edge = luma - smoothened

        sharpen = strength * edge
        sharpened = sharpen >> 10

        out_y = luma + sharpened
        self.img[:, :, 0] = np.uint8(np.clip(out_y, 0, 255))
        return self.img

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
            if self.platform["rgb_output"]:
                print("   - Invalid input for Sharpen: RGB image format.")
                self.parm_sha["is_enable"] = False
            else:    
                start = time.time()
                s_out = self.apply_sharpen()
                print(f"  Execution time: {time.time() - start:.3f}s")
                self.img = s_out

        self.save()
        return self.img

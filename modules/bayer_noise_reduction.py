"""
File: bayer_noise_reduction.py
Description: Noise reduction in bayer domain
Code / Paper  Reference:
https://www.researchgate.net/publication/261753644_Green_Channel_Guiding_Denoising_on_Bayer_Image
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import time
import warnings
import os
import numpy as np
from scipy import ndimage
from util.utils import create_coeff_file


class BayerNoiseReduction:
    """
    Noise Reduction in Bayer domain
    """

    def __init__(self, img, sensor_info, parm_bnr, platform):
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.save_lut = platform["save_lut"]

    def apply_bnr(self):
        """
        Apply bnr to the input image and return the output image
        """
        in_img = self.img.astype(np.int16)
        bayer_pattern = self.sensor_info["bayer_pattern"]
        width, height = self.sensor_info["width"], self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]

        # extract BNR parameters
        filt_size = self.parm_bnr["filter_window"]
        # s stands for spatial kernel parameters, r stands for range kernel parameters
        stddev_s_red, stddev_r_red = (
            self.parm_bnr["r_std_dev_s"],
            self.parm_bnr["r_std_dev_r"],
        )
        stddev_s_green, stddev_r_green = (
            self.parm_bnr["g_std_dev_s"],
            self.parm_bnr["g_std_dev_r"],
        )
        stddev_s_blue, stddev_r_blue = (
            self.parm_bnr["b_std_dev_s"],
            self.parm_bnr["b_std_dev_r"],
        )

        # assuming image is in 12-bit range, converting to [0 1] range
        # in_img = np.float32(in_img) / (2 ** bit_depth - 1)

        interp_g = np.zeros((height, width), dtype=np.int16)
        in_img_r = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.int16
        )
        in_img_b = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.int16
        )

        # convert bayer image into sub-images for filtering each colour ch
        in_img_raw = in_img.copy()
        if bayer_pattern == "rggb":
            in_img_r = in_img_raw[0:height:2, 0:width:2]
            in_img_b = in_img_raw[1:height:2, 1:width:2]
        elif bayer_pattern == "bggr":
            in_img_r = in_img_raw[1:height:2, 1:width:2]
            in_img_b = in_img_raw[0:height:2, 0:width:2]
        elif bayer_pattern == "grbg":
            in_img_r = in_img_raw[0:height:2, 1:width:2]
            in_img_b = in_img_raw[1:height:2, 0:width:2]
        elif bayer_pattern == "gbrg":
            in_img_r = in_img_raw[1:height:2, 0:width:2]
            in_img_b = in_img_raw[0:height:2, 1:width:2]

        # define the G interpolation kernel
        interp_kern_g_at_r = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.int32,
        )

        # interp_kern_g_at_r = interp_kern_g_at_r / np.sum(interp_kern_g_at_r)

        interp_kern_g_at_b = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.int32,
        )

        # interp_kern_g_at_b = interp_kern_g_at_b / np.sum(interp_kern_g_at_b)

        # convolve the kernel with image and mask the result based on given bayer pattern
        kern_filt_g_at_r = ndimage.convolve(
            np.int32(in_img), interp_kern_g_at_r, mode="reflect"
        )
        kern_filt_g_at_b = ndimage.convolve(
            np.int32(in_img), interp_kern_g_at_b, mode="reflect"
        )
        kern_filt_g_at_r = np.int32(kern_filt_g_at_r / 8)
        kern_filt_g_at_b = np.int32(kern_filt_g_at_b / 8)

        # clip any interpolation overshoots to [0 max] range
        kern_filt_g_at_r = np.clip(kern_filt_g_at_r, 0, 2**bit_depth - 1)
        kern_filt_g_at_b = np.clip(kern_filt_g_at_b, 0, 2**bit_depth - 1)

        interp_g = in_img.copy()
        interp_g_at_r = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.int16
        )
        interp_g_at_b = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.int16
        )

        if bayer_pattern == "rggb":
            # extract R and B location Green pixels to form interpG image
            interp_g[0:height:2, 0:width:2] = kern_filt_g_at_r[0:height:2, 0:width:2]
            interp_g[1:height:2, 1:width:2] = kern_filt_g_at_b[1:height:2, 1:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[0:height:2, 0:width:2]
            interp_g_at_b = kern_filt_g_at_b[1:height:2, 1:width:2]

        elif bayer_pattern == "bggr":
            # extract R and B location Green pixels to form interpG image
            interp_g[1:height:2, 1:width:2] = kern_filt_g_at_r[1:height:2, 1:width:2]
            interp_g[0:height:2, 0:width:2] = kern_filt_g_at_b[0:height:2, 0:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[1:height:2, 1:width:2]
            interp_g_at_b = kern_filt_g_at_b[0:height:2, 0:width:2]

        elif bayer_pattern == "grbg":
            # extract R and B location Green pixels to form interpG image
            interp_g[0:height:2, 1:width:2] = kern_filt_g_at_r[0:height:2, 1:width:2]
            interp_g[1:height:2, 0:width:2] = kern_filt_g_at_b[1:height:2, 0:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[0:height:2, 1:width:2]
            interp_g_at_b = kern_filt_g_at_b[1:height:2, 0:width:2]

        elif bayer_pattern == "gbrg":
            # extract R and B location Green pixels to form interpG image
            interp_g[1:height:2, 0:width:2] = kern_filt_g_at_r[1:height:2, 0:width:2]
            interp_g[0:height:2, 1:width:2] = kern_filt_g_at_b[0:height:2, 1:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[1:height:2, 0:width:2]
            interp_g_at_b = kern_filt_g_at_b[0:height:2, 1:width:2]

        # BNR window / filter size will be the same for full image and smaller for sub-image
        filt_size_g = int((filt_size + 1) / 2)
        filt_size_r = int((filt_size + 1) / 2)
        filt_size_b = int((filt_size + 1) / 2)

        # apply joint bilateral filter to the image with G channel as guidance signal
        out_img_r = self.fast_joint_bilateral_filter(
            in_img_r,
            interp_g_at_r,
            filt_size_r,
            stddev_s_red,
            filt_size_r,
            stddev_r_red,
            2,
            "R",
        )
        out_img_g = self.fast_joint_bilateral_filter(
            interp_g,
            interp_g,
            filt_size_g,
            stddev_s_green,
            filt_size_g,
            stddev_r_green,
            1,
            "G",
        )
        out_img_b = self.fast_joint_bilateral_filter(
            in_img_b,
            interp_g_at_b,
            filt_size_b,
            stddev_s_blue,
            filt_size_b,
            stddev_r_blue,
            2,
            "B",
        )

        # join the colour pixel images back into the bayer image
        bnr_out_img = np.zeros(in_img.shape)
        bnr_out_img = out_img_g.copy()

        if bayer_pattern == "rggb":
            bnr_out_img[0:height:2, 0:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[1:height:2, 1:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        elif bayer_pattern == "bggr":
            bnr_out_img[1:height:2, 1:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[0:height:2, 0:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        elif bayer_pattern == "grbg":
            bnr_out_img[0:height:2, 1:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[1:height:2, 0:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        elif bayer_pattern == "gbrg":
            bnr_out_img[1:height:2, 0:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[0:height:2, 1:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        return bnr_out_img

    def x_bf_make_color_curve(self, n_ind, max_diff, sigma_color, factor):
        """
        Generating Look-up-table based on color difference
        """
        # Create an empty 2D array to store the curve values
        curve = np.zeros((n_ind, 2), np.int16)

        # Iterate over the indices
        for i in range(n_ind):
            # Calculate the color difference based on the index
            diff = max_diff * (i + 1) // (n_ind + 1)

            # Assign the color difference value to the first column
            # of the curve array
            curve[i, 0] = diff

            # Calculate the corresponding value based on the color
            # difference using a formula
            curve[i, 1] = np.int16(
                factor * np.exp(-(diff**2) / (2 * sigma_color**2)) + 0.5
            )
        return curve

    def x_bf_make_color_kern(self, img_kern, color_curve1, max_weight):
        """
        Generating color kernel
        """
        # Create a new color curve array with one additional
        # row and the same number of columns as color_curve1
        color_curve = np.zeros(
            [color_curve1.shape[0] + 1, color_curve1.shape[1]], dtype=color_curve1.dtype
        )

        # Set the first row of color_curve as [0, max_weight]
        color_curve[0, :] = [0, max_weight]

        # Copy the values from color_curve1 to color_curve
        # starting from the second row
        color_curve[1:, :] = color_curve1

        # Create an empty kernel array with the same shape as
        # img_kern and the same dtype as color_curve
        kern = np.zeros(img_kern.shape, color_curve.dtype)

        # Find the indices for each value in img_kern in the
        # color_curve array
        indices = np.searchsorted(color_curve[:, 0], img_kern, side="right") - 1

        # Assign values to kern based on the indices using
        # conditional logic
        kern = np.where(indices >= 0, color_curve[indices, 1], kern)
        kern = np.where(img_kern == 0, max_weight, kern)
        kern = np.where(img_kern > color_curve[-1, 0], color_curve[-1, 1], kern)
        return kern

    def gauss_kern_raw(self, kern, std_dev, stride):
        """
        Applying Gaussian Filter
        """
        if kern % 2 == 0:
            warnings.warn("kernel size (kern) cannot be even, setting it as odd value")
            kern = kern + 1

        if kern <= 0:
            warnings.warn("kernel size (kern) cannot be <= zero, setting it as 3")
            kern = 3

        out_kern = np.zeros((kern, kern), dtype=np.float32)

        for i in range(0, kern):
            for j in range(0, kern):
                # stride is used to adjust the gaussian weights for neighbourhood
                # pixel that are 'stride' spaces apart in a bayer image
                out_kern[i, j] = np.exp(
                    -1
                    * (
                        (stride * (i - ((kern - 1) / 2))) ** 2
                        + (stride * (j - ((kern - 1) / 2))) ** 2
                    )
                    / (2 * (std_dev**2))
                )

        return out_kern

    def fast_joint_bilateral_filter(
        self,
        in_img,
        guide_img,
        spatial_kern,
        stddev_s,
        range_kern,
        stddev_r,
        stride,
        channel_name,
    ):
        """
        Applying Joint Bilateral Filter
        """

        # determining color curve
        bpp = self.sensor_info["bit_depth"]
        curve = self.x_bf_make_color_curve(
            9, 2 * stddev_r * (2**bpp - 1), stddev_r * (2**bpp - 1), 255
        )  # 255 is the scaling factor

        if self.save_lut:
            folder_name = "coefficients/BNR"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            create_coeff_file(
                np.flip(curve[:, 0]),
                "coefficients/BNR/color_curve_diff_" + str(stddev_r) + channel_name,
                bpp,
            )
            create_coeff_file(
                np.flip(curve[:, 1]),
                "coefficients/BNR/color_curve_weights_" + str(stddev_r) + channel_name,
                8,
            )

        # check if filter window sizes spatial_kern and range_kern greater than zero and are odd
        if spatial_kern <= 0:
            spatial_kern = 3
            warnings.warn(
                "spatial kernel size (spatial_kern) cannot be <= zero, setting it as 3"
            )
        elif spatial_kern % 2 == 0:
            warnings.warn(
                "range kernel size (spatial_kern) cannot be even, assigning it an odd value"
            )
            spatial_kern = spatial_kern + 1

        if range_kern <= 0:
            range_kern = 3
            warnings.warn(
                "range kernel size (range_kern) cannot be <= zero, setting it as 3"
            )
        elif range_kern % 2 == 0:
            warnings.warn(
                "range kernel size (range_kern) cannot be even, assigning it an odd value"
            )
            range_kern = range_kern + 1

        # check if range_kern > spatial_kern
        if range_kern > spatial_kern:
            warnings.warn(
                "range kernel size (range_kern) cannot be more than..."
                "spatial kernel size (spatial_kern)"
            )
            range_kern = spatial_kern

        # spawn a NxN gaussian kernel
        s_kern = self.gauss_kern_raw(spatial_kern, stddev_s, stride)
        s_kern = np.uint8(
            255 * s_kern + 0.5
        )  # scaling by a factor and adding an offset
        if self.save_lut:
            create_coeff_file(
                s_kern,
                "coefficients/BNR/s_kern"
                + str(spatial_kern)
                + "x"
                + str(spatial_kern)
                + "_"
                + str(stddev_s)
                + channel_name,
                8,
            )

        # pad the image with half arm length of the kernel;
        # padType='constant' => pad value = 0; 'reflect' is more suitable
        pad_len = int((spatial_kern - 1) / 2)
        in_img_ext = np.pad(in_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect")
        guide_img_ext = np.pad(
            guide_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect"
        )

        # filt_out = np.zeros(in_img.shape, dtype=np.float32)
        norm_fact = np.zeros(in_img.shape, dtype=np.int32)
        sum_filt_out = np.zeros(in_img.shape, dtype=np.int32)

        for i in range(spatial_kern):
            for j in range(spatial_kern):
                # Creating shifted arrays for processing each pixel in the window
                in_img_ext_array = in_img_ext[
                    i : i + in_img.shape[0], j : j + in_img.shape[1], ...
                ]
                guide_img_ext_array = guide_img_ext[
                    i : i + in_img.shape[0], j : j + in_img.shape[1], ...
                ]

                diff = np.abs(guide_img - guide_img_ext_array)

                color_kern = self.x_bf_make_color_kern(diff, curve, 255)
                # Adding normalization factor for each pixel needed to average out the
                # final result
                norm_fact += s_kern[i, j] * np.int32(color_kern)
                # Summing up the final result
                sum_filt_out += (
                    s_kern[i, j] * np.int32(color_kern) * np.int32(in_img_ext_array)
                )

        filt_out = np.int16(np.int32(sum_filt_out) / np.int32(norm_fact))
        return filt_out

    def execute(self):
        """
        Appling BNR to input RAW image and returns the output image
        """
        print("Bayer Noise Reduction = " + str(self.enable))

        if self.enable is False:
            # return the same image as input image
            return self.img
        else:
            start = time.time()
            bnr_out = self.apply_bnr()
            print(f"  Execution time: {time.time() - start:.3f}s")
            return bnr_out

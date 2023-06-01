"""
File: noise_reduction_2d.py
Description:
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf
Implementation inspired from:
Fast Open ISP Author: Qiu Jueqin (qiujueqin@gmail.com)
& scikit-image (nl_means_denoising)
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import os
import numpy as np
from tqdm import tqdm
from util.utils import create_coeff_file


class NoiseReduction2d:
    """
    2D Noise Reduction
    """

    def __init__(self, img, sensor_info, parm_2dnr, platform):
        self.img = img
        self.enable = parm_2dnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_2dnr = parm_2dnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.save_lut = platform["save_lut"]


    def make_weighted_curve(self, n_ind):
        """
        Creating weighting LUT
        """
        h_par = self.parm_2dnr["wts"]
        curve = np.zeros((n_ind, 2), np.int32)
        diff = np.linspace(0, 255, n_ind)
        # Considering maximum weight to be 31 (5 bit)
        wts = (np.exp(-(diff**2) / h_par ** 2) * 31).astype(np.int32)
        curve[:, 0] = diff
        curve[:, 1] = wts
        return curve

    def apply_weights(self, img_kern, color_curve):
        """
        Applying weights to kernel
        """
        kern = np.zeros(img_kern.shape, color_curve.dtype)
        indices = np.searchsorted(color_curve[:, 0], img_kern, side="right") - 1
        kern = np.where(indices >= 0, color_curve[indices, 1], kern)
        kern = np.where(img_kern == 0, color_curve[0, 1], kern)
        kern = np.where(img_kern > color_curve[-1, 0], color_curve[-1, 1], kern)
        return kern

    def apply_nlm(self):
        """
        Applying Non-local Means Filter
        """
        # Input YUV image
        in_image = self.img

        # Search window and patch sizes
        window_size = self.parm_2dnr["window_size"]

        # Extracting Y channel to apply the 2DNR module
        input_image = in_image[:, :, 0]

        if in_image.dtype == "float32":
            input_image = np.round(255 * input_image).astype(np.uint8)

        # Declaring empty array for output image after denoising
        denoised_out = np.empty(in_image.shape, dtype=np.uint8)

        # Padding the input_image
        pads = window_size // 2
        wtspadded_y_in = np.pad(input_image, pads, mode="reflect")

        # Declaration of denoised Y channel and weights
        denoised_y_channel = np.int32(np.zeros(input_image.shape))
        final_weights = np.int32(np.zeros(input_image.shape))

        # Generating LUT weighs based on euclidean distance between intensities
        # Assigning weights to similar pixels in descending order (most similar
        # will have the largest weight_for_each_shifted_array)
        # weights_lut = self.get_weights()
        no_of_levels = 32  # it can be 256 to utilize the full difference range
        curve = self.make_weighted_curve(no_of_levels)

        if self.save_lut:
            # Writing weight coefficients to the file
            folder_name = "coefficients/2DNR"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            create_coeff_file(
                np.flip(curve[:, 0]), "coefficients/2DNR/color_curve_diff", 8
            )
            create_coeff_file(
                np.flip(curve[:, 1]), "coefficients/2DNR/color_curve_weights", 5
            )

        for i in tqdm(
            range(window_size), disable=self.is_progress, leave=self.is_leave
        ):
            for j in range(window_size):
                # Creating arrays starting from pixels according to the search window --
                # There will N = search_window*search_window stacked arrays of input
                # image size
                array_for_each_pixel_in_sw = np.int32(
                    wtspadded_y_in[
                        i : i + input_image.shape[0], j : j + input_image.shape[1], ...
                    ]
                )

                # Finding euclidean distance between pixels based on their
                # intensities and applying weights accordingly
                distance = np.abs(input_image - array_for_each_pixel_in_sw)

                # Assigning weights to the pixels based on their distance (most similar
                # will have the largest weight)
                weight_for_each_shifted_array = self.apply_weights(distance, curve)

                # Adding up all the weighted similar pixels
                denoised_y_channel += (
                    array_for_each_pixel_in_sw * weight_for_each_shifted_array
                )

                # Adding up all the weights for final mean values at each pixel location
                final_weights += weight_for_each_shifted_array


        # Averaging out all the pixel
        denoised_y_channel = np.float32(denoised_y_channel) / final_weights
        denoised_y_channel = np.uint8(
            np.where(
                denoised_y_channel >= 0,
                np.floor(denoised_y_channel + 0.5),
                np.ceil(denoised_y_channel - 0.5),
            )
        )

        if in_image.dtype == "float32":
            denoised_y_channel = np.float32((denoised_y_channel) / 255.0)
            denoised_out = denoised_out.astype("float32")

        # Reconstructing the final output
        denoised_out[:, :, 0] = denoised_y_channel
        denoised_out[:, :, 1] = in_image[:, :, 1]
        denoised_out[:, :, 2] = in_image[:, :, 2]

        return denoised_out

    def execute(self):
        """
        Executing 2D noise reduction module
        """
        print("Noise Reduction 2d = " + str(self.enable))

        if self.enable is False:
            return self.img
        else:
            start = time.time()
            s_out = self.apply_nlm()
            print(f"  Execution time: {time.time() - start:.3f}s")
            return s_out

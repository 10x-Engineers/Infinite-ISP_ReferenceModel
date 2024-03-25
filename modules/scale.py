"""
File: scale.py
Description: Implements both hardware friendly and non hardware freindly scaling
Code / Paper  Reference:
https://patentimages.storage.googleapis.com/f9/11/65/a2b66f52c6dbd4/US8538199.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import re
import numpy as np
from util.utils import crop, stride_convolve2d, save_output_array, save_output_array_yuv


class Scale:
    """Scale color image to given size."""

    def __init__(self, img, platform, sensor_info, parm_sca, conv_std):
        self.img = img.copy()
        self.enable = parm_sca["is_enable"]
        self.is_save = parm_sca["is_save"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.conv_std = conv_std
        self.parm_sca = parm_sca
        self.get_scaling_params()

    def apply_scaling(self):
        """Execute scaling."""

        # check if scaling is needed
        if self.old_size == self.new_size:
            if self.is_debug:
                print("   - Scale - Output size is the same as input size.")
            return self.img

        scaled_img = np.empty((self.new_size[0], self.new_size[1], 3), dtype="uint8")

        debug_flag = self.parm_sca["is_debug"]

        # Loop over each channel to resize the image
        for i in range(3):

            ch_arr = self.img[:, :, i]
            scale_2d = Scale2D(ch_arr, self.sensor_info, self.parm_sca)
            scaled_ch = scale_2d.execute()

            # If input size is invalid, the Scale2D class returns the image as it is.
            if scaled_ch.shape == self.old_size:
                return self.img
            else:
                scaled_img[:, :, i] = scaled_ch

            # Because each channel is scaled in the same way, the is_deug flag is turned
            # off after the first channel has been scaled.
            self.parm_sca["is_debug"] = False

        self.parm_sca["is_debug"] = debug_flag
        return scaled_img

    def get_scaling_params(self):
        """Save parameters as instance attributes."""
        self.is_debug = self.parm_sca["is_debug"]
        self.old_size = (self.img.shape[0], self.img.shape[1])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])

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
                    "Out_scale_",
                    self.platform,
                    self.sensor_info["bit_depth"],
                    self.sensor_info["bayer_pattern"]
                )
            else:
                save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_",
                    self.platform,
                    self.conv_std
                )

    def execute(self):
        """Execute scaling if enabled."""
        print("Scale = " + str(self.enable))

        if self.enable:
            start = time.time()
            scaled_img = self.apply_scaling()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = scaled_img
        self.save()
        return self.img


################################################################################
class Scale2D:
    """Scale 2D image to given size."""

    def __init__(self, single_channel, sensor_info, parm_sca):
        self.single_channel = single_channel
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
        self.get_scaling_params()

    def fast_nearest_neighbor(self, new_size):

        """Down scale by an integer factor using NN method using convolution."""

        old_height, old_width = (
            self.single_channel.shape[0],
            self.single_channel.shape[1],
        )
        new_height, new_width = new_size[0], new_size[1]

        # As new_size is less than old_size, scale factor is defined s.t it is >1 for downscaling
        scale_height, scale_width = old_height / new_height, old_width / new_width

        kernel = np.zeros((int(scale_height), int(scale_width)))
        kernel[0, 0] = 1

        scaled_img = stride_convolve2d(self.single_channel, kernel)
        return scaled_img

    def scale_nearest_neighbor(self, new_size):

        """
        Upscale/Downscale 2D array by any scale factor using Nearest Neighbor (NN) algorithm.
        """

        old_height, old_width = (
            self.single_channel.shape[0],
            self.single_channel.shape[1],
        )
        new_height, new_width = new_size[0], new_size[1]
        scale_height, scale_width = new_height / old_height, new_width / old_width

        scaled_img = np.zeros((new_height, new_width), self.single_channel.dtype)

        for row in range(new_height):
            for col in range(new_width):
                row_nearest = int(np.floor(row / scale_height))
                col_nearest = int(np.floor(col / scale_width))
                scaled_img[row, col] = self.single_channel[row_nearest, col_nearest]
        return scaled_img

    def hardware_dep_scaling(self):
        """Set algorithm workflow."""
        # check and set the flow of the algorithm
        scale_info = self.validate_input_output()

        # apply scaling according to the flow
        return self.apply_algo(scale_info)

    def apply_algo(self, scale_info):

        """
        Scale 2D array using hardware friendly approach comprising of 2 steps:
           1. Downscale with int factor
           2. Crop
        """

        # check if input size is valid
        if scale_info == [[None, None], [None, None]]:
            print(
                "   - Invalid input size. It must be one of the following:\n"
                "   - 1920x1080\n"
                "   - 1920x1440"
            )
            return self.single_channel

        # check if output size is valid
        if scale_info == [[1, 0], [1, 0]]:
            print("   - Invalid output size.")
            return self.single_channel
        else:
            # step 1: Downscale by int fcator
            if scale_info[0][0] > 1 or scale_info[1][0] > 1:

                self.single_channel = self.fast_nearest_neighbor(
                    (
                        self.old_size[0] // scale_info[0][0],
                        self.old_size[1] // scale_info[1][0],
                    )
                )

                if self.is_debug:
                    print(
                        "   - Scale - Shape after downscaling by integer factor "
                        + f"({scale_info[0][0]}, {scale_info[1][0]}):",
                        self.single_channel.shape,
                    )

            # step 2: crop
            if scale_info[0][1] > 0 or scale_info[1][1] > 0:
                self.single_channel = crop(
                    self.single_channel, scale_info[0][1], scale_info[1][1]
                )

                if self.is_debug:
                    print(
                        "   - Scale - Shape after cropping "
                        + f"({scale_info[0][1]}, {scale_info[1][1]}): ",
                        self.single_channel.shape,
                    )
            return self.single_channel

    def validate_input_output(self):
        """Chcek if the size of the input image is valid according to the set workflow."""
        valid_size = [(1080, 1920), (1440, 1920)]
        sizes = [OUT1080x1920, OUT1440x1920]

        # Check if input size is valid
        if self.old_size not in valid_size:
            scale_info = [[None, None], [None, None]]
            return scale_info

        idx = valid_size.index(self.old_size)
        size_obj = sizes[idx](self.new_size)
        scale_info = size_obj.execute()
        return scale_info

    def execute(self):
        """Execute scaling."""
        self.single_channel = self.hardware_dep_scaling()

        if self.is_debug:
            print(
                "   - Scale - Shape of scaled image for a single channel = ",
                self.single_channel.shape,
            )
        return self.single_channel

    def get_scaling_params(self):
        """Save parameters as instance attributes."""
        self.is_debug = self.parm_sca["is_debug"]
        self.old_size = (self.single_channel.shape[0], self.single_channel.shape[1])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])


#############################################################################
# The structure and working of the following two classes is exactly the same
class OUT1080x1920:
    """
    The scale module can only be used for specific input and output sizes.
    This class checks if the given output size can be achieved by implemented approach by
    creating a nested list (SCALE_LIST below) with corresponding constants used to execute
    this scaling approach comprising of the following 2 steps:
    1. Downscale with int factor
    2. Crop

    Instance Attributes:
    -------------------
    SCALE_LIST:  a nested list with two sublists containing constants used to
    scale height (index 0) and width (index 1) to the given NEW_SIZE using the two
    steps above.

    The elements in each list correspond to the following constants:
    1. Scale factor [int]: (default 1) scale factor for downscaling.
    2. Crop value [int] : (defaut 0) number of rows or columns to be cropped.
    """

    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (480, 640):
            self.scale_list = [[2, 60], [3, 0]]

        elif self.new_size == (360, 640):
            self.scale_list = [[3, 0], [3, 0]]

        else:
            self.scale_list = [[1, 0], [1, 0]]

    def execute(self):
        """Get crop/scale factors to the corresponding input size"""
        return self.scale_list


#############################################################################
class OUT1440x1920:
    """
    This class works exactly as OUT1080x1920.
    """

    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (720, 960):
            self.scale_list = [[2, 0], [2, 0]]

        elif self.new_size == (480, 640):
            self.scale_list = [[3, 0], [3, 0]]

        else:
            self.scale_list = [[1, 0], [1, 0]]

    def execute(self):
        """Get crop/scale factors to the corresponding input size"""
        return self.scale_list


#############################################################################

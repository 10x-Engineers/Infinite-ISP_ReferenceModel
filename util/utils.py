"""
File: utils.py
Description: Common helper functions for all algorithms
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
from datetime import datetime
import random
import warnings
from pathlib import Path
import numpy as np
import yaml
from fxpmath import Fxp
from scipy.signal import correlate2d
import matplotlib.pyplot as plt


def introduce_defect(img, total_defective_pixels, padding):
    """
    This function randomly replaces pixels values with extremely high or low
    pixel values to create dead pixels (Dps).
    Note that the defective pixel values are never introduced on the periphery
    of the image to ensure that there are no adjacent DPs.
    Parameters
    ----------
    img: 2D ndarray
    total_defective_pixels: number of Dps to introduce in img.
    padding: bool value (set to True to add padding)
    Returns
    -------
    defective image: image/padded img containing specified (by TOTAL_DEFECTIVE_PIXELS)
    number of dead pixels.
    orig_val: ndarray of size same as img/padded img containing original pixel values
    in place of introduced DPs and zero elsewhere.
    """

    if padding:
        padded_img = np.pad(img, ((2, 2), (2, 2)), "reflect")
    else:
        padded_img = img.copy()

    orig_val = np.zeros((padded_img.shape[0], padded_img.shape[1]))

    while total_defective_pixels:
        defect = [
            random.randrange(1, 15),
            random.randrange(4081, 4095),
        ]  # stuck low int b/w 1 and 15, stuck high float b/w 4081 and 4095
        defect_val = defect[random.randint(0, 1)]
        random_row, random_col = random.randint(2, img.shape[0] - 3), random.randint(
            2, img.shape[1] - 3
        )
        left, right = (
            orig_val[random_row, random_col - 2],
            orig_val[random_row, random_col + 2],
        )
        top, bottom = (
            orig_val[random_row - 2, random_col],
            orig_val[random_row + 2, random_col],
        )
        neighbours = [left, right, top, bottom]

        if (
            not any(neighbours) and orig_val[random_row, random_col] == 0
        ):  # if all neighbouring values in orig_val are 0 and the pixel itself is not defective
            orig_val[random_row, random_col] = padded_img[random_row, random_col]
            padded_img[random_row, random_col] = defect_val
            total_defective_pixels -= 1

    return padded_img, orig_val


def gauss_kern_raw(size, std_dev, stride):
    """
    This function takes in size, standard deviation and spatial stride required for adjacet
    weights to output a gaussian kernel of size NxN
    Parameters
    ----------
    size:   size of gaussian kernel, odd
    std_dev: standard deviation of the gaussian kernel
    stride: spatial stride between to be considered for adjacent gaussian weights
    Returns
    -------
    outKern: an output gaussian kernel of size NxN
    """

    if size % 2 == 0:
        warnings.warn("kernel size (N) cannot be even, setting it as odd value")
        size = size + 1

    if size <= 0:
        warnings.warn("kernel size (N) cannot be <= zero, setting it as 3")
        size = 3

    out_kern = np.zeros((size, size), dtype=np.float32)

    for i in range(0, size):
        for j in range(0, size):
            out_kern[i, j] = np.exp(
                -1
                * (
                    (stride * (i - ((size - 1) / 2))) ** 2
                    + (stride * (j - ((size - 1) / 2))) ** 2
                )
                / (2 * (std_dev**2))
            )

    sum_kern = np.sum(out_kern)
    out_kern[0:size:1, 0:size:1] = out_kern[0:size:1, 0:size:1] / sum_kern

    return out_kern


def crop(img, rows_to_crop=0, cols_to_crop=0):
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

    if rows_to_crop:
        if rows_to_crop % 2 == 0:
            img = img[rows_to_crop // 2 : -rows_to_crop // 2, :]
        else:
            img = img[0:-1, :]
    if cols_to_crop:
        if cols_to_crop % 2 == 0:
            img = img[:, cols_to_crop // 2 : -cols_to_crop // 2]
        else:
            img = img[:, 0:-1]
    return img


def stride_convolve2d(matrix, kernel):
    """2D convolution"""
    return correlate2d(matrix, kernel, mode="valid")[
        :: kernel.shape[0], :: kernel.shape[1]
    ]


def get_approximate(decimal, register_bits, frac_precision_bits):
    """
    Returns Fixed Float Approximation of a decimal
    -- Fxp function from fxpmath library of python is used that takes following inputs:
    decimal : number to be converted to fixed point number
    signed_value : falg to indicate if we need signed fixed point number
    register_bits : bit depth of fixed point number
    frac_precision_bits : bit depth of fractional part of a fixed point number
    """
    fixed_float = Fxp(decimal, False, register_bits, frac_precision_bits)
    return fixed_float(), fixed_float.bin()


def approx_sqrt(number, num_iterations=5):
    """
    Get Approximate Square_root of a Number
    """
    # Initial estimate
    sqrt = np.uint64(number / 2)
    # Only five iterations are performed to get an approximate square-root
    for _ in range(num_iterations):
        n_by_sqrt = np.uint64(np.uint64(number) / sqrt)
        sqrt = np.uint64((sqrt + n_by_sqrt) / 2)
        # print(sqrt)
    # Integer is returned to reduce algorithm complexity
    return sqrt.astype(np.uint64)


def display_ae_statistics(ae_feedback, awb_gains):
    """
    Print AE Stats for current frame
    """
    # Logs for AWB
    if awb_gains is None:
        print("   - 3A Stats    - AWB is Disable")
    else:
        print("   - 3A Stats    - AWB Rgain = ", awb_gains[0])
        print("   - 3A Stats    - AWB Bgain = ", awb_gains[1])

    # Logs for AE
    if ae_feedback is None:
        print("   - 3A Stats    - AE is Disable")
    else:
        if ae_feedback < 0:
            print("   - 3A Stats    - AE Feedback = Underexposed")
        elif ae_feedback > 0:
            print("   - 3A Stats    - AE Feedback = Overexposed")
        else:
            print("   - 3A Stats    - AE Feedback = Correct Exposure")


def reconstruct_yuv_from_422_custom(yuv_422_custom, width, height):
    """
    Reconstruct a YUV from YUV 422 format
    """
    # Create an empty 3D YUV image (height, width, channels)
    yuv_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Rearrange the flattened 4:2:2 YUV data back to 3D YUV format
    yuv_img[:, 0::2, 0] = yuv_422_custom[0::4].reshape(height, -1)
    yuv_img[:, 0::2, 1] = yuv_422_custom[1::4].reshape(height, -1)
    yuv_img[:, 1::2, 0] = yuv_422_custom[2::4].reshape(height, -1)
    yuv_img[:, 0::2, 2] = yuv_422_custom[3::4].reshape(height, -1)

    # Replicate the U and V (chroma) channels to the odd columns
    yuv_img[:, 1::2, 1] = yuv_img[:, 0::2, 1]
    yuv_img[:, 1::2, 2] = yuv_img[:, 0::2, 2]

    return yuv_img


def reconstruct_yuv_from_444_custom(yuv_444_custom, width, height):
    """
    Reconstruct a YUV from YUV 444 format
    """
    # Create an empty 3D YUV image (height, width, channels)
    yuv_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Rearrange the flattened 4:2:2 YUV data back to 3D YUV format
    yuv_img[:, 0::1, 0] = yuv_444_custom[0::3].reshape(height, -1)
    yuv_img[:, 0::1, 1] = yuv_444_custom[1::3].reshape(height, -1)
    yuv_img[:, 0::1, 2] = yuv_444_custom[2::3].reshape(height, -1)

    return yuv_img


def get_image_from_yuv_format_conversion(yuv_img, height, width, yuv_custom_format):
    """
    Convert YUV image into RGB based on its format & Conversion Standard
    """

    # Reconstruct the 3D YUV image from the custom given format YUV data
    if yuv_custom_format == "422":
        yuv_img = reconstruct_yuv_from_422_custom(yuv_img, width, height)
    else:
        yuv_img = reconstruct_yuv_from_444_custom(yuv_img, width, height)

    return yuv_img


def create_coeff_file(numbers, filename, weight_bits):
    """
    Creating file for coefficients
    """
    extension = ".txt"
    coeff = "{"
    # write the array to the file
    if len(numbers.shape) == 2:
        for n_ind in range(0, numbers.shape[0]):
            for m_ind in range(0, numbers.shape[1]):
                # convert the number to its binary representation
                coeff = (
                    coeff
                    + "{"
                    + str(weight_bits)
                    + "'d"
                    + str(int(numbers[n_ind, m_ind]))
                    + "},"
                )
            coeff = coeff + "\n"
        coeff = coeff[:-2]
        coeff = coeff + "};"
    else:
        for m_ind in range(0, numbers.shape[0]):
            # convert the number to its binary representation
            coeff = (
                coeff + "{" + str(weight_bits) + "'d" + str(int(numbers[m_ind])) + "},"
            )
        coeff = coeff[:-1]
        coeff = coeff + "};"
    with open("{}{}".format(filename, extension), "w") as fil:
        fil.write(coeff)
    return coeff


# utilities to save the config_automate exactly as config.yml
class CustomDumper(yaml.Dumper):

    """This class is a custom YAML dumper that overrides the default behavior
    of the increase_indent and write_line_break methods. It ensures that indentations
    and line breaks are inserted correctly in the output YAML file."""

    def increase_indent(self, flow=False, indentless=False):
        """For indentation"""
        return super(CustomDumper, self).increase_indent(flow, False)

    def write_line_break(self, data=None):
        """For line break"""
        super().write_line_break(data)
        if len(self.indents) == 1:
            self.stream.write("\n")


def represent_list(self, data):
    """This function ensures that the lookup table are stored on flow style
    to keep the saved yaml file readable."""
    return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class SaveOutput:
    """Class to save module or pipeline output"""

    def __init__(self):
        self.pipleine_outpath = Path("./out_frames")
        self.module_outpath = Path("./out_frames/module_output")
        self.config_outpath = Path("./out_frames")
        self.create_out_dirs()

    def create_out_dirs(self):
        """Create directories if it does not exist."""
        for path in [self.pipleine_outpath, self.module_outpath, self.config_outpath]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    def reset_outpaths(self, new_pipeline_path, new_mod_path, new_config_path):
        """Reset outpaths"""
        self.pipleine_outpath = Path(new_pipeline_path)
        self.module_outpath = Path(new_mod_path)
        self.config_outpath = Path(new_config_path)
        self.create_out_dirs()

    def save_output_array(
        self, img_name, output_array, module_name, platform, bitdepth
    ):
        """
        Saves output array [raw/rgb] for pipline modules
        """

        # filename identifies input image and isp pipeline module for which the output
        # array is saved
        filename = self.module_outpath / (module_name + img_name.split(".")[0])

        if platform["save_format"] == "npy" or platform["save_format"] == "both":
            # save image as npy array
            np.save(filename, output_array.astype("uint16"))

        if platform["save_format"] == "png" or platform["save_format"] == "both":
            # convert image to 8-bit image if required
            if output_array.dtype != np.uint8 and len(output_array.shape) > 2:
                shift_by = bitdepth - 8
                output_array = (output_array >> shift_by).astype("uint8")

            # save Image as .png
            plt.imsave(filename.with_suffix(".png"), output_array)

    def save_output_array_yuv(self, img_name, output_array, module_name, platform):
        """
        Saves output array [yuv] for pipline modules
        """

        # filename identifies input image and isp pipeline module for which the output
        # array is saved
        filename = self.module_outpath / (module_name + img_name.split(".")[0])

        # save image as .npy array
        if platform["save_format"] == "npy" or platform["save_format"] == "both":
            np.save(filename, output_array.astype("uint16"))

        # save image as .png
        if platform["save_format"] == "png" or platform["save_format"] == "both":
            plt.imsave(filename.with_suffix(".png"), output_array)

    def save_output_3a(self, img_name, output_array, module_name):
        """Save the stats returned by the 3A module in a txt file"""

        filename = self.module_outpath / (module_name + img_name.split(".")[0])

        if output_array is not None:
            with open(filename.with_suffix(".txt"), "w", encoding="utf-8") as txt_file:
                if "auto_white_balance" in module_name:
                    print(f"RGain = {output_array[0]}", file=txt_file)
                    print(f"BGain = {output_array[1]}", file=txt_file)
                elif "auto_exposure" in module_name:
                    pass

    def save_pipeline_output(self, img_name, output_img, config_file):
        """
        Saves the output image (png) and config file at pipleine_outpath
        """

        # Time Stamp for output filename
        dt_string = datetime.now().strftime("_%Y%m%d_%H%M%S")

        # Set list format to flowstyle to dump yaml file
        yaml.add_representer(list, represent_list)

        # Storing configuration file for output image
        with open(
            self.config_outpath / (img_name + dt_string + ".yaml"),
            "w",
            encoding="utf-8",
        ) as file:
            yaml.dump(
                config_file,
                file,
                sort_keys=False,
                Dumper=CustomDumper,
                width=17000,
            )

        # Save Image as .png
        plt.imsave(self.pipleine_outpath / (img_name + dt_string + ".png"), output_img)

"""
File: utils.py
Description: Common helper functions for all algorithms
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import os
from shutil import move
from pathlib import Path
import yaml
from fxpmath import Fxp
import numpy as np


def get_approximate(decimal, register_bits, frac_precision_bits):
    """
    Returns Fixed Float Approximation of a decimal
    """
    fixed_float = Fxp(decimal, False, register_bits, frac_precision_bits)
    return fixed_float(), fixed_float.bin()


def rev_yuv(arr):
    """This function reverses the order of YUV channels in yuv image ARR."""
    if len(arr.shape) != 3:
        print("Input array must be 3D. Input array not modified.")
        return

    out_arr = np.zeros(arr.shape, arr.dtype)
    out_arr[:, :, 0] = arr[:, :, 2]
    out_arr[:, :, 1] = arr[:, :, 1]
    out_arr[:, :, 2] = arr[:, :, 0]

    return out_arr


# automation script related utilities
def is_valid(ref_lst, sub_lst):
    """This function checks the order of the ennteries of SUB_LST w.r.t
    REF_LST. Returns True if SUB_LST entries are in the same order as
    REF_LST, False otherwise."""

    # get indices w.r.t reference list of each entry in sub list
    dut_idx = []
    for entry in sub_lst:
        if entry in ref_lst:
            dut_idx.append(ref_lst.index(entry))

    dut_array = np.array(dut_idx)

    # compute differences of the consective indices
    differences = np.diff(dut_array)

    # The list is in ascending order if all differences are non-negative.
    return np.all(differences >= 0)


def arrange_channels(arr):
    """This function changes the shape of the NumPy array
    ARR from (height,width,channel) to (channel, height, width)"""
    if len(arr.shape) != 3:
        print("Input array must be 3D. Input array not modified.")
        return
    height, width, channel = arr.shape
    out_arr = np.zeros((channel, height, width), arr.dtype)
    out_arr[0, :, :] = arr[:, :, 0]
    out_arr[1, :, :] = arr[:, :, 1]
    out_arr[2, :, :] = arr[:, :, 2]

    return out_arr


def rename_files(path, sub_str):
    """This function appends In_ substring to files which are input to the DUT.
    These files are identified by SUB_STR."""

    files = [f_name for f_name in os.listdir(path) if sub_str in f_name]
    for file in files:
        new_name = "In_" + "_".join(file.split("_")[1:])
        os.rename(path + file, path + new_name)


def find_files(filename, search_path):
    """
    This function is used to find the files in the search_path
    """
    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
    return False


def restructure_dir(path, out_module_name):
    """
    This function restuctures the given folder as follows:
    - folder (specified by path)
        - input  (input to DUT file with "In_" in filename)
        - GM_out  (output to DUT file with "Out_" in filename)
        - config file
        - Input to isp_pipeline files
        - Output of isp_pipeline files
    """
    files = [
        f_name
        for f_name in os.listdir(path)
        if ".npy" == f_name[-4:] or ".png" == f_name[-4:]
    ]

    in_dir = Path(path).joinpath("input")
    out_dir = Path(path).joinpath("GM_out")

    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if file[0:3] == "In_":
            move(path + file, str(in_dir) + "/" + file)

        if "Out_" + out_module_name + "_" in file:
            move(path + file, str(out_dir) + "/" + file)


def convert_npytobin(path, nametag, raw_ext=".bin"):
    """This function takes a numpy array and saves each channel of the single
    file (.npy) as a binary file (.bin) in the given directory."""

    dir_path = "./" + str(Path(path).parent) + "/"
    in_filename = str(Path(path).stem)

    try:
        np_arr = np.load(path)
    except ValueError:
        np_arr = np.load(path, allow_pickle=False)
        print("ValueError in", in_filename)
    if len(np_arr.shape) == 3:
        for idx, channel in enumerate(nametag):
            filename = channel + "_" + in_filename + raw_ext
            with open(dir_path + filename, "wb") as raw_file:
                np_arr[:, :, idx].tofile(raw_file)
    else:
        if len(np_arr.shape) != 2:
            print("Input array must be a 2D array. Input array not saved.")
            return
        with open(dir_path + in_filename + raw_ext, "wb") as raw_file:
            np_arr.tofile(raw_file)
        return np_arr


def get_input_tv(path, rev_flag, rgb_conv_flag):

    """This function converts the NumPy arrays to binary files in input folder,
    and reorders the output files to ensure accurate comparison."""

    # for Input files convert npy to raw
    filelist = [file for file in os.listdir(path + "input/") if ".npy" == file[-4:]]
    if not filelist:
        print("Empty folder found. Enable is_save flag to save input TVs.")
        exit()
    # check if module output is RGB or YUV
    yuv_out = [
        "color_space_conversion",
        "2d_noise_reduction",
    ]
    rgb_out = [
        "demosaic",
        "color_correction_matrix",
        "gamma_correction",
    ]

    rgb_or_yuv = [
        "rgb_conversion",
        "invalid_region_crop",
        "scale",
        "yuv_conversion_format",
    ]
    any_file_name = filelist[0]

    if any([m_name in any_file_name for m_name in yuv_out]):
        if rev_flag:
            f_nametag = "RGB"
        else:
            f_nametag = "BGR"

    elif any([m_name in any_file_name for m_name in rgb_out]):
        f_nametag = "RGB"

    elif any([m_name in any_file_name for m_name in rgb_or_yuv]):
        if rev_flag is False and rgb_conv_flag is False:
            f_nametag = "BGR"
        else:
            f_nametag = "RGB"

    else:
        f_nametag = ""

    for file in filelist:
        f_path = path + "input/" + file
        convert_npytobin(f_path, f_nametag)

    # for output files change sequence of channels (h,w,ch)-->(ch,h,w)
    files = [file for file in os.listdir(path + "GM_out/") if ".npy" == file[-4:]]
    for file in files:
        f_path = path + "GM_out/" + file
        np_arr = np.load(f_path)
        if len(np_arr.shape) == 3:
            out_arr = arrange_channels(np_arr)
            np.save(path + "GM_out/" + file, out_arr)


# update config params
def update_config(config_file, module, keys, values, is_save):
    """This function updates a scecific dictionary (module) in a nested dictionary(config file)."""
    for key, value in zip(keys, values):
        if keys and config_file[module][key] != value:
            config_file[module][key] = value

    # enable is_debug flag
    config_file[module]["is_debug"] = True

    if is_save:
        config_file[module]["is_save"] = True
    else:
        config_file[module]["is_save"] = False


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


def save_config(dict_obj, save_path):
    """This function saves a dictionary as a yaml file using custom dumper."""

    # Set list format to flowstyle to dump yaml file
    yaml.add_representer(list, represent_list)

    with open(save_path, "w", encoding="utf-8") as file:
        yaml.dump(
            dict_obj,
            file,
            sort_keys=False,
            Dumper=CustomDumper,
            width=17000,
        )

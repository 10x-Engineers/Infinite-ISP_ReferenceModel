"""
File: utils.py
Description: Common helper functions for all algorithms
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import os
import sys
from shutil import move
from pathlib import Path
import subprocess
import argparse
import re
import yaml
from fxpmath import Fxp
import numpy as np
import pandas as pd


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


# function to define key for the sorted function
def num_tags(file_name):
    """Extract numerical tag present at the very end of file_name."""
    try:
        f_tag = int(file_name.split("_")[-1].split(".")[0])

    except (IndexError, ValueError):
        print(
            "Append Numerical tags at the end of file name to define "
            "file ordering for video processing."
        )

        exit()

    return f_tag


# Burst capture verification
def sensor_bin_to_sensor_raw(path, height, width, bits, bayer):
    """Converts the image sensor memory dumps (.bin) of
    RAW Burst Capture from the FPGA Platform to
    Bayer RAW frames (.raw) containing valid pixel
    data.
    The function loops over multiple scenes and replaces
    the binary files with raw files."""

    raw_paths = []
    for scene_dir in path.iterdir():
        if scene_dir.is_dir():
            for subdir in scene_dir.iterdir():
                if subdir.is_dir() and "RAW" in subdir.name:
                    raw_paths.append(subdir)

    # Images are stored in the form of rows where the size of each row in bytes
    # should be a multiple of 256, each such row size is called 'stride'
    # For raw10 format, 3 pixels are packed into 4 bytes
    stride = np.floor(np.floor(np.floor((width + 2) / 3) * 4 + 256 - 1) / 256) * 256
    stride = stride.astype(np.uint16)
    pixels_in_stride = int((stride / 4) * 3)

    for scene_path in raw_paths:
        # Look for raw files to check if the data needs processing
        scene_raw_files = list(scene_path.glob("*.raw"))
        scene_bin_files = list(scene_path.glob("*.bin"))

        if len(scene_raw_files) != len(scene_bin_files):
            start, end = 1, len(scene_bin_files)
            scene_name = scene_bin_files[0].name.split("_")[1]

            for index in range(start, end + 1):
                # reading the dumped binary file
                filename = (
                    "RAW_"
                    + scene_name
                    + "_"
                    + str(width)
                    + "x"
                    + str(height)
                    + "_"
                    + str(bits)
                    + "bits_"
                    + bayer
                    + "_"
                    + str(index)
                    + ".bin"
                )
                filepath = scene_path / filename
                print(
                    "Processing "
                    + scene_path.parent.name
                    + " RAW"
                    + str(index)
                    + " ..."
                )
                with open(filepath, "rb") as file:
                    # read the contents of the file into a new array
                    arr = np.fromfile(file, dtype=np.uint8)

                # Reshape the array into groups of 4 elements
                grouped_array = arr.reshape(-1, 4)
                flipped_array = np.flip(grouped_array, axis=1)
                result_list = []
                for inner_array in flipped_array:
                    # Convert each element to binary and concatenate
                    binary_concatenated = "".join(
                        [format(x, "08b") for x in inner_array]
                    )
                    result_list.append(
                        (
                            int(binary_concatenated[22:32], 2),
                            int(binary_concatenated[12:22], 2),
                            int(binary_concatenated[2:12], 2),
                        )
                    )
                img = (
                    np.array(result_list)
                    .reshape((height, pixels_in_stride))[:, 0:width]
                    .astype(np.uint16)
                )

                # dumping a .raw file for inf_isp
                filename = filename[:-4]
                extension = ".raw"

                with open(
                    "{}{}".format(str(scene_path / filename), extension), "wb"
                ) as file:
                    img.tofile(file)


def rev_fpga_bin(
    path, out_height, out_width, bits, bayer, sns_height, sns_width, sel_format
):
    """Converts multiple ISP output memory dumps (.bin)
    from the FPGA Platform to corresponding output
    reversing the order since the file that came from
    FPGA is BGR/YUV BGR/YUV BGR/YUV and so on."""

    isp_out_paths = []
    for scene_dir in path.iterdir():
        if scene_dir.is_dir():
            for subdir in scene_dir.iterdir():
                if subdir.is_dir() and "ISPout" in subdir.name:
                    isp_out_paths.append(subdir)

        # create a new dierctory to save the proocessed data
        Path(scene_dir / "processed_fpga_output").mkdir(parents=True, exist_ok=True)

    # Images are stored in the form of rows where the size of each row in bytes
    # should be a multiple of 256, each such row size is called 'stride'
    stride = np.floor(np.floor(np.floor((out_width * 3 + 255) / 256)) * 256)
    stride = stride.astype(np.uint16)

    for scene_path in isp_out_paths:
        # Look for processed files to check if the data needs processing
        proc_files = list((scene_path.parent / "processed_fpga_output").glob("*.bin"))
        scene_bin_files = list(scene_path.iterdir())
        if len(proc_files) != len(scene_bin_files):
            start, end = 1, len(scene_bin_files)
            scene_name = scene_bin_files[0].name.split("_")[1]

            for iter_num in range(start, end + 1):
                # repalce the first number in the filename as per the iteration number
                filename = (
                    sel_format
                    + "_"
                    + scene_name
                    + "_"
                    + str(sns_width)
                    + "x"
                    + str(sns_height)
                    + "_"
                    + str(bits)
                    + "bits_"
                    + bayer
                    + "_"
                    + str(iter_num)
                    + ".bin"
                )
                print(
                    "Processing "
                    + scene_path.parent.name
                    + " BIN"
                    + str(iter_num)
                    + " ..."
                )
                with open(scene_path / filename, "rb") as file:
                    arr = np.fromfile(file, dtype=np.uint8)
                file.close()

                arr = np.reshape(arr, (out_height, stride))
                # print(arr.shape)

                arr_trun = arr[:, 0 : out_width * 3]
                # print(arr_trun.shape)

                # flatten the shape
                arr_flat = arr_trun.flatten()
                arr_flat_u16 = arr_flat.astype(np.uint16)
                arr_corrected = np.zeros(arr_flat_u16.shape, dtype=np.uint16)

                # reversing the order since the file that came from
                # FPGA is BGR/YUV BGR/YUV BGR/YUV ...
                arr_corrected[0::3] = arr_flat[2::3]
                arr_corrected[1::3] = arr_flat[1::3]
                arr_corrected[2::3] = arr_flat[0::3]

                # dumping the strides removed binary file in the FPGA_Bin directory
                arr_corrected.tofile(
                    scene_path.parent / "processed_fpga_output" / filename
                )


def fpga_dir_is_valid(directory_path, start_frame_num=None, end_frame_num=None):
    """
    This function checks if the FPGA results are placed correctly.
    Each scene should have a the following structure.
    Scene
      ├── ISPout (with FPGA output)
      ├── RAW    (with .bin input files for Infinite-ISP_ReferenceModel)
      ├── Stats_Debug.txt (with 3A stats from FPGA)
      └── PrintfLogs.txt (with interrupt information)
    The directory is said to be a valid directory if
    contains equal numer of raw files, binary files
    (fpga output) and number of rows in the text file
    with 3A stats.
    """

    for scene in directory_path.iterdir():
        # List the directory content
        scene_content = [file.name for file in scene.iterdir()]

        if (
            "RAW" in scene_content
            and "ISPout" in scene_content
            and "Stats_Debug.txt" in scene_content
        ):
            # read the text file with stats
            stats_df = pd.read_csv(scene / "Stats_Debug.txt", delimiter=r"\s+")

            if start_frame_num is None or end_frame_num is None:
                # compare all frames
                # count all raw files in RAW directory
                num_raw_files = len(list((scene / "RAW").glob("*.raw")))

                # count all bin files in ISPout directory
                num_bin_files = len(list((scene / "ISPout").glob("*.bin")))

                # count the number of rows
                number_of_frame_stats = stats_df.shape[0]

                if num_bin_files == num_raw_files == number_of_frame_stats:
                    continue
                else:
                    # Extra stats are given
                    if num_raw_files <= number_of_frame_stats:
                        # Look for stats corresponding to each frame
                        for raw_file in (scene / "RAW").iterdir():
                            # get frame num
                            frame_num = num_tags(raw_file.name)
                            frame_stats_exist = frame_num in stats_df["Frame"].values
                            if not frame_stats_exist:
                                print(f"Stats not found in {scene.name}!")
                                exit()
                    else:
                        print(
                            f"Insufficient information to compare all frames in {scene.name}!"
                        )
                        exit()
            else:
                # To compare consective frames
                # look for frames from num start_frame_num to end_frame_num
                if start_frame_num < end_frame_num:
                    for frame_num in range(start_frame_num, end_frame_num + 1, 1):
                        num_raw_files = len(
                            list((scene / "RAW").glob(f"*{frame_num}.raw"))
                        )
                        num_bin_files = len(
                            list((scene / "ISPout").glob(f"*{frame_num}.bin"))
                        )
                        frame_stats_exist = frame_num in stats_df["Frame"].values

                        if num_raw_files != num_bin_files != 1 or not frame_stats_exist:
                            if num_raw_files == 0 or num_bin_files == 0:
                                print(f"Specified frames not found in {scene.name}!")
                                exit()
                            else:
                                print(f"Stats not found in {scene.name}!")
                                exit()
                else:
                    print("START_FRAME_NUM should be less than END_FRAME_NUM!")
                    exit()
        else:
            print(f"Invalid {scene.name} directory structure!")
            exit()

    return True


def get_folder_name(text):
    """The name of the directory containing the test vectors is
    extracted from the given text using regex."""

    pattern = r"Folder Name:\s+([\w-]+)"
    match = re.search(pattern, text)

    try:
        folder_name = re.sub(r"Folder Name:\s*", "", match.group())
    except AttributeError:
        print("Folder Name not found in printed logs.")
        exit()

    return folder_name


def arg_parse():
    """This function is used to parse arguments to the compare_output.py file."""

    list_type = lambda string: [int(i) for i in string.split(",")]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--PATH_TO_RTL_RESULTS", help="Specify path to RTL generated outputs."
    )
    parser.add_argument(
        "--PATH_TO_GM_RESULTS",
        help="Specify path to Infinite ISP generated ouptputs.",
    )
    parser.add_argument(
        "--REMOVE_ROWS",
        help="Used to remove n rows to allign RTL and Infinite ISP output.",
        type=int,
    )
    parser.add_argument(
        "--REMOVE_BORDER",
        help="Used to remove n sized border to allign RTL and Infinite ISP output.",
        type=list_type,
    )
    args = parser.parse_args()
    return args


def run_compare_output(fpga_outframes_path, foldername, remove_rows, remove_border):
    """This function compare the FPGA output and outputs generated using the current repo
    using compare_output.py."""

    gm_out_path = f"./test_vectors/{foldername}/GM_out/"

    # execute compare_output.py to compare results
    # values of the global variables must be passed as string type to
    # the argument parser
    # list is passed as a comma separated string of integers
    str_lst = (
        str(remove_border[0])
        if len(remove_border) == 1
        else ",".join(str(i) for i in remove_border)
    )

    global_params = {
        "PATH_TO_RTL_RESULTS": str(fpga_outframes_path),
        "PATH_TO_GM_RESULTS": gm_out_path,
        "REMOVE_ROWS": str(remove_rows),
        "REMOVE_BORDER": str_lst,
    }

    with open(
        Path(f"./test_vectors/{foldername}") / "FPGA_GM_comparison_results.txt",
        "a",
        encoding="utf-8",
    ) as text_file:
        subprocess.run(
            [
                sys.executable,
                "./test_vector_generation/compare_output.py",
                "--PATH_TO_RTL_RESULTS",
                global_params["PATH_TO_RTL_RESULTS"],
                "--PATH_TO_GM_RESULTS",
                global_params["PATH_TO_GM_RESULTS"],
                "--REMOVE_ROWS",
                global_params["REMOVE_ROWS"],
                "--REMOVE_BORDER",
                global_params["REMOVE_BORDER"],
            ],
            check=True,
            stdout=text_file,
            text=True,
        )


def remove_arr_border(arr, lst):
    """This function removes border from the given ARR.
    If ARR is 3 channel, the shape is assumed to be: (ch, h, w)."""

    # add axis
    if len(arr.shape) == 2:
        arr = arr.reshape((1, arr.shape[0], arr.shape[1]))

    # if lst has only one integer, equal amount of
    # rows/columns are removed from all sides
    if len(lst) == 1:
        if lst[0] != 0:
            return (
                arr[0, lst[0] : -lst[0], lst[0] : -lst[0]]
                if arr.shape[0] == 1
                else arr[:, lst[0] : -lst[0], lst[0] : -lst[0]]
            )
    elif len(lst) == 4:
        top, bottom = lst[0], lst[1]
        left, right = lst[2], lst[3]

        # remove rows from top only
        if top:
            arr = arr[:, top:, :]

        # remove rows from bottom only
        if bottom:
            arr = arr[:, 0:-bottom, :]

        # remove cols from left only
        if left:
            arr = arr[:, :, left:]

        # remove cols from right only
        if right:
            arr = arr[:, :, 0:-right]
    else:
        print("No border removed. Length of input list shoould be 1 or 4.")
        arr = arr.copy()

    return arr[0, :, :] if arr.shape[0] == 1 else arr


def remove_rows_for_comparison(gm_array, rtl_array, num):
    """This function  removes N rows from top of the RTL _ARR and
    NUM rows from the bottom of GM_ARR for comparison purposes.
    If ARR is 3 channel, the shape is assumed to be: (ch, h, w)"""

    # add axis
    if len(gm_array.shape) == 2:
        gm_array = gm_array.reshape((1, gm_array.shape[0], gm_array.shape[1]))
    if len(rtl_array.shape) == 2:
        rtl_array = rtl_array.reshape((1, rtl_array.shape[0], rtl_array.shape[1]))

    gm_array = gm_array[:, 0:-num, :]
    rtl_array = rtl_array[:, num:, :]

    # remove axis
    if gm_array.shape[0] == 1:
        gm_array = gm_array[0, :, :]
        rtl_array = rtl_array[0, :, :]

    return gm_array, rtl_array


def create_df_from_isp_logs(txt_path):
    """This function extracts the 3A stats printed in the ISP pipleine
    logs and returns a pandas data frame to facilitate stats comparison."""

    # read txt file
    with open(txt_path) as file:
        text = file.read()

    # create an empty dataframe
    dataframe = pd.DataFrame(
        columns=[
            "Frame",
            "WB_RGain",
            "WB_BGain",
            "DG_Gain",
            "AWB_RGain",
            "AWB_BGain",
            "AE_Skewness",
            "Target_Skewness",
            "New_DG_Gain",
            "AE_Response",
        ]
    )

    # Extract filenames
    pattern = r"Filename:\s+(.*)\n"
    filenames = re.findall(pattern, text)
    frame_num = [int(f_name.split("_")[-1].split(".")[0]) for f_name in filenames]
    dataframe["Frame"] = frame_num

    # Extract AWB R and B gain and save the hex values as strings in the dataframe
    r_awb_pattern = r"(?i)-\s+AWB\s+-\s+RGain\s+=\s+([01]{16})"
    r_awb_gains = re.findall(r_awb_pattern, text)
    awb_rgain = list(map(hex, [int(binary_val, 2) for binary_val in r_awb_gains]))
    dataframe["AWB_RGain"] = awb_rgain

    b_awb_pattern = r"(?i)-\s+AWB\s+-\s+BGain\s+=\s+([01]{16})"
    b_awb_gains = re.findall(b_awb_pattern, text)
    awb_bgain = list(map(hex, [int(binary_val, 2) for binary_val in b_awb_gains]))
    dataframe["AWB_BGain"] = awb_bgain

    # Extract WB R and B gain and save the hex values as strings in the dataframe
    r_wb_pattern = r"-\s+WB\s+-\s+red gain \(U16\.8\):.*([01]{16})"
    r_wb_gains = re.findall(r_wb_pattern, text)
    wb_rgain = list(map(hex, [int(binary_val, 2) for binary_val in r_wb_gains]))
    dataframe["WB_RGain"] = wb_rgain

    b_wb_pattern = r"-\s+WB\s+-\s+blue gain \(U16\.8\):.*([01]{16})"
    b_wb_gains = re.findall(b_wb_pattern, text)
    wb_bgain = list(map(hex, [int(binary_val, 2) for binary_val in b_wb_gains]))
    dataframe["WB_BGain"] = wb_bgain

    # Extract Digital gain for current frame and save the integer values in dataframe
    dg_pattern = r"-\s+DG\s+-\s+Applied Gain\s+=\s+(\d+)"
    dg_idx = re.findall(dg_pattern, text)
    dg_idx = [int(val) for val in dg_idx]
    dataframe["DG_Gain"] = dg_idx

    # Extract AE response and save as integer values in dataframe
    ae_res_pattern = r"(?i)-\s+3A\s+Stats\s+-\s+AE Feedback\s+=\s+(\w+\s?\w+)"
    ae_res = re.findall(ae_res_pattern, text)

    for idx, response in enumerate(ae_res):
        if response == "Underexposed":
            ae_res[idx] = -1
        elif response == "Overexposed":
            ae_res[idx] = 1
        elif response == "Correct Exposure":
            ae_res[idx] = 0

    dataframe["AE_Response"] = ae_res

    # Extract AE target skeness and save the hex values as strings in dataframe
    target_skewness_pattern = r"-\s+AE\s+-\s+Histogram Skewness Range\s+=\s+(\d+\.\d+)"
    ae_tar_skewness = re.findall(target_skewness_pattern, text)
    ae_tar_skewness = [hex(int(float(val) * 256)) for val in ae_tar_skewness]
    dataframe["Target_Skewness"] = ae_tar_skewness

    # Extract AE approx skeness and save the hex values as strings in dataframe
    skewness_pattern = r"-\s+AE\s+-\s+Approx_Skewness Int\s+=\s+-?(\d+\.\d+)"
    ae_skewness = re.findall(skewness_pattern, text)
    ae_skewness = [hex(int(float(val) * 256)) for val in ae_skewness]
    dataframe["AE_Skewness"] = ae_skewness

    dataframe.to_csv(f"{txt_path.parent}/GM_Stats_Debug.xlsx", index=False)

    return dataframe


def compare_frame_stats(fpga_stats_path, gm_stats_path):
    """To compare the AWB and AE stats, the function takes in two text files
    reads and compares the stats."""

    fpga_stats_df = pd.read_csv(fpga_stats_path, delimiter=r"\s+")
    gm_stats_df = create_df_from_isp_logs(gm_stats_path)

    # Remove New_DG_Index as this is not associated with the current frame
    fpga_stats_df.drop("New_DG_Gain", axis=1, inplace=True)
    gm_stats_df.drop("New_DG_Gain", axis=1, inplace=True)

    mismatch = False
    for index, row in gm_stats_df.iterrows():
        for column, value in row.items():
            if isinstance(value, int):
                # Perform comparison for integer values
                if value != gm_stats_df.at[index, column]:
                    print(f"Mismatch: Frame {row['Frame']}, {column}")
                    print(f"FPGA: {value}    Infinite-ISP_ReferenceModel: {gm_stats_df.at[index, column]}")

                    mismatch = True

            elif isinstance(value, str):
                # Convert hexadecimal strings to integers for comparison
                fpga_hex_value = int(value, 16)
                gm_hex_value = int(gm_stats_df.at[index, column], 16)
                if fpga_hex_value != gm_hex_value:
                    print(f"Mismatch: Frame {row['Frame']},  {column}")
                    print(f"FPGA: {fpga_hex_value}    Infinite-ISP_ReferenceModel: {gm_hex_value}")
                    mismatch = True
    if not mismatch:
        print("No Mismatch Found!!!")


# Function to extract numbers from a string
def extract_numbers(line):
    """Function to extract numbers from a string"""
    return [int(num) for num in re.findall(r"\d+", line)]


def raw_isvalid(fpga_outframes_path, min_val, max_val, line_number):
    """This function checks teh validity of the raw files based
    on the interupt information."""
    isvalid = False

    # Iterate through all folders in the current directory
    for foldername in os.listdir(fpga_outframes_path):
        folder_path = os.path.join(fpga_outframes_path, foldername)

        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path):
            logs_file_path = os.path.join(folder_path, "Printf_Logs.txt")

            # Check if logs.txt exists in the current folder
            if os.path.exists(logs_file_path):
                with open(logs_file_path, "r") as file:
                    lines = file.readlines()

                    # Ensure the file has at least 10 lines
                    if len(lines) >= line_number:
                        tenth_line = lines[
                            line_number - 1
                        ].strip()  # Index 9 corresponds to the 10th line (0-indexed)

                        # Extract numbers from the line
                        numbers = extract_numbers(tenth_line)

                        # Extract the last six numbers
                        last_six_values = numbers[-6:]

                        # Assign the last six values to separate integer variables if
                        # there are at least six values
                        if len(last_six_values) >= 6:
                            var1, var2, var3, var4, var5, var6 = last_six_values
                            if (
                                (var1 < min_val)
                                | (var1 > max_val)
                                | (var2 < min_val)
                                | (var2 > max_val)
                                | (var3 < min_val)
                                | (var3 > max_val)
                                | (var4 < min_val)
                                | (var4 > max_val)
                                | (var5 < min_val)
                                | (var5 > max_val)
                                | (var6 < min_val)
                                | (var6 > max_val)
                            ):
                                print("Error in the folder:", foldername)
                                return False

                                # print("Folder:", foldername)
                                # print("Separate variables:", var1, var2, var3, var4, var5, var6)
                            else:
                                print("Succesful Run for folder:", foldername)
                                isvalid = True
                        else:
                            print("Folder:", foldername)
                            print(
                                "Insufficient data to extract last six values in",
                                logs_file_path,
                            )
                            return False
                    else:
                        print(f"Not enough lines in text file for {foldername}")
                        return False
            else:
                print(f"Printf_Logs File not found in {foldername}!")
                isvalid = True

    return isvalid


def update_datapath(tv_config_path, dir_name):
    """This function updates the datapath in tv_config file."""

    # read the tv_config
    with open(tv_config_path, "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    # update datapath
    parent_dir = Path(config["dataset_path"]).parent.parent
    raw_folder_name = Path(config["dataset_path"]).name
    config["dataset_path"] = "./" + str(parent_dir / dir_name / raw_folder_name)

    # save config file
    save_config(config, tv_config_path)


def update_config_for_scene(tv_config_path, fpga_stats_path):
    """Update the AWB gains and DG index in config file automatically
    when processing a new scene."""

    # Read stats
    fpga_stats_df = pd.read_csv(fpga_stats_path, delimiter=r"\s+")

    # Extract WB gains and DG
    wb_rgain = fpga_stats_df["WB_RGain"][0]
    wb_bgain = fpga_stats_df["WB_BGain"][0]
    digital_gain = fpga_stats_df["DG_Gain"][0]

    # convert hex values to decimal (float), for 8 bit precesion: divide by 2^8
    wb_rgain = int(wb_rgain, 16) / 256
    wb_bgain = int(wb_bgain, 16) / 256

    # Read tv_config
    with open(tv_config_path, "r", encoding="UTF-8") as file:
        tv_config = yaml.safe_load(file)

    # Get config path
    config_path = Path(tv_config["config_path"])

    # Read config
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Update values
    dg_index = config["digital_gain"]["gain_array"].index(digital_gain)

    config["white_balance"]["r_gain"] = wb_rgain
    config["white_balance"]["b_gain"] = wb_bgain
    config["digital_gain"]["current_gain"] = dg_index

    # Save config
    save_config(config, config_path)


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


def reconstrct_yuv422_for_rtl(arr, height, width):
    """Reconstruct a YUV from YUV 422 format."""

    # Create an empty 3D YUV image (height, width, channels)
    rtl_img = np.zeros((height * width * 2,), dtype=np.uint16)

    # select y, u and v channels from the binary input array
    arr_y = arr[2::3]
    arr_c = arr[1::3]

    # Rearrange the channels to construct 3D YUV image
    rtl_img[0::2] = arr_y
    rtl_img[1::2] = arr_c

    return rtl_img

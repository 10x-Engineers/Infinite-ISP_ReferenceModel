"""
DESCRIPTION: ...
"""

import sys
import os
from pathlib import Path
import shutil
from datetime import datetime
import yaml
import tv_utils

# add parent path to system path
sys.path.append(".")
import infinite_isp

# load and read the automation config file
with open("./test_vector_generation/tv_config.yml", "r", encoding="utf-8") as file:
    automation_config = yaml.safe_load(file)

DATASET_PATH = automation_config["dataset_path"]
CONFIG_PATH = automation_config["config_path"]
INPUT_EXT = automation_config["input_ext"]
REV_YUV = automation_config["rev_yuv"]

# The input array of the first module and the output array of
# the last module are saved in test_vectors directory.
DUT = automation_config["dut"]

# Read the status of each module from the config file
CROP = automation_config["crop"]
DPC = automation_config["dead_pixel_correction"]
BLC = automation_config["black_level_correction"]
OECF = automation_config["oecf"]
DG = automation_config["digital_gain"]
BNR = automation_config["bayer_noise_reduction"]
WB = automation_config["white_balance"]
DEM = automation_config["demosaic"]
CCM = automation_config["color_correction_matrix"]
GC = automation_config["gamma_correction"]
CSC = automation_config["color_space_conversion"]
NR2D = automation_config["2d_noise_reduction"]
RGB_CONV = automation_config["rgb_conversion"]
IRC = automation_config["invalid_region_crop"]
SCALE = automation_config["scale"]
YUV = automation_config["yuv_conversion_format"]


# Define Directory name with datetime stamp to save outputs
dut_names = DUT[0] if len(DUT) == 1 else DUT[0] + "_to_" + DUT[-1]
folder_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + dut_names

# Parent folder for Images (Path is relative to ./in_frames/normal/)
PARENT_FOLDER = DATASET_PATH.rsplit("./in_frames/normal/", maxsplit=1)[-1]

# Get the list of all files in the DATASET_PATH
DIRECTORY_CONTENT = os.listdir(DATASET_PATH)

# Save a copy of the default config file
RETAINED_CONFIG = "./config/default-configs.yml"
shutil.copy(CONFIG_PATH, RETAINED_CONFIG)

# create a folder to save the Results, delete directory if it exists already
SAVE_PATH = "./test_vectors/Results/"
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)
Path(SAVE_PATH).mkdir(parents=True, exist_ok=False)

# load and update config file
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# remove modules that do not process the current frame
default_modules = ["digital_gain", "demosaic", "color_space_conversion"]
module_tags = list(config.keys())[2:]
remove = [
    "auto_exposure",
    "auto_white_balance",
]
_ = [module_tags.remove(module) for module in remove]

# Check if the DUT list is defined corectly
if not tv_utils.is_valid(module_tags, DUT):
    print("The sequence of modules in DUT is incorrect.")
    exit()

# ensure that the modules are in same order as they are in the default config file
new_params = [
    CROP,
    DPC,
    BLC,
    OECF,
    DG,
    BNR,
    WB,
    DEM,
    CCM,
    GC,
    CSC,
    NR2D,
    RGB_CONV,
    IRC,
    SCALE,
    YUV,
]

# Set generate_tv to True to indicate that automation file is being executed
config["platform"]["generate_tv"] = True

# update save_lut flag in config
config["platform"]["save_lut"] = True

# update rev_uv flag in config
config["platform"]["rev_yuv_channels"] = REV_YUV

# It is mandatory to save test vectors as numpy arrays
if config["platform"]["save_format"] == "png":
    config["platform"]["save_format"] = "both"

# update render_3a flag in config
config["platform"]["render_3a"] = False

for idx, module in enumerate(module_tags):
    # save the input and output arrays of module under test with is_save flag
    try:
        # save the input to DUT with is_save flag
        if module in DUT or module_tags[idx + 1] in DUT:
            IS_SAVE = True
        elif module in default_modules:

            IS_SAVE = new_params[idx]["is_save"]
        else:
            IS_SAVE = False

    except IndexError:
        IS_SAVE = True if module in DUT else False

    # Enable non-default modules in DUT
    if module in DUT and module not in default_modules:
        new_params[idx]["is_enable"] = True

    tv_utils.update_config(
        config, module, new_params[idx].keys(), new_params[idx].values(), IS_SAVE
    )
# Set list format to flowstyle to dump yaml file
yaml.add_representer(list, tv_utils.represent_list)

# save the automation config along with generated results
with open(SAVE_PATH + "/tv_config.yml", "w", encoding="utf-8") as file:
    yaml.dump(
        automation_config,
        file,
        sort_keys=False,
        Dumper=tv_utils.CustomDumper,
        width=17000,
    )

# loop over images
RAW_FILENAMES = [
    filename
    for filename in DIRECTORY_CONTENT
    if Path(DATASET_PATH, filename).suffix in [".raw"]
]
# update filename in config
config["platform"]["filename"] = RAW_FILENAMES[0]

# save the created config file as a yaml file along with its results
with open(SAVE_PATH + "/configs_automate.yml", "w", encoding="utf-8") as file:
    yaml.dump(config, file, sort_keys=False, Dumper=tv_utils.CustomDumper, width=17000)

# create and infinite_isp object
inf_isp = infinite_isp.InfiniteISP(DATASET_PATH, SAVE_PATH + "/configs_automate.yml")

for i, raw_filename in enumerate(RAW_FILENAMES):

    # update filename in infinite_isp object
    inf_isp.raw_file = raw_filename

    with open(
        SAVE_PATH + "/isp_pipeline_log.txt",
        "a",
        encoding="utf-8",
    ) as text_file:
        sys.stdout = text_file
        inf_isp.execute()
        sys.stdout = sys.__stdout__

# Remove path from sys path
sys.path.remove(".")

# Place back the original config file
shutil.copy(RETAINED_CONFIG, CONFIG_PATH)
os.remove(RETAINED_CONFIG)

# rename output files of the previous module to DUT as "In_" to identify these files
# as input to DUT files
if "crop" in DUT:
    tv_utils.rename_files(SAVE_PATH, "Inpipeline_crop_")
else:
    input_module = module_tags[module_tags.index(DUT[0]) - 1]
    tv_utils.rename_files(SAVE_PATH, "Out_" + input_module + "_")

# restructure the directory
tv_utils.restructure_dir(SAVE_PATH, DUT[-1])

# convert the saved numpy arrays to bin files
tv_utils.get_input_tv(SAVE_PATH, INPUT_EXT, REV_YUV, RGB_CONV["is_enable"])

# Remove empty folder andrename results folder with datetime stamp
os.rename(SAVE_PATH, f"./test_vectors/{folder_name}")

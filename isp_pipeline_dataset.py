"""
This script is used to run isp_pipeline.py on a dataset placed in ./inframes/normal/data
It also fetches if a separate config of a raw image is present othewise uses the default config
"""

import os
from pathlib import Path
import yaml
from tqdm import tqdm
from test_vector_generation import tv_utils
from infinite_isp import InfiniteISP

# The path of the dataset
DATASET_PATH = "./in_frames/normal/data/"

# Parent folder for Images (Path is relative to ./in_frames/normal/)
PARENT_FOLDER = DATASET_PATH.rsplit('./in_frames/normal/', maxsplit=1)[-1]

# The path for default config
DEFAULT_CONFIG = "./config/configs.yml"

# Get the list of all files in the DATASET_PATH
DIRECTORY_CONTENT = os.listdir(DATASET_PATH)

# Get the list of all raw images in the DATASET_PATH
raw_images = [
    x
    for x in DIRECTORY_CONTENT
    if (Path(DATASET_PATH, x).suffix in [".raw", ".NEF"])
]

def find_files(filename, search_path):
    """
    This function is used to find the files in the search_path
    """
    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
    return False

# Set list format to flowstyle to dump yaml file
yaml.add_representer(list, tv_utils.represent_list)

infinite_isp = InfiniteISP(DATASET_PATH, DEFAULT_CONFIG)

# set generate_tv flag to false
infinite_isp.c_yaml["platform"]["generate_tv"] = False

IS_DEFAULT_CONFIG = True

for raw in tqdm(raw_images, ncols=100, leave=True):

    config_file = Path(raw).stem + "-configs.yml"

    # check if the config file exists in the DATASET_PATH
    if find_files(config_file, DATASET_PATH):

        print(f"Found {config_file}.")

        # use raw config file in dataset
        infinite_isp.load_config(DATASET_PATH + config_file)
        IS_DEFAULT_CONFIG = False
        infinite_isp.execute()

    else:
        print(f"Not Found {config_file}, Changing filename in default config file.")

        # copy default config file
        if not IS_DEFAULT_CONFIG:
            infinite_isp.load_config(DEFAULT_CONFIG)
            IS_DEFAULT_CONFIG = True

        infinite_isp.execute(raw)

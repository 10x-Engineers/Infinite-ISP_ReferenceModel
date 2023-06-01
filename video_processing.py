"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import os
from tqdm import tqdm
from infinite_isp import InfiniteISP

CONFIG_PATH = "./config/configs.yml"

# Get the path to the inputfile
DATASET_PATH = "./in_frames/normal"
raw_files = [f_name for f_name in os.listdir(DATASET_PATH) if ".raw" in f_name]
raw_files.sort()

infinite_isp = InfiniteISP(DATASET_PATH, CONFIG_PATH)

# set generate_tv flag to false
infinite_isp.c_yaml["platform"]["generate_tv"] = False

for file in tqdm(raw_files, disable=False, leave=True):

    infinite_isp.execute(file)
    infinite_isp.load_3a_statistics()

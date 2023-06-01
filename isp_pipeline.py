"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP

CONFIG_PATH = "./config/configs.yml"
RAW_DATA = "./in_frames/normal"

infinite_isp = InfiniteISP(RAW_DATA, CONFIG_PATH)

# set generate_tv flag to false
infinite_isp.c_yaml["platform"]["generate_tv"] = False

infinite_isp.execute()

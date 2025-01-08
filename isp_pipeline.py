"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP

CONFIG_PATH = "./in_frames/normal/data/Outdoor1_2592x1536_10bit_GRBG-configs.yml"
RAW_DATA = "./in_frames/normal/data"
FILENAME = None

if __name__ == "__main__":
    infinite_isp = InfiniteISP(RAW_DATA, CONFIG_PATH)
    infinite_isp.execute(img_path=FILENAME)

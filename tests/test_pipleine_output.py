"""
File: test_pipeline_output.py
Description: Basic tests to ensure the pipeline is functioning correctly.
             All tests are executed using files in test_data folder.
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from pathlib import Path
import sys

sys.path.append(".")
import shutil
import numpy as np
import matplotlib.pyplot as plt
import yaml
from infinite_isp import InfiniteISP


def test_output_not_blank():
    """Test to verify the pipeline output is not blank (zero frame)."""

    #
    raw_path = Path("./tests/test_data")
    config_path = raw_path / "Indoor1_2592x1536_10bit_GRBG-configs.yml"
    filemane = "Indoor1_2592x1536_10bit_GRBG.raw"

    # run pipeline on test data
    inf_isp = InfiniteISP(raw_path, config_path)
    inf_isp.set_save_paths(Path("./tests"))
    inf_isp.execute(img_path=filemane)

    # load output
    out_path = Path("./tests/out_frames")
    out_filename = list(out_path.glob(r"*Indoor1_2592x1536_10bit_GRBG*.png"))

    assert len(out_filename) == 1, "Output file not generated by infiniteISP pipeline!"

    out_png_path = out_filename[0]
    out_img = plt.imread(str(out_png_path.resolve()))

    # Check if all pixel values in the image are zero
    assert not np.all(out_img == 0), "The output frame has all zeros!"

    # restore directory state
    shutil.rmtree("./tests/out_frames")
    shutil.rmtree("./tests/config")


def test_default_config_state():
    """Test to check the default state of the config file. This configuration has
    been tuned for the sample image ColorChecker_2592x1536_10bit_GRBG."""

    # Load the default config from test_data
    with open("./tests/test_data/default_configs.yml", "r", encoding="utf-8") as file:
        def_config = yaml.safe_load(file)

    # load the config in RM directory
    with open("./config/configs.yml", "r", encoding="utf-8") as file:
        rm_config = yaml.safe_load(file)

    assert def_config["platform"] == rm_config["platform"]
    assert def_config["sensor_info"] == rm_config["sensor_info"]
    assert def_config["crop"] == rm_config["crop"]
    assert def_config["dead_pixel_correction"] == rm_config["dead_pixel_correction"]
    assert def_config["black_level_correction"] == rm_config["black_level_correction"]
    assert def_config["oecf"] == rm_config["oecf"]
    assert def_config["digital_gain"] == rm_config["digital_gain"]
    assert def_config["bayer_noise_reduction"] == rm_config["bayer_noise_reduction"]
    assert def_config["auto_white_balance"] == rm_config["auto_white_balance"]
    assert def_config["white_balance"] == rm_config["white_balance"]
    assert def_config["demosaic"] == rm_config["demosaic"]
    assert def_config["color_correction_matrix"] == rm_config["color_correction_matrix"]
    assert def_config["gamma_correction"] == rm_config["gamma_correction"]
    assert def_config["auto_exposure"] == rm_config["auto_exposure"]
    assert def_config["color_space_conversion"] == rm_config["color_space_conversion"]
    assert def_config["2d_noise_reduction"] == rm_config["2d_noise_reduction"]
    assert def_config["rgb_conversion"] == rm_config["rgb_conversion"]
    assert def_config["invalid_region_crop"] == rm_config["invalid_region_crop"]
    assert def_config["on_screen_display"] == rm_config["on_screen_display"]
    assert def_config["scale"] == rm_config["scale"]
    assert def_config["yuv_conversion_format"] == rm_config["yuv_conversion_format"]
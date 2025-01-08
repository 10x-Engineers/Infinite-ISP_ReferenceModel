"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
from pathlib import Path
import numpy as np
import yaml
import rawpy
import util.utils as util

from modules.dead_pixel_correction import DeadPixelCorrection as DPC
from modules.digital_gain import DigitalGain as DG
from modules.bayer_noise_reduction import BayerNoiseReduction as BNR
from modules.black_level_correction import BlackLevelCorrection as BLC
from modules.oecf import OECF
from modules.auto_white_balance import AutoWhiteBalance as AWB
from modules.white_balance import WhiteBalance as WB
from modules.gamma_correction import GammaCorrection as GC
from modules.demosaic import Demosaic
from modules.color_correction_matrix import ColorCorrectionMatrix as CCM
from modules.color_space_conversion import ColorSpaceConversion as CSC
from modules.sharpen import Sharpening as SHARP
from modules.yuv_conv_format import YUVConvFormat as YUV_C
from modules.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion import RGBConversion as RGBC
from modules.invalid_region_crop import InvalidRegionCrop as IRC
from modules.on_screen_display import OnScreenDisplay as OSD
from modules.scale import Scale
from modules.crop import Crop
from modules.auto_exposure import AutoExposure as AE


class InfiniteISP:
    """
    Infinite-ISP Pipeline
    """

    def __init__(self, data_path, config_path):
        """
        Constructor: Initialize with config and raw file path
        and Load configuration parameter from yaml file
        """
        self.data_path = data_path
        self.load_config(config_path)
        self.save_output_obj = util.SaveOutput()

    def load_config(self, config_path):
        """
        Load config information to respective module parameters
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as file:
            c_yaml = yaml.safe_load(file)

        # Extract workspace info
        self.platform = c_yaml["platform"]
        self.raw_file = self.platform["filename"]
        self.render_3a = self.platform["render_3a"]

        # Extract basic sensor info
        self.sensor_info = c_yaml["sensor_info"]

        # Get isp module params
        self.parm_cro = c_yaml["crop"]
        self.parm_dpc = c_yaml["dead_pixel_correction"]
        self.parm_blc = c_yaml["black_level_correction"]
        self.parm_oec = c_yaml["oecf"]
        self.parm_dga = c_yaml["digital_gain"]
        self.parm_bnr = c_yaml["bayer_noise_reduction"]
        self.parm_awb = c_yaml["auto_white_balance"]
        self.parm_wbc = c_yaml["white_balance"]
        self.parm_dem = c_yaml["demosaic"]
        self.parm_ccm = c_yaml["color_correction_matrix"]
        self.parm_gmc = c_yaml["gamma_correction"]
        self.parm_ae = c_yaml["auto_exposure"]
        self.parm_csc = c_yaml["color_space_conversion"]
        self.parm_sha = c_yaml["sharpen"]
        self.parm_2dn = c_yaml["2d_noise_reduction"]
        self.parm_rgb = c_yaml["rgb_conversion"]
        self.parm_irc = c_yaml["invalid_region_crop"]
        self.parm_sca = c_yaml["scale"]
        self.parm_yuv = c_yaml["yuv_conversion_format"]
        self.parm_osd = c_yaml["on_screen_display"]
        self.c_yaml = c_yaml

        self.platform["rgb_output"] = self.parm_rgb["is_enable"]

        return c_yaml

    def load_raw(self):
        """
        Load raw image from provided path
        """
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        self.out_file = "Out_" + self.in_file

        self.platform["in_file"] = self.in_file
        self.platform["out_file"] = self.out_file

        width = self.sensor_info["width"]
        height = self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]

        # Load Raw
        if path_object.suffix == ".raw":
            if bit_depth > 8:
                self.raw = np.fromfile(raw_path, dtype=np.uint16).reshape(
                    (height, width)
                )
            else:
                self.raw = (
                    np.fromfile(raw_path, dtype=np.uint8)
                    .reshape((height, width))
                    .astype(np.uint16)
                )
        else:
            img = rawpy.imread(raw_path)
            self.raw = img.raw_image

    def run_pipeline(self, visualize_output=True):
        """
        Run ISP-Pipeline for a raw-input image
        """

        # =====================================================================
        # Cropping
        crop = Crop(
            self.raw,
            self.platform,
            self.sensor_info,
            self.parm_cro,
            self.save_output_obj,
        )
        cropped_img = crop.execute()

        # =====================================================================
        #  Dead pixels correction
        dpc = DPC(
            cropped_img,
            self.platform,
            self.sensor_info,
            self.parm_dpc,
            self.save_output_obj,
        )
        dpc_raw = dpc.execute()

        # =====================================================================
        # Black level correction
        blc = BLC(
            dpc_raw,
            self.platform,
            self.sensor_info,
            self.parm_blc,
            self.save_output_obj,
        )
        blc_raw = blc.execute()

        # =====================================================================
        # OECF
        oecf = OECF(
            blc_raw,
            self.platform,
            self.sensor_info,
            self.parm_oec,
            self.save_output_obj,
        )
        oecf_raw = oecf.execute()

        # =====================================================================
        # Digital Gain
        dga = DG(
            oecf_raw,
            self.platform,
            self.sensor_info,
            self.parm_dga,
            self.save_output_obj,
        )
        dga_raw, self.dga_current_gain = dga.execute()

        # =====================================================================
        # Bayer noise reduction
        bnr = BNR(
            dga_raw,
            self.platform,
            self.sensor_info,
            self.parm_bnr,
            self.save_output_obj,
        )
        bnr_raw = bnr.execute()

        # =====================================================================
        # Auto White Balance
        awb = AWB(
            bnr_raw,
            self.sensor_info,
            self.parm_awb,
        )
        self.awb_gains = awb.execute()

        # =====================================================================
        # White balancing
        wbc = WB(
            bnr_raw,
            self.platform,
            self.sensor_info,
            self.parm_wbc,
            self.save_output_obj,
        )
        wb_raw = wbc.execute()

        # =====================================================================
        # CFA demosaicing
        cfa_inter = Demosaic(
            wb_raw, self.platform, self.sensor_info, self.parm_dem, self.save_output_obj
        )
        demos_img = cfa_inter.execute()

        # =====================================================================
        # Color correction matrix
        ccm = CCM(
            demos_img,
            self.platform,
            self.sensor_info,
            self.parm_ccm,
            self.save_output_obj,
        )
        ccm_img = ccm.execute()

        # =====================================================================
        # Gamma
        gmc = GC(
            ccm_img,
            self.platform,
            self.sensor_info,
            self.parm_gmc,
            self.save_output_obj,
        )
        gamma_raw = gmc.execute()

        # =====================================================================
        # Auto-Exposure
        aef = AE(gamma_raw, self.sensor_info, self.parm_ae)
        self.ae_feedback = aef.execute()

        # =====================================================================
        # Color space conversion
        csc = CSC(
            gamma_raw,
            self.platform,
            self.sensor_info,
            self.parm_csc,
            self.save_output_obj,
        )
        csc_img = csc.execute()

        # =====================================================================
        # Sharpening
        sha = SHARP(
            csc_img,
            self.platform,
            self.sensor_info,
            self.parm_sha,
            self.save_output_obj,
        )
        sha_img = sha.execute()
        
        # =====================================================================
        # 2d noise reduction
        nr2d = NR2D(
            sha_img,
            self.sensor_info,
            self.parm_2dn,
            self.platform,
            self.save_output_obj,
        )
        nr2d_img = nr2d.execute()

        # =====================================================================
        # RGB conversion
        rgbc = RGBC(
            nr2d_img,
            self.platform,
            self.sensor_info,
            self.parm_rgb,
            self.parm_csc,
            self.save_output_obj,
        )
        rgbc_img = rgbc.execute()

        # =====================================================================
        # crop image to 1920x1080 or 1920x1440
        irc = IRC(
            rgbc_img,
            self.platform,
            self.sensor_info,
            self.parm_irc,
            self.save_output_obj,
        )
        irc_img = irc.execute()

        # =====================================================================
        # on-screen display for 10xEngineers logo
        osd = OSD(
            irc_img,
            self.platform,
            self.sensor_info,
            self.parm_osd,
            self.save_output_obj,
        )
        osd_img = osd.execute()

        # =====================================================================
        # Scaling
        scale = Scale(
            osd_img,
            self.platform,
            self.sensor_info,
            self.parm_sca,
            self.save_output_obj,
        )
        scaled_img = scale.execute()

        # =====================================================================
        # YUV saving format 444, 422 etc
        yuv = YUV_C(
            scaled_img,
            self.platform,
            self.sensor_info,
            self.parm_yuv,
            self.save_output_obj,
        )
        yuv_conv = yuv.execute()

        out_img = yuv_conv  # original Output of ISP
        out_dim = scaled_img.shape  # dimensions of Output Image

        # ======================================================================
        # Is not part of ISP-pipeline only assists in visualizing output results
        if visualize_output:
            # There can be two out_img formats depending upon which modules are
            # enabled 1. YUV    2. RGB

            if self.parm_yuv["is_enable"] is True:
                # YUV_C is enabled and RGB_C is disabled: Output is compressed YUV
                # To display : Need to decompress it and convert it to RGB.
                image_height, image_width, _ = out_dim
                yuv_custom_format = self.parm_yuv["conv_type"]

                yuv_conv = util.get_image_from_yuv_format_conversion(
                    yuv_conv, image_height, image_width, yuv_custom_format
                )

                rgbc.yuv_img = yuv_conv
                out_rgb = rgbc.yuv_to_rgb()

            # elif self.parm_rgb["is_enable"] is False:
            #     # RGB_C is disabled: Output is 3D - YUV
            #     # To display : Only convert it to RGB
            #     rgbc.yuv_img = yuv_conv
            #     out_rgb = rgbc.yuv_to_rgb()

            # else:
                # RGB_C is enabled: Output is RGB
                # no further processing is needed for display
            else:
                out_rgb = out_img

            # If both RGB_C and YUV_C are enabled. Infinite-ISP will generate
            # an output but it will be an invalid image.

            self.save_output_obj.save_pipeline_output(
                self.out_file, out_rgb, self.c_yaml
            )

    def execute(self, img_path=None):
        """
        Start execution of Infinite-ISP
        """
        if img_path is not None:
            self.raw_file = img_path
            self.c_yaml["platform"]["filename"] = self.raw_file

        self.load_raw()

        # Print Logs to mark start of pipeline Execution
        print(50 * "-" + "\nLoading RAW Image Done......\n")
        print("Filename: ", self.in_file)

        # Note Initial Time for Pipeline Execution
        start = time.time()

        if not self.render_3a:
            # Run ISP-Pipeline once
            self.run_pipeline(visualize_output=True)
            # Display 3A Statistics
        else:
            # Run ISP-Pipeline till Correct Exposure with AWB gains
            self.execute_with_3a_statistics()

        util.display_ae_statistics(self.ae_feedback, self.awb_gains)

        # Print Logs to mark end of pipeline Execution
        print(50 * "-" + "\n")

        # Calculate pipeline execution time
        print(f"\nPipeline Elapsed Time: {time.time() - start:.3f}s")

    def load_3a_statistics(self, awb_on=True, ae_on=True):
        """
        Update 3A Stats into WB and DG modules parameters
        """
        # Update 3A in c_yaml too because it is output config
        if awb_on is True and self.parm_wbc["is_auto"] and self.parm_awb["is_enable"]:
            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"]["r_gain"] = float(
                self.awb_gains[0]
            )
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"]["b_gain"] = float(
                self.awb_gains[1]
            )
        if ae_on is True and self.parm_dga["is_auto"] and self.parm_ae["is_enable"]:
            self.parm_dga["ae_feedback"] = self.c_yaml["digital_gain"][
                "ae_feedback"
            ] = self.ae_feedback
            self.parm_dga["current_gain"] = self.c_yaml["digital_gain"][
                "current_gain"
            ] = self.dga_current_gain

    def execute_with_3a_statistics(self):
        """
        Execute Infinite-ISP with AWB gains and correct exposure
        """

        # Maximum Iterations depend on total permissible gains
        max_dg = len(self.parm_dga["gain_array"])

        # Run ISP-Pipeline
        self.run_pipeline(visualize_output=False)
        self.load_3a_statistics()
        while not (
            (self.ae_feedback == 0)
            or (self.ae_feedback == -1 and self.dga_current_gain == max_dg)
            or (self.ae_feedback == 1 and self.dga_current_gain == 0)
            or self.ae_feedback is None
        ):
            self.run_pipeline(visualize_output=False)
            self.load_3a_statistics()

        self.run_pipeline(visualize_output=True)

    def update_sensor_info(self, sensor_info, update_blc_wb=False):
        """
        Update sensor_info in config files
        """
        self.sensor_info["width"] = self.c_yaml["sensor_info"]["width"] = sensor_info[0]

        self.sensor_info["height"] = self.c_yaml["sensor_info"]["height"] = sensor_info[
            1
        ]

        self.sensor_info["bit_depth"] = self.c_yaml["sensor_info"][
            "bit_depth"
        ] = sensor_info[2]

        self.sensor_info["bayer_pattern"] = self.c_yaml["sensor_info"][
            "bayer_pattern"
        ] = sensor_info[3]

        if update_blc_wb:
            self.parm_blc["r_offset"] = self.c_yaml["black_level_correction"][
                "r_offset"
            ] = sensor_info[4][0]
            self.parm_blc["gr_offset"] = self.c_yaml["black_level_correction"][
                "gr_offset"
            ] = sensor_info[4][1]
            self.parm_blc["gb_offset"] = self.c_yaml["black_level_correction"][
                "gb_offset"
            ] = sensor_info[4][2]
            self.parm_blc["b_offset"] = self.c_yaml["black_level_correction"][
                "b_offset"
            ] = sensor_info[4][3]

            self.parm_blc["r_sat"] = self.c_yaml["black_level_correction"][
                "r_sat"
            ] = sensor_info[5]
            self.parm_blc["gr_sat"] = self.c_yaml["black_level_correction"][
                "gr_sat"
            ] = sensor_info[5]
            self.parm_blc["gb_sat"] = self.c_yaml["black_level_correction"][
                "gb_sat"
            ] = sensor_info[5]
            self.parm_blc["b_sat"] = self.c_yaml["black_level_correction"][
                "b_sat"
            ] = sensor_info[5]

            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"][
                "r_gain"
            ] = sensor_info[6][0]
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"][
                "b_gain"
            ] = sensor_info[6][2]

    def set_save_paths(self, new_path):
        """Set paths to save module/pipeline output"""
        pipeline_outpath = new_path / "out_frames"
        module_outpath = new_path
        config_outpath = new_path / "config"

        self.save_output_obj.reset_outpaths(
            pipeline_outpath, module_outpath, config_outpath
        )

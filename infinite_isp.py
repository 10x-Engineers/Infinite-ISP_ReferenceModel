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
from modules.yuv_conv_format import YUVConvFormat as YUV_C
from modules.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion import RGBConversion as RGBC
from modules.invalid_region_crop import InvalidRegionCrop as IRC
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
        self.bit_range = self.sensor_info["range"]
        self.bayer = self.sensor_info["bayer_pattern"]
        self.width = self.sensor_info["width"]
        self.height = self.sensor_info["height"]
        self.bpp = self.sensor_info["bit_depth"]
        self.rev_yuv = self.platform["rev_yuv_channels"]

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
        self.parm_2dn = c_yaml["2d_noise_reduction"]
        self.parm_rgb = c_yaml["rgb_conversion"]
        self.parm_irc = c_yaml["invalid_region_crop"]
        self.parm_sca = c_yaml["scale"]
        self.parm_yuv = c_yaml["yuv_conversion_format"]
        self.c_yaml = c_yaml

    def load_raw(self):
        """
        Load raw image from provided path
        """
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        self.out_file = "Out_" + self.in_file

        # Load Raw
        if path_object.suffix == ".raw":
            if self.bpp > 8:
                self.raw = np.fromfile(raw_path, dtype=np.uint16).reshape(
                    (self.height, self.width)
                )
            else:
                self.raw = np.fromfile(raw_path, dtype=np.uint8).reshape(
                    (self.height, self.width)
                ).astype(np.uint16)
        else:
            img = rawpy.imread(raw_path)
            self.raw = img.raw_image

    def run_pipeline(self, visualize_output):
        """
        Run ISP-Pipeline for a raw-input image
        """
        # save pipeline input array
        if self.parm_cro["is_save"]:
            util.save_output_array(
                self.in_file, self.raw, "Inpipeline_crop_", self.platform, self.bpp
            )

        # =====================================================================
        # Cropping
        crop = Crop(self.raw, self.sensor_info, self.parm_cro)
        cropped_img = crop.execute()

        # save module output if enabled
        if self.parm_cro["is_save"]:
            util.save_output_array(
                self.in_file, cropped_img, "Out_crop_", self.platform, self.bpp
            )

        # =====================================================================

        #  Dead pixels correction
        dpc = DPC(cropped_img, self.sensor_info, self.parm_dpc, self.platform)
        dpc_raw = dpc.execute()

        # save module output if enabled
        if self.parm_dpc["is_save"]:
            util.save_output_array(
                self.in_file, dpc_raw, "Out_dead_pixel_correction_", self.platform, self.bpp
            )

        # =====================================================================

        # Black level correction
        blc = BLC(dpc_raw, self.sensor_info, self.parm_blc)
        blc_raw = blc.execute()

        # save module output if enabled
        if self.parm_blc["is_save"]:
            util.save_output_array(
                self.in_file, blc_raw, "Out_black_level_correction_", self.platform, self.bpp
            )

        # =====================================================================

        # OECF
        oecf = OECF(blc_raw, self.sensor_info, self.parm_oec)
        oecf_raw = oecf.execute()

        # save module output if enabled
        if self.parm_oec["is_save"]:
            util.save_output_array(self.in_file, oecf_raw, "Out_oecf_", self.platform, self.bpp)

        # =====================================================================
        # Digital Gain
        dga = DG(oecf_raw, self.sensor_info, self.parm_dga)
        dga_raw, self.dga_current_gain = dga.execute()

        # save module output if enabled
        if self.parm_dga["is_save"]:
            util.save_output_array(
                self.in_file, dga_raw, "Out_digital_gain_", self.platform, self.bpp
            )

        # =====================================================================
        # Bayer noise reduction
        bnr = BNR(dga_raw, self.sensor_info, self.parm_bnr, self.platform)
        bnr_raw = bnr.execute()

        # save module output if enabled
        if self.parm_bnr["is_save"]:
            util.save_output_array(
                self.in_file, bnr_raw, "Out_bayer_noise_reduction_", self.platform, self.bpp
            )

        # =====================================================================
        # Auto White Balance
        awb = AWB(bnr_raw, self.sensor_info, self.parm_awb)
        self.awb_gains = awb.execute()

        # =====================================================================
        # White balancing
        wbc = WB(bnr_raw, self.sensor_info, self.parm_wbc)
        wb_raw = wbc.execute()

        # save module output if enabled
        if self.parm_wbc["is_save"]:
            util.save_output_array(
                self.in_file, wb_raw, "Out_white_balance_", self.platform, self.bpp
            )

        # =====================================================================
        # CFA demosaicing
        cfa_inter = Demosaic(wb_raw, self.sensor_info)
        demos_img = cfa_inter.execute()

        # save module output if enabled
        if self.parm_dem["is_save"]:
            util.save_output_array(
                self.in_file, demos_img, "Out_demosaic_", self.platform, self.bpp
            )

        # =====================================================================
        # Color correction matrix
        ccm = CCM(demos_img, self.sensor_info, self.parm_ccm)
        ccm_img = ccm.execute()

        # save module output if enabled
        if self.parm_ccm["is_save"]:
            util.save_output_array(
                self.in_file, ccm_img, "Out_color_correction_matrix_", self.platform, self.bpp
            )

        # =====================================================================
        # Gamma
        gmc = GC(ccm_img, self.sensor_info, self.parm_gmc)
        gamma_raw = gmc.execute()
        # save module output if enabled
        if self.parm_gmc["is_save"]:
            util.save_output_array(
                self.in_file, gamma_raw, "Out_gamma_correction_", self.platform, self.bpp
            )

        # =====================================================================
        # Auto-Exposure
        aef = AE(gamma_raw, self.sensor_info, self.parm_ae)
        self.ae_feedback = aef.execute()

        # =====================================================================
        # Color space conversion
        csc = CSC(gamma_raw, self.sensor_info, self.parm_csc)
        csc_img = csc.execute()
        # save module output if enabled
        if self.parm_csc["is_save"]:
            util.save_output_array_yuv(
                self.in_file,
                csc_img,
                "Out_color_space_conversion_",
                self.rev_yuv,
                self.platform,
            )

        # =====================================================================
        # 2d noise reduction
        nr2d = NR2D(csc_img, self.sensor_info, self.parm_2dn, self.platform)
        nr2d_img = nr2d.execute()
        # save module output if enabled
        if self.parm_2dn["is_save"]:
            util.save_output_array_yuv(
                self.in_file,
                nr2d_img,
                "Out_2d_noise_reduction_",
                self.rev_yuv,
                self.platform,
            )

        # =====================================================================
        # RGB conversion
        rgbc = RGBC(nr2d_img, self.sensor_info, self.parm_rgb, self.parm_csc)
        rgbc_img = rgbc.execute()
        # save module output if enabled
        if self.parm_rgb["is_save"]:
            util.save_output_array(
                self.in_file, rgbc_img, "Out_rgb_conversion_", self.platform, self.bpp
            )

        # np.save("output.npy", rgbc_img.astype(np.uint16))

        # =====================================================================
        # crop image to 1920x1080 or 1920x1440
        irc = IRC(rgbc_img, self.parm_irc)
        irc_img = irc.execute()
        # save module output if enabled
        if self.parm_irc["is_save"]:
            util.save_output_array_yuv(
                self.in_file,
                irc_img,
                "Out_invalid_region_crop_",
                self.rev_yuv,
                self.platform,
            )

        # =====================================================================
        # Scaling
        scale = Scale(irc_img, self.sensor_info, self.parm_sca)
        scaled_img = scale.execute()
        # save module output if enabled
        if self.parm_sca["is_save"]:
            util.save_output_array_yuv(
                self.in_file, scaled_img, "Out_scale_", self.rev_yuv, self.platform
            )

        # =====================================================================
        # YUV saving format 444, 422 etc
        yuv = YUV_C(
            scaled_img, self.sensor_info, self.parm_yuv, self.in_file
        )  # parm_csc)
        yuv_conv = yuv.execute()

        # save module output if enabled
        if self.parm_yuv["is_save"]:
            self.platform["save_format"] = "npy"
            util.save_output_array(
                self.in_file,
                yuv_conv,
                f"Out_yuv_conversion_format_{self.parm_yuv['conv_type']}_",
                self.platform, self.bpp
            )
            self.platform["save_format"] = self.c_yaml["platform"]["save_format"]

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

            elif self.parm_rgb["is_enable"] is False:

                # RGB_C is disabled: Output is 3D - YUV
                # To display : Only convert it to RGB
                rgbc.yuv_img = yuv_conv
                out_rgb = rgbc.yuv_to_rgb()

            else:
                # RGB_C is enabled: Output is RGB
                # no further processing is needed for display
                out_rgb = out_img

            # If both RGB_C and YUV_C are enabled. Infinite-ISP will generate
            # an output but it will be an invalid image.

            util.save_pipeline_output(
                self.out_file, out_rgb, self.c_yaml, self.platform["generate_tv"]
            )

    def execute(self, img_path=None, visualize_output=True):
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
            self.run_pipeline(visualize_output)
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
            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"][
                "r_gain"
            ] = self.awb_gains[0]
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"][
                "b_gain"
            ] = self.awb_gains[1]
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
        ):
            self.run_pipeline(visualize_output=False)
            self.load_3a_statistics()

        self.run_pipeline(visualize_output=True)

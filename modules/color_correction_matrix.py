"""
File: color_correction_matrix.py
Description: Applies the 3x3 correction matrix on the image
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np


class ColorCorrectionMatrix:
    "Apply the color correction 3x3 matrix"

    def __init__(self, img, platform, sensor_info, parm_ccm, save_out_obj):
        self.img = img.copy()
        self.enable = parm_ccm["is_enable"]
        self.is_save = parm_ccm["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        self.bit_depth = sensor_info["bit_depth"]
        self.ccm_mat = None
        self.save_out_obj = save_out_obj

    def apply_ccm(self):
        """
        Apply CCM Params
        """
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])

        self.ccm_mat = np.int16([r_1, r_2, r_3])

        # # normalize nbit to 0-1 img
        # self.img = np.float32(self.img) / (2**self.bit_depth - 1)

        # convert to nx3
        img1 = self.img.reshape(((self.img.shape[0] * self.img.shape[1], 3)))

        # keeping imatest convention of colum sum to 1 mat. O*A => A = ccm
        out = np.matmul(img1, self.ccm_mat.transpose()) / 1024

        # clipping after ccm is must to eliminate neg values
        out = np.uint16(np.clip(out, 0, 2**self.bit_depth - 1))

        # convert back
        out = out.reshape(self.img.shape)
        # out = np.uint16(out * (2**self.bit_depth - 1))
        return out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            self.save_out_obj.save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_",
                self.platform,
                self.bit_depth,
            )

    def execute(self):
        """Execute ccm if enabled."""
        print("Color Correction Matrix = " + str(self.enable))

        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img

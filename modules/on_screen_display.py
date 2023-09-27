"""
File: oecf.py
Description: Implements the opto electronic conversion function as a LUT
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import save_output_array


class OnScreenDisplay:
    "On-Screen Display for 10xEngineers Logo"

    def __init__(self, img, platform, sensor_info, parm_osd, parm_csc):
        self.img = img.copy()
        self.enable = parm_osd["is_enable"]
        self.is_save = parm_osd["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_osd = parm_osd
        self.x_offset = parm_osd["x_offset"]
        self.y_offset = parm_osd["y_offset"]
        self.rgb_osd = self.platform["rgb_output"]
        self.csc_conv = parm_csc["conv_standard"]

    def apply_osd(self):
        """Apply OSD."""
        logo = np.load("./docs/assets/osd-logo.npy")
        height, width = logo.shape

        if self.rgb_osd:

            vectorized_g = np.vectorize(lambda x:90 if x == 0 else x)
            vectorized_b = np.vectorize(lambda x:160 if x == 0 else x)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 0] = logo
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 1] = vectorized_g(logo)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 2] = vectorized_b(logo)
        elif self.csc_conv == 1:
            vectorized_y = np.vectorize(lambda x:81 if x == 0 else 235)
            vectorized_u = np.vectorize(lambda x:168 if x == 0 else 128)
            vectorized_v = np.vectorize(lambda x:86 if x == 0 else 128)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 0] = vectorized_y(logo)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 1] = vectorized_u(logo)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 2] = vectorized_v(logo)
        else:
            vectorized_y = np.vectorize(lambda x:87 if x == 0 else 255)
            vectorized_u = np.vectorize(lambda x:86 if x == 0 else 128)
            vectorized_v = np.vectorize(lambda x:184 if x == 0 else 128)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 0] = vectorized_y(logo)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 1] = vectorized_u(logo)
            self.img[self.x_offset: self.x_offset + height, self.y_offset: self.y_offset + width, 2] = vectorized_v(logo)


        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_osd_",
                self.platform,
                self.sensor_info["bit_depth"],
            )

    def execute(self):
        """Execute OECF if enabled."""
        print("On-screen Display Function = " + str(self.enable))

        if self.enable:
            start = time.time()
            oecf_out = self.apply_osd()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = oecf_out
        self.save()
        return self.img

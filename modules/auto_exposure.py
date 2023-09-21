"""
File: auto_exposure.py
Description: 3A-AE Runs the Auto exposure algorithm in a loop
Code / Paper  Reference: https://www.atlantis-press.com/article/25875811.pdf
                         http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Probability%20and%20statistics/CRC%20-%20standard%20probability%20and%20Statistics%20tables%20and%20formulae%20-%20DANIEL%20ZWILLINGER.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import get_approximate, approx_sqrt


class AutoExposure:
    """
    Auto Exposure Module
    """

    def __init__(self, img, sensor_info, parm_ae):
        self.img = img
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.stats_window_offset = np.array(parm_ae["stats_window_offset"])
        self.center_illuminance = parm_ae["center_illuminance"]
        self.histogram_skewness_range = parm_ae["histogram_skewness"]
        self.sensor_info = sensor_info
        self.param_ae = parm_ae
        self.bit_depth = sensor_info["bit_depth"]

        # Pipeline modules included in AE Feedback Loop
        # (White Balance) wb module is renamed to wbc (white balance correction)
        # gc (Gamma Correction) module is renamed to gcm (Gamma Correction Module)

    def get_exposure_feedback(self):
        """
        Get Correct Exposure by Adjusting Digital Gain
        """
        # Convert Image into 8-bit for AE Calculation
        self.img = self.img >> (self.bit_depth - 8)
        self.bit_depth = 8

        self.img = self.apply_window_offset_crop()

        # calculate the exposure metric
        return self.determine_exposure()

    def apply_window_offset_crop(self):
        """
        Get AE Stats window by cropping the offsets
        """
        offsets = self.stats_window_offset
        top = int(offsets[0])
        bottom = None if offsets[1] == 0 else -int(offsets[1])
        left = int(offsets[2])
        right = None if offsets[3] == 0 else -int(offsets[3])
        return self.img[top:bottom, left:right, :]

    def determine_exposure(self):
        """
        Image Exposure Estimation using Skewness Luminance of Histograms
        """

        # plt.imshow(self.img)
        # plt.show()

        # For Luminance Histograms, Image is first converted into greyscale image
        # Function also returns average luminance of image which is used as AE-Stat
        grey_img, avg_lum = self.get_greyscale_image(self.img)
        print("Average luminance is = ", avg_lum)

        # Histogram skewness Calculation for AE Stats
        skewness = self.get_luminance_histogram_skewness(grey_img)

        # get the ranges
        upper_limit, _ = get_approximate(self.histogram_skewness_range, 16, 8)
        lower_limit = -1 * upper_limit

        if self.is_debug:
            print("   - AE - Histogram Skewness Range = ", upper_limit)

        # see if skewness is within range
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        else:
            return 0

    def get_greyscale_image(self, img):
        """
        Conversion of an Image into Greyscale Image
        """
        # Each RGB pixels with [77, 150, 29] to get its luminance

        grey_img_int = np.clip(
            (np.dot(img[..., :3], [77, 150, 29]) / 256), 0, (2**self.bit_depth)
        ).astype(np.uint16)

        return grey_img_int, np.average(grey_img_int, axis=(0, 1))

    def get_luminance_histogram_skewness(self, img):
        """
        Skewness Calculation in reference to:
        Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics
        Tables and Formulae. Chapman & Hall: New York. 2000. Section 2.2.24.1
        """
        img_orig = np.copy(img)
        # First subtract central luminance to calculate skewness around it
        img = img.astype(np.float64) - self.center_illuminance

        # The sample skewness is computed as the Fisher-Pearson coefficient of
        # skewness, i.e. (m_3 / m_2**(3/2)) * g_1
        # where m_2 is 2nd moment (variance) and m_3 is third moment skewness
        # img = img.astype(np.float64) - np.average(img)
        img_size = img.size
        m_2 = (np.sum(np.power(img, 2)) / img_size).astype(int)
        m_3 = (np.sum(np.power(img, 3)) / img_size).astype(int)

        skewness = np.nan_to_num(m_3 / abs(m_2) ** (3 / 2))
        if self.is_debug:
            print("   - AE - Actual_Skewness = ", skewness)

        # all integer calc
        img_int = img_orig.astype(np.int64) - self.center_illuminance
        img_int_size = img_int.size
        m_2_int = np.sum(np.power(img_int, 2)).astype(np.int64)
        m_3_int = np.sum(np.power(img_int, 3)).astype(np.int64)
        m_2_int = np.int64(m_2_int / img_int_size)
        m_3_int = np.int64(m_3_int / img_int_size)
        sign_m3_int = np.sign(m_3_int)
        # all integer calc

        m_2_int = m_2_int >> 6
        m_3_int = abs(m_3_int) >> 9

        approx_sqrt_m_2_int = approx_sqrt(m_2_int)
        new_skewness_int = (
            np.int64((m_3_int * 256) / (m_2_int * approx_sqrt_m_2_int)) / 256
        )
        # new_skewness_int = sign_m3_int * new_skewness_int
        if self.is_debug:
            print("   - AE - Approx_Skewness Int = ", new_skewness_int)

        return new_skewness_int

    def execute(self):
        """
        Execute Auto Exposure
        """
        print("Auto Exposure= " + str(self.enable))

        if self.enable is False:
            return None
        else:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            print(f"  Execution time: {time.time()-start:.3f}s")
            return ae_feedback

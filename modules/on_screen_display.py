"""
File: osd.py
Description: Puts a logo on the final pipeline image
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
from pathlib import Path
import numpy as np


class OnScreenDisplay:
    """
    This class implements the functionality for adding an on-screen display (OSD)
    to an image. Specifically, it is designed to overlay a 10xEngineers logo onto the image.
    The class allows customization of the logo's position and colors,
    and supports alpha blending for transparency effects.
    """

    def __init__(self, img, platform, sensor_info, parm_osd, save_out_obj):
        """
        Initializes the OnScreenDisplay instance with the image to be processed,
        platform information, sensor details, and OSD parameters.
        """
        # Image and configuration settings
        self.img = img.copy()  # Copy of the image to avoid modifying the original
        self.enable = parm_osd["is_enable"]  # Flag to enable/disable OSD
        self.is_save = parm_osd["is_save"]  # Flag to enable/disable saving the output
        self.platform = platform  # Platform information
        self.sensor_info = sensor_info  # Sensor information
        self.parm_osd = parm_osd  # OSD parameters
        self.save_out_obj = save_out_obj

        # OSD position and appearance settings
        self.x_offset = parm_osd["x_offset"]  # Horizontal position offset for the logo
        self.y_offset = parm_osd["y_offset"]  # Vertical position offset for the logo
        self.width = parm_osd["width"]  # Width of the logo
        self.height = parm_osd["height"]  # Height of the logo
        self.bg_color = parm_osd["bg_color"]  # Background color of the logo
        self.fg_color = parm_osd["fg_color"]  # Foreground color of the logo
        self.alpha = parm_osd["alpha"]  # Alpha value for transparency

        # Maximum allowed dimensions for the logo
        self.maximum_height = 128
        self.maximum_width = 256

    def get_logo_numpy(self):
        """
        Reads a logo from a specified text file and converts it into a NumPy array.
        The text file should contain the logo in a bitmap format represented by hexadecimal values.
        Each line in the file represents a row of the logo.
        """

        file_path = "./docs/assets/10xLogo.txt"

        if not Path(file_path).exists():
            # search recursively in the subdirectories
            file_path = list(Path.cwd().rglob("*Logo.txt"))[0]

        with open(file_path, "r") as file:
            img = []
            for line in file:
                # Split the line by commas
                # Convert each hexadecimal value in the line into a binary string
                hex_numbers = line.strip().split(",")[:-1]
                row = ""
                for hex_num in hex_numbers:
                    row = row + bin(int(hex_num, 16))[2:].zfill(32)
                img.append(np.array([int(bit) for bit in row]))
            return np.array(img)

    def blend_pixels(self, foreground_pixel, background_pixel, alpha):
        """
        Performs alpha blending on a pair of pixels.
        This function blends a foreground pixel with a background pixel based on the alpha value.
        Alpha blending is used to combine the logo and the image with transparency.
        """
        blended_pixel = [
            (int(alpha * f + (255 - alpha) * b) >> 8)
            for f, b in zip(foreground_pixel, background_pixel)
        ]
        return blended_pixel

    def alpha_blend(self, foreground_image, background_image, alpha):
        """
        Applies alpha blending to overlay a foreground image (logo) onto a background image.
        Assumes that both images are of the same dimensions and each pixel is in RGB format.
        The alpha parameter determines the transparency level of the foreground image.
        """
        # Assuming both images have the same dimensions and each pixel is in RGB format
        blended_image = []
        for fg_row, bg_row in zip(foreground_image, background_image):
            blended_row = []
            for fg_pixel, bg_pixel in zip(fg_row, bg_row):
                blended_row.append(self.blend_pixels(fg_pixel, bg_pixel, alpha))
            blended_image.append(blended_row)
        return blended_image

    def apply_osd(self):
        """
        Applies the On-Screen Display (OSD) to the image.
        It first retrieves the logo, checks size constraints, and
        then overlays the logo onto the image.
        The logo is positioned based on specified offsets and is
        blended using the alpha_blend method.
        """

        logo = self.get_logo_numpy()

        height, width = logo.shape
        # Check size constraints and return original image if constraints are not met
        if (
            height + self.y_offset > self.sensor_info["height"]
            or width + self.x_offset > self.sensor_info["width"]
        ):
            return self.img

        if height > max(self.height, self.maximum_height) or width > max(
            self.width, self.maximum_width
        ):
            return self.img

        logo_img = np.empty((self.height, self.width, 3))

        # Create a colorized version of the logo and apply alpha blending
        logo_img[:, :, 0] = np.vectorize(
            lambda x: self.bg_color[0] if x == 0 else self.fg_color[0]
        )(logo)
        logo_img[:, :, 1] = np.vectorize(
            lambda x: self.bg_color[1] if x == 0 else self.fg_color[1]
        )(logo)
        logo_img[:, :, 2] = np.vectorize(
            lambda x: self.bg_color[2] if x == 0 else self.fg_color[2]
        )(logo)

        transparent_logo = self.alpha_blend(
            logo_img,
            self.img[
                self.y_offset : self.y_offset + height,
                self.x_offset : self.x_offset + width,
                :,
            ],
            self.alpha,
        )

        # Overlay the logo on the original image at the specified position
        self.img[
            self.y_offset : self.y_offset + height,
            self.x_offset : self.x_offset + width,
            :,
        ] = transparent_logo

        return self.img

    def save(self):
        """
        Saves the output image with the applied OSD.
        The function is called only if the saving option is enabled.
        It utilizes utility functions for saving, handling file naming and format.
        """
        if self.is_save:
            if self.platform["rgb_output"]:
                self.save_out_obj.save_output_array(
                    self.platform["in_file"],
                    self.img,
                    "Out_on_screen_display_",
                    self.platform,
                    self.sensor_info["bit_depth"],
                )
            else:
                self.save_out_obj.save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_on_screen_display_",
                    self.platform,
                )

    def execute(self):
        """
        Executes the OSD application process.
        It first checks if OSD is enabled, applies the OSD to the image,
        measures the execution time, and saves the output if required.
        This is the main method that orchestrates the OSD mdoules from infinite_isp.
        """
        print("On-screen Display Function = " + str(self.enable))

        if self.enable:
            start = time.time()
            oecf_out = self.apply_osd()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = oecf_out
        self.save()
        return self.img

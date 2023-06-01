# User Guide


You can run the project by simply executing the [isp_pipeline.py](../isp_pipeline.py). This is the main file that loads all the algorithmic parameters from the [configs.yml](../config/configs.yml)
The config file contains tags for each module implemented in the pipeline. A detailed documentation of implemented algorithms is provided [here](algorithm-description.pdf). Whereas, brief description of configuration parameters is as follows:

### Platform

Platform contains configuration parameters that are not part of the ISP pipeline but helps in pipeline execution and debugging:

| platform            | Details | 
| -----------         | --- |
| filename            | Specifies the file name for running the pipeline. The file should be placed in the `RAW_PATH` or `DATASET_PATH` mentioned in the scripts. |
| disable_progress_bar| Enables or disables the progress bar for time taking modules|
| leave_pbar_string   |  Hides or unhides the progress bar upon completion |
| save_lut            | Flag to store LUT files for 2DNR and BNR |
| generate_tv         | Indicates that ISP pipeline is running through [automate_execution.py](../test_vector_generation/automate_execution.py) for debugging instead of [isp_pipeline.py](../isp_pipeline.py)|
|rev_yuv_channels| Use for debugging - Generate YUV test vectors with reversed channels (VUY)|
|save_format| Use for Debugging - Set module output format <br> - `npy` <br> - `png` <br> - `both` |
|rendered_3a| Returns 3a rendered final image with awb gains and correct exposure|

### Sensor_info

Sensor specifications used by each module in the ISP-pipeline.

| sensor Info   | Details | 
| -----------   | --- |
| bayer_pattern | Specifies the bayer patter of the RAW image in lowercase letters <br> - `bggr` <br> - `rgbg` <br> - `rggb` <br> - `grbg`|
| range         | Saturation level of the sensor |
| bitdep        | The bit depth of the raw image |
| width         | The width of the input raw image |
| height        | The height of the input raw image |

### Debugging Parameters

Below parameters are present each ISP pipeline module they effect the functionality but helps in debugging the module.

| parameters   | Details | 
| -----------   | --- |
| is_debug  | Flag to output module debug logs|
|is_save    | Saves module output according to the format defined in `platform.save_format` |



### Crop

| crop          | Details | 
| -----------   | --- |
| is_enable      |  Enables or disables this module. When enabled it only crops if bayer pattern is not disturbed
| new_width     |  New width of the input RAW image after cropping
| new_height    |  New height of the input RAW image after cropping

### Dead Pixel Correction 

| dead_pixel_correction | Details |
| -----------           |   ---   |
| is_enable              |  Enables or disables this module
| dp_threshold          |  The threshold for tuning the dpc module. The lower the threshold more are the chances of pixels being detected as dead and hence corrected  

### Black Level Correction 

| black_level_correction  | Details |
| -----------             |   ---   |
| is_enable                |  Enables or disables this module
| r_offset                |  Red channel offset
| gr_offset               |  Gr channel offset
| gb_offset               |  Gb channel offset
| b_offset                |  Blue channel offset
| is_linear                |  Enables or disables linearization. When enabled the BLC offset maps to zero and saturation maps to the highest possible bit range given by the user  
| r_sat                   | Red channel saturation level  
| gr_sat                  |  Gr channel saturation level
| gb_sat                  |  Gb channel saturation level
| b_sat                   |  Blue channel saturation level

### Opto-Electronic Conversion Function 

| OECF  | Details |
| -----------     |   ---   |
| is_enable        | Enables or disables this module
| r_lut           | The look up table for oecf curve. This curve is mostly sensor dependent and is found by calibration using some standard technique 

### Digital Gain

| digital_gain    | Details |
| -----------     |   ---   |
| is_enable       | This is a essential module and cannot be disabled 
| is_auto         | Flag to calculated digital gain using AE Feedback
| gain_array      | Gains array. List of permissible digital gains |
| current_gain    | Index for the current gain in gain_array. It starts from zero |
| ae_feedback| AE feedback, it has only following values <br> - `1` (Overexposed)  <br> - `-1` (Underexposed)  <br> - `0` (Correct Exposure) |

### Bayer Noise Reduction

| bayer_noise_reduction   | Details |
| -----------             |   ---   |
| is_enable                | When enabled reduces the noise in bayer domain using the user given parameters |
| filter_window             | Filter window <br>Should be an odd window size |
| r_std_dev_s               | Red channel gaussian kernel strength. The more the strength the more the blurring. Cannot be zero  
| r_std_dev_r               | Red channel range kernel strength. The more the strength the more the edges are preserved. Cannot be zero
| g_std_dev_s               | Gr and Gb gaussian kernel strength
| g_std_dev_r               | Gr and Gb range kernel strength
| b_std_dev_s               | Blue channel gaussian kernel strength
| b_std_dev_r               | Blue channel range kernel strength

### 3A - Auto White Balance (AWB)
| auto_white_balance      | Details |
| -----------             |   ---   |
| is_enable           | When enabled calculates white balance gains for current frame  |
| stats_window_offset | Specifies the crop dimensions to obtain a stats calculation window <br> - Should be an array of elements `[Up, Down, Left Right]` <br> - Should be a multiple of 4 |
| underexposed_pecentage   | Set % of dark pixels to exclude before AWB gain calculation|
| overexposed_pecentage    | Set % of saturated pixels to exclude before AWB gain calculation|

### White balance

| white_balance           | Details |
| -----------             |   ---   |
| is_enable               | Applies white balance gains when enabled |
| is_auto                 | Flag to apply AWB gains|
| r_gain                  | Red channel gain  |
| b_gain                  | Blue channel gain |




### Color Correction Matrix (CCM)

| color_correction_matrix                 | Details |
| -----------                             |   ---   |
| is_enable                                | When enabled applies the user given 3x3 CCM to the 3D RGB image having rows sum to 1 convention  |
| corrected_red                           | Row 1 of CCM
| corrected_green                         | Row 2 of CCM
| corrected_blue                          | Row 3 of CCM

### Gamma Correction
| gamma_correction        | Details |
| -----------             |   ---   |
| is_enable                | When enabled  applies tone mapping gamma using the LUT  |
| gamma_lut_8               | The look up table for gamma curve for 8 bit Image |
| gamma_lut_10              | The look up table for gamma curve for 10 bit Image |
| gamma_lut_12              | The look up table for gamma curve for 12 bit Image |
| gamma_lut_14              | The look up table for gamma curve for 14 bit Image |

### 3A - Auto Exposure
| auto_exposure      | Details                                                                                      
|--------------------|----------------------------------------------------------------------------------------------|
| is_enable           | When enabled applies the 3A- Auto Exposure algorithm                                         |
| stats_window_offset | Specifies the crop dimensions to obtain a stats calculation window <br> - Should be an array of elements `[Up, Down, Left, Right]` <br> - Should be a multiple of 4 |                                                            |  
| center_illuminance | The value of center illuminance for skewness calculation ranges from 0 to 255. Default is 90 |   
| histogram_skewness | The range of histogram skewness should be between 0 and 1 for correct exposure calculation   |  

### Color Space Conversion (CSC)

| color_space_conversion | Details                                                                             |  
|------------------------|------------------------------------------------------------------------------------                                |   
| conv_standard          | The standard to be used for conversion <br> - `1` : Bt.709 HD <br> - `2` : Bt.601/407 |   
   
### 2d Noise Reduction

| 2d_noise_reduction | Details                                           | 
|--------------------|---------------------------------------------------|
| is_enable          | When enabled applies the 2D noise reduction       |  
| window_size        | Search window size for applying non-local means   |    
| wts                | Smoothening strength parameter                    |
### RGB Conversion

| rgb_conversion | Details                                           | 
|--------------------|---------------------------------------------------|
| is_enable           | When enabled sets pipeline output format to RGB otherwise it is YUV | 

### Invalid Region Crop
| invalid_region_crop    | Details                                                |
|---------------------------|--------------------------------------------------------|
| is_enable                  | Enables or disables this module                        |   
| crop_to_size               | Only have two values that sets crop dimensions <br> - `1` (1920x1080) <br> - `2` (1920x1440) | 
| height_start_idx           | Starting height-index for crop| 
| width_start_idx            | Starting width-index for crop| 


### Scaling 

| scale            | Details |   
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------
| is_enable         | When enabled down scales the input image                                                                                                                                                                                                       
| new_width        | Down scaled width of the output image                                                                                                      
| new_height       | Down scaled height of the output image                                                                                       
### YUV Format 
| yuv_conversion_format     | Details                                                |
|---------------------------|--------------------------------------------------------|
| is_enable                  | Enables or disables the module                        |   
| conv_type                 | Selects the YCbCr to YUV format <br> - `444` <br> - `422` |  

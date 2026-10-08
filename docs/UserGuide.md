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
| is_enable       | This is an essential module and cannot be disabled 
| is_auto         | Flag to let the 3A - Auto Exposure control choose `current_gain` for the next frame
| gain_array      | Gains array. List of permissible digital gains |
| current_gain    | Index for the current gain in gain_array. It starts from zero |

### 3A - Auto Exposure Statistics

The HDR-ISP AE statistics block (RTL `axis_ae_stat`), on the raw after digital gain. Per cell of a 3x3 grid it counts green pixels below `shadow_threshold` and above `highlight_threshold`, sums the green pixel values (32-bit, saturating) and counts the pixels of every channel at 0 and at full scale (32-bit counters). Comparisons are strict.

| auto_exposure_stats  | Details |
| -----------          |   ---   |
| is_enable            | When enabled computes the AE statistics. Must be enabled for the 3A - Auto Exposure control |
| col_starts           | 4 ascending column boundaries of the grid in pixels, the last closing the grid, e.g. `[0, 864, 1728, 2592]`. Cells are `[start, next start)` |
| row_starts           | 4 ascending row boundaries of the grid in pixels, e.g. `[0, 512, 1024, 1536]` |
| shadow_threshold     | Green pixels below it are counted as shadows (DN at the sensor bit depth) |
| highlight_threshold  | Green pixels above it are counted as highlights (DN at the sensor bit depth) |
| is_save              | Saves the per-cell statistics as a txt file |

### 3A - Auto Exposure

The HDR-ISP AE control law (`ae_ctrl.c`, EV law) as an RTL block, in integer arithmetic. It meters a clip-corrected, centre-weighted green mean from the statistics, computes the exposure error `log2(target_mean) - log2(mean)` in Q8 EV (256 = 1 EV), caps it when the highlight or clip fraction is over budget, never darkens a mostly dark frame, and makes a damped, slew-limited move inside a deadband with hysteresis. The move picks the nearest gain in `digital_gain.gain_array` for the next frame. A target between two gains settles on one of them (the smaller error, never a brighter gain that breaks the highlight / clip budget) instead of alternating, until the scene changes. In `render_3a` the pipeline re-runs until the gain holds.

Hardware: the block runs once per frame, during vertical blanking, after the statistics block latches the frame's statistics; its gain applies from the next frame. It has one shared divider (at most 17 divisions per frame) and a Q8 log2 unit. The register widths and internal bit widths are listed in [modules/auto_exposure.py](../modules/auto_exposure.py). With the block enabled, `digital_gain.gain_array` is its gain table: at most 128 gains, each a multiple of 1/256 from 1/256 to 255.996.

| auto_exposure      | Register | Details |
| -----------        | ---      |   ---   |
| is_enable          |          | When enabled applies the 3A - Auto Exposure control |
| target_mean        | bit_depth bits, 1 .. 2^bit_depth - 1 | Metered green mean to reach (DN at the sensor bit depth) |
| valid_min_pct      | 7 bits, 0 .. 100 | A grid cell needs this % of unclipped greens to be metered |
| grid_weights       | 9 x 8 bits, 0 .. 255 | 9 metering weights, cell index = row * 3 + col |
| ev_deadband        | 10 bits, 0 .. 1023 | Errors within it (Q8 EV) do not move the gain |
| ev_hyst            | 10 bits, 0 .. 1023 | Extra deadband (Q8 EV) once converged |
| ev_damp            | 7 bits, 0 .. 100 | % of the error moved per frame |
| ev_damp_fast       | 7 bits, 0 .. 100 | % of the error moved on a scene change |
| ev_slew_max        | 12 bits, 1 .. 4095 | Largest move per frame (Q8 EV) |
| ev_slew_fast       | 12 bits, 1 .. 4095 | Largest move on a scene change (Q8 EV) |
| scene_change_ev    | 12 bits, 0 .. 4095 | An error above it (Q8 EV) is a scene change |
| vm_iir_a           | 7 bits, 1 .. 100 | % IIR on the metered mean between frames (bypassed in `render_3a`) |
| hi_frac_pm         | 10 bits, 0 .. 1000 | Highlight budget: greens above `highlight_threshold`, per mille |
| hi_k               | 8 bits, 0 .. 255 | % strength of the highlight cap |
| clip_frac_pm       | 10 bits, 0 .. 1000 | Clip budget: pixels at full scale, per mille |
| clip_k             | 8 bits, 0 .. 255 | % strength of the clip cap |
| dark_frac_pm       | 10 bits, 0 .. 1000 | Above this shadow fraction (per mille) the frame is never made darker |
| ev_max_pull        | 12 bits, 0 .. 4095 | The caps may pull at most this much (Q8 EV) below the mean's request |

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
| stats_window_offset | Specifies the crop dimensions to obtain a stats calculation window <br> - Should be an array of elements `[Up, Down, Left, Right]` <br> - Should be a multiple of 4 |
| underexposed_percentage   | Set % of dark pixels to exclude before AWB gain calculation|
| overexposed_percentage    | Set % of saturated pixels to exclude before AWB gain calculation|

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

### Saturation Enhancement

The HDR-ISP saturation block (RTL `axis_sat`), on the 8-bit YUV after 2D noise reduction. Y passes through; each chroma sample becomes `clip(((C - 128) * SAT_GAIN) >>> 8 + 128)`, where `SAT_GAIN` is the 12-bit Q4.8 register and 128 the chroma pedestal of the colour space conversion.

| saturation_enhancement | Details                                           |
|--------------------|---------------------------------------------------|
| is_enable          | When enabled applies the saturation enhancement   |
| algorithm          | `global` (the RTL algorithm). `hue_rolloff` exists only in the algorithm-design model and is refused here |
| saturation_gain    | Chroma gain, truncated to the Q4.8 register (1/256 steps). Allowed 0 to 15.12: above it the RTL's 12-bit add stage wraps |
| is_save            | Saves the module output |

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

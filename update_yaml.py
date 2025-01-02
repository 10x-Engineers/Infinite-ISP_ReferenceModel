import tv_utils
import sys
from ast import literal_eval

config_select = sys.argv[1]
with open("./tb/include/Infinite-ISP_ReferenceModel/config/configs.yml", "r", encoding="UTF-8") as file:
    configs = tv_utils.yaml.safe_load(file)
     
if(config_select == 'sensor_info'): #For sensor 
    configs["sensor_info"]["bayer_pattern"] = sys.argv[2]
    configs["sensor_info"]["range"] = int(sys.argv[3])
    configs["sensor_info"]["bit_depth"] = int(sys.argv[4])
    configs["sensor_info"]["width"] = int(sys.argv[5])
    configs["sensor_info"]["height"] = int(sys.argv[6])
    #filename = str(sys.argv[7])

if(config_select == 'platform'):   
    configs["platform"]["filename"] = str(sys.argv[2])
    #filename = str(sys.argv[3])

if config_select == 'crop':  # For crop
    configs["crop"]["is_enable"] = bool(int(sys.argv[2]))
    configs["crop"]["new_width"] = int(sys.argv[3])
    configs["crop"]["new_height"] = int(sys.argv[4])
    #filename = str(sys.argv[5])

if config_select == 'dpc':
    configs["dead_pixel_correction"]["is_enable"] = bool(int(sys.argv[2]))
    configs["dead_pixel_correction"]["dp_threshold"] = int(sys.argv[3])
    #filename = str(sys.argv[4])

if config_select == 'blc':
    configs["black_level_correction"]["is_enable"] = bool(int(sys.argv[2]))
    configs["black_level_correction"]["r_offset"] = int(sys.argv[3])
    configs["black_level_correction"]["gr_offset"] = int(sys.argv[4])
    configs["black_level_correction"]["gb_offset"] = int(sys.argv[5])
    configs["black_level_correction"]["b_offset"] = int(sys.argv[6])
    configs["black_level_correction"]["is_linear"] = bool(int(sys.argv[7]))
    configs["black_level_correction"]["linear_r"] = float(sys.argv[8])
    configs["black_level_correction"]["linear_gr"] = float(sys.argv[9])
    configs["black_level_correction"]["linear_gb"] = float(sys.argv[10])
    configs["black_level_correction"]["linear_b"] = float(sys.argv[11])
    #filename = str(sys.argv[12])

if config_select == 'dg':
    configs["digital_gain"]["is_enable"] = bool(int(sys.argv[2]))
    configs["digital_gain"]["is_auto"] = bool(int(sys.argv[3]))
    configs["digital_gain"]["gain_array"] = literal_eval(sys.argv[4])
    configs["digital_gain"]["current_gain"] = int(sys.argv[5])
    configs["digital_gain"]["ae_feedback"] = int(sys.argv[6])
    #filename = str(sys.argv[7])

if config_select == 'yuvCFormat':
    configs["yuv_conversion_format"]["is_enable"] = bool(int(sys.argv[2]))
    configs["yuv_conversion_format"]["conv_type"] = sys.argv[3]
    #filename = str(sys.argv[4])

if config_select == 'demosaic':
    configs["demosaic"]["is_enable"] = bool(int(sys.argv[2]))
    #filename = str(sys.argv[3])

if config_select == 'scale':
    configs["scale"]["is_enable"] = bool(int(sys.argv[2]))
    configs["scale"]["new_width"] = int(sys.argv[3])
    configs["scale"]["new_height"] = int(sys.argv[4])
    #filename = str(sys.argv[5])

if config_select == '2dnr':
    configs["2d_noise_reduction"]["is_enable"] = bool(int(sys.argv[2]))
    configs["2d_noise_reduction"]["diff_value"] = literal_eval(sys.argv[3])
    configs["2d_noise_reduction"]["wts"] = literal_eval(sys.argv[4])
    #filename = str(sys.argv[5])

if config_select == 'csc':
    configs["color_space_conversion"]["is_enable"] = bool(int(sys.argv[2]))
    configs["color_space_conversion"]["conv_standard"] = int(sys.argv[3])
    #filename = str(sys.argv[4])

if config_select == 'ae':
    configs["auto_exposure"]["is_enable"] = bool(int(sys.argv[2]))
    configs["auto_exposure"]["stats_window_offset"] = literal_eval(sys.argv[3])
    configs["auto_exposure"]["center_illuminance"] = int(sys.argv[4])
    configs["auto_exposure"]["histogram_skewness"] = float(sys.argv[5])
    #filename = str(sys.argv[6])

if config_select == 'awb':
    configs["auto_white_balance"]["is_enable"] = bool(int(sys.argv[2]))
    configs["auto_white_balance"]["stats_window_offset"] = literal_eval(sys.argv[3])
    configs["auto_white_balance"]["underexposed_limit"] = int(sys.argv[4])
    configs["auto_white_balance"]["overexposed_limit"] = int(sys.argv[5])
    #filename = str(sys.argv[6])

if config_select == 'wb':
    configs["white_balance"]["is_enable"] = bool(int(sys.argv[2]))
    configs["white_balance"]["is_auto"] = bool(int(sys.argv[3]))
    configs["white_balance"]["r_gain"] = float(sys.argv[4])
    configs["white_balance"]["b_gain"] = float(sys.argv[5])
    #filename = str(sys.argv[6])

if config_select == 'ccm':
    configs["color_correction_matrix"]["is_enable"] = bool(int(sys.argv[2]))
    configs["color_correction_matrix"]["corrected_red"] = literal_eval(sys.argv[3])
    configs["color_correction_matrix"]["corrected_green"] = literal_eval(sys.argv[4])
    configs["color_correction_matrix"]["corrected_blue"] = literal_eval(sys.argv[5])
    #filename = str(sys.argv[6])

if config_select == 'bnr':
    configs["bayer_noise_reduction"]["is_enable"] = bool(int(sys.argv[2]))
    configs["bayer_noise_reduction"]["filter_window"] = int(sys.argv[3])
    configs["bayer_noise_reduction"]["space_kern_r"] = literal_eval(sys.argv[4])
    configs["bayer_noise_reduction"]["space_kern_g"] = literal_eval(sys.argv[5])
    configs["bayer_noise_reduction"]["space_kern_b"] = literal_eval(sys.argv[6])
    configs["bayer_noise_reduction"]["color_curve_x_r"] = literal_eval(sys.argv[7])
    configs["bayer_noise_reduction"]["color_curve_y_r"] = literal_eval(sys.argv[8])
    configs["bayer_noise_reduction"]["color_curve_x_g"] = literal_eval(sys.argv[9])
    configs["bayer_noise_reduction"]["color_curve_y_g"] = literal_eval(sys.argv[10])
    configs["bayer_noise_reduction"]["color_curve_x_b"] = literal_eval(sys.argv[11])
    configs["bayer_noise_reduction"]["color_curve_y_b"] = literal_eval(sys.argv[12])
    #filename = str(sys.argv[13])

if config_select == 'oecf':
    configs["oecf"]["is_enable"] = bool(int(sys.argv[2]))
    configs["oecf"]["r_lut"] = literal_eval(sys.argv[3])
    #filename = str(sys.argv[4])

if config_select == 'gamma':
    configs["gamma_correction"]["is_enable"] = bool(int(sys.argv[2]))
    configs["gamma_correction"]["gamma_lut"] = literal_eval(sys.argv[3])
    #filename = str(sys.argv[4])

# Adding the two new modules
if(config_select == 'rgb_conversion'):
    configs["rgb_conversion"]["is_enable"] = bool(int(sys.argv[2]))
    #filename = str(sys.argv[3])

if config_select == 'invalid_region_crop':
    configs["invalid_region_crop"]["is_enable"] = bool(int(sys.argv[2]))
    configs["invalid_region_crop"]["crop_to_size"] = int(sys.argv[3])
    configs["invalid_region_crop"]["height_start_idx"] = int(sys.argv[4])
    configs["invalid_region_crop"]["width_start_idx"] = int(sys.argv[5])
    #filename = str(sys.argv[6])

if config_select == 'on_screen_display':
    configs["on_screen_display"]["is_enable"] = bool(int(sys.argv[2]))
    configs["on_screen_display"]["height"] = int(sys.argv[3])
    configs["on_screen_display"]["width"] = int(sys.argv[4])
    configs["on_screen_display"]["x_offset"] = int(sys.argv[5])
    configs["on_screen_display"]["y_offset"] = int(sys.argv[6])
    configs["on_screen_display"]["fg_color"] = literal_eval(sys.argv[7])
    configs["on_screen_display"]["bg_color"] = literal_eval(sys.argv[8])
    configs["on_screen_display"]["alpha"] = int(sys.argv[9])
    #filename = str(sys.argv[10])

# Save the updated configs back to the file
tv_utils.save_config(configs, "./tb/include/Infinite-ISP_ReferenceModel/config/configs.yml")
#tv_utils.save_config(configs, "./tb/include/Infinite-ISP_ReferenceModel/in_frames/normal/file"+#filename+"-configs.yml")

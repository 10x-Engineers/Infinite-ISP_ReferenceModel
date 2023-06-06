# Automated Testing for Infinite-ISP
The automated script is designed to streamline the testing and debugging process of Infinite-ISP. It includes a configuration file that allows for customization of automation settings. The script provides comprehensive analysis of the Device Under Test (DUT), which can be a single module or a sequence of modules. It returns input to the DUT as well as its output, which can be in the form of a numpy array or a PNG image. The configuration file contains specific details and parameters for the automated script, enabling efficient and effective testing and debugging of the Infinite-ISP.

## How to Run?

Follow the following steps to run the automated testing:

1.  Go to [test_vector_generation](test_vector_generation)

2.  Set automation preferences through [tv_config.yml](test_vector_generation/tv_config.yml).

3. Run [automate_execution.py](test_vector_generation/automate_execution.py)

## Automation Configuration Parameters

The configuration file [tv_config.yml](test_vector_generation/tv_config.yml) plays a crucial role in the automated script, allowing users to customize and control the various parameters of the Infinite-ISP for optimal testing and debugging. By utilizing this configuration file, users can effortlessly modify the Infinite-ISP settings without the need to manually enable or disable individual modules within the default configuration. Below are the detailed configuration parameters for the script:

| Parameters | Details                                           | 
|--------------------|---------------------------------------------------|
| dataset_path           | Path to the dataset <br> Testing is done on all raw images in the mentioned folder | 
| config_path           | Path to the config file for dataset images |   
| input_ext        | Extension for binary files produced by automated script <br> - `.raw` <br> - `.bin`  |     
| dut         | Device Under Test <br> It can be single module or set of modules in sequence. Details of how to set DUT are below   |
| is_enable         | Flag to enable non-default modules   |
| is_save         | Flag to save the test vector for default modules   | 

### DUT

The DUT (Device Under Test) refers to the specific module or set of modules that are being tested. The DUT can be any of the following modules:

1. crop
2. dead_pixel_correction
3. black_level_correction
4. oecf
5. digital_gain
6. bayer_noise_reduction
7. white_balance
8. demosaic
9. color_correction_matrix
10. gamma_correction
11. color_space_conversion
12. 2d_noise_reduction
13. rgb_conversion
14. invalid_region_crop
15. scale
16. yuv_conversion_format

When specifying the DUT (Device Under Test), it is important to use the exact names from the provided list. The DUT can be defined in two ways:

1. Single Module: The DUT can be any individual module from the list above, such as "demosaic" or "scale"

2. Multiple Modules in Sequence: When testing multiple modules in sequence, it is crucial to follow the specified order. For example, "crop, white_balance, scale" would be a valid sequence. However, "white_balance" cannot precede the "digital_gain" in the DUT list.

By accurately matching the DUT name to the provided list, it is ensured that the correct modules are being tested, either individually or in the specified order, while considering any dependencies between the modules.

### Default Automation Config Parameters

The aforementioned parameters are intrinsic to the automated script, while the remaining configuration parameters are sourced from the given config. During automated testing, these parameters take precedence over the given script, allowing for their replacement and utilization. Once the testing is completed, the given config is restored to ensure consistency and proper functioning of the system. This approach enables a seamless integration of the automated scripts with the given configuration, providing reliable and efficient testing capabilities.

1. platform:
  <br> - save_lut: True
  <br> - is_debug: True
  <br> - generate_tv: True
  <br> - save_format: npy or both
  <br> - render_3a: False 

2. For non-default modules in DUT list: 
 <br> - is_enable: True
 <br> - is_save: True

 ## Results

 After automated script execution, results are stored in a folder in [test_vectors](test_vector_generation/test_vectors) with a name that includes Time-Stamp and DUT. The final result folder contains:

 1. Input-Output results (numpy-array and png-image) for each DUT.
 2. Final output of Infinite-ISP
 3. Configuration file of Infinite-ISP
 4. Configuration file of automation script.
 5. Logs for ISP-Pipeline execution.
#!/bin/bash

# Clear the .cache folder
# rm -rf ~/.cache/*

# Inventory the images
python3 my_work/new_inventory_images.py

# Run the testing script
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0 output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth

# Convert the results pickle file to json
python3 my_work/pickle2json.py output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/inference_model_final_wo_optim/ycbv_test/results.pkl
python3 my_work/convert_results.py

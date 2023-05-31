import os
import json
from PIL import Image

# Path to the test folder
dataset_folder = "datasets/BOP_DATASETS/ycbv"
test_folder = os.path.join(os.path.join(dataset_folder, "test"))

data_list = []
# Loop through subdirectories with strictly positive integer names
for subdir_name in os.listdir(test_folder):
    if subdir_name.isdigit() and int(subdir_name) > 0:
        subdir_path = os.path.join(os.path.join(test_folder, subdir_name), "rgb")
        # Loop through images in the subdirectory
        for filename in os.listdir(subdir_path):
            filepath = os.path.join(subdir_path, filename)
            # Check if image is 640x480
            with Image.open(filepath) as img:
                if img.size == (640, 480):
                    # Get the name of the image
                    image_name = os.path.splitext(filename)[0]
                    print(f"Found matching image in {subdir_name}: {image_name}")
                    data_list.append((subdir_name, image_name))

test_targets = []
for (subdir_name, image_name) in data_list:
    # Remove the leading 0 from subdir_name and image_name
    test_targets.append({"im_id": int(str(image_name).lstrip("0")), "inst_count": 1, "obj_id": 11, "scene_id": int(str(subdir_name).lstrip("0"))})
# Modify the JSON file
with open(os.path.join(dataset_folder, "test_targets.json"), "w") as f:
    json.dump(test_targets, f)

# Modify the text file
with open(os.path.join(os.path.join(dataset_folder, "image_sets"), "keyframe.txt"), "w") as f:
    for (subdir_name, image_name) in data_list:
        f.write(f"{subdir_name}/{image_name}\n")

# Modify scene_gt
for (subdir_name, image_name) in data_list:
    # Construct path to scene_gt.json
    json_path = os.path.join(os.path.join(test_folder, subdir_name), "scene_gt.json")
    # Load scene_gt.json if it exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            scene_gt = json.load(f)
    else:
        scene_gt = {}
    image_name = int(str(image_name).lstrip("0"))
    # Check if image_name is in scene_gt
    if image_name not in scene_gt:
        # Add default entry for image_name
        scene_gt[image_name] = [{
            "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "cam_t_m2c": [0, 0, 0],
            "obj_id": 11
        }]
        # Write updated scene_gt to file
        with open(json_path, "w") as f:
            json.dump(scene_gt, f)

# Modify scene_gt_info
for (subdir_name, image_name) in data_list:
    # Construct path to scene_gt_info.json
    json_path = os.path.join(os.path.join(test_folder, subdir_name), "scene_gt_info.json")
    # Load scene_gt_info.json if it exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            scene_gt_info = json.load(f)
    else:
        scene_gt_info = {}
    image_name = int(str(image_name).lstrip("0"))
    # Check if image_name is in scene_gt_info
    if image_name not in scene_gt_info:
        # Add default entry for image_name
        scene_gt_info[image_name] = [{
            "bbox_obj": [0, 640, 0, 480],
            "bbox_visib": [0, 640, 0, 480],
            "px_count_all": 307200,
            "px_count_valid": 307200,
            "px_count_visib": 307200,
            "visib_fract": 1.0
        }]
        # Write updated scene_gt_info to file
        with open(json_path, "w") as f:
            json.dump(scene_gt_info, f)

# Modify scene_camera
for (subdir_name, image_name) in data_list:
    # Construct path to scene_camera.json
    json_path = os.path.join(os.path.join(test_folder, subdir_name), "scene_camera.json")
    # Load scene_camera.json if it exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            scene_camera = json.load(f)
    else:
        scene_camera = {}
    image_name = int(str(image_name).lstrip("0"))
    # Check if image_name is in scene_camera
    if image_name not in scene_camera:
        # Add default entry for image_name
        scene_camera[image_name] = {
            "cam_K": [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0],
            "cam_R_w2c": [0.775038, 0.630563, -0.0413049, 0.1427, -0.238322, -0.960645, -0.615591, 0.738643, -0.27469],
            "cam_t_w2c": [22.278120142899976, 67.27103635299997, 833.583980809],
            "depth_scale": 0.1
        }
        # Write updated scene_camera to file
        with open(json_path, "w") as f:
            json.dump(scene_camera, f)
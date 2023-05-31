import json
import numpy as np
import os
from PIL import Image
import shutil

# Path to the test folder
dataset_folder = "datasets/BOP_DATASETS/ycbv"
type = "train_pbr" # other: test
folder = os.path.join(os.path.join(dataset_folder, type))

data_list = []
data_dic = {}
# Create a new black image with the specified dimensions
black_image = Image.new('RGB', (640, 480), color='black')
black_image.convert('1')
# Define the custom threshold value
threshold = 50
# Loop through subdirectories with strictly positive integer names
for subdir_name in os.listdir(folder):
    if subdir_name.isdigit() and int(subdir_name) > 0:
        data_dic[subdir_name] = []
        subdir_path_source = os.path.join(os.path.join(folder, subdir_name), "source")
        subdir_path_rbg = os.path.join(os.path.join(folder, subdir_name), "rgb")
        subdir_path_mask = os.path.join(os.path.join(folder, subdir_name), "mask")
        subdir_path_mask_visib = os.path.join(os.path.join(folder, subdir_name), "mask_visib")
        os.makedirs(subdir_path_rbg, exist_ok=True)
        os.makedirs(subdir_path_mask, exist_ok=True)
        os.makedirs(subdir_path_mask_visib, exist_ok=True)
        # Loop through images in the subdirectory
        for filename in os.listdir(subdir_path_source):
            filepath = os.path.join(subdir_path_source, filename)
            # Check if image is 640x480
            with Image.open(filepath) as img:
                if img.size == (640, 480):
                    # Get the name of the image
                    image_name = os.path.splitext(filename)[0]
                    parts = image_name.split('_')
                    image_name = parts[0]
                    # Create the new name with six digits
                    new_image_name = image_name.zfill(6) + ('_000011' if len(parts)>1 else '')
                    subdirs_path = [subdir_path_mask, subdir_path_mask_visib] if len(parts)>1 else [subdir_path_rbg]
                    for subdir_path in subdirs_path:
                        # Construct the new filepath
                        new_filepath = os.path.join(subdir_path, f"{new_image_name}.png")
                        if len(parts)==1:
                            # Copy the file to the destination directory
                            shutil.copy2(filepath, new_filepath)
                            data_list.append((subdir_name, new_image_name))
                            data_dic[subdir_name].append(str(new_image_name).lstrip("0"))
                        else:
                            image = Image.open(filepath)
                            # Convert the image to binary using the custom threshold
                            binary_image = image.convert('L').point(lambda pixel: 0 if pixel < threshold else 255, mode='1')
                            # Save the binary image at the new file path
                            binary_image.save(new_filepath)
                            # Create black images for other object's masks
                            for i in range(21):
                                if (i!=11):
                                    new_image_name_black = image_name.zfill(6) + '_' + str(i).zfill(6)
                                    new_filepath_black = os.path.join(subdir_path, f"{new_image_name_black}.png")
                                    # Save the black image as a PNG file
                                    black_image.save(new_filepath_black)


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

for subdir_name in data_dic:
    # Construct path to data.json
    data_path = os.path.join(os.path.join(folder, subdir_name), "data.json")
    
    # Load data_path.json if it exists
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data_gt = json.load(f)
        
        if type=="test":
            # Modify test_bboxes
            # Construct path to test_bboxes.json
            json_path = os.path.join(os.path.join(folder, "test_bboxes"), "yolox_x_640_ycbv_real_pbr_ycbv_bop_test.json")
            # Load test_bboxes.json if it exists
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    test_bboxes = json.load(f)
            else:
                test_bboxes = {}
            for image_name in data_dic[subdir_name]:
                # Check if image_name is in scene_gt
                if image_name not in test_bboxes:
                    # Add entry for image_name
                    test_bboxes["{}/{}".format(subdir_name.lstrip("0"), image_name)] = [{
                        "bbox_est": data_gt[image_name]["bbox"],
                        "obj_id": 11
                    }]
            # Write updated test_bboxes to file
            with open(json_path, "w") as f:
                json.dump(test_bboxes, f, indent=2)

        # Modify scene_gt
        # Construct path to scene_gt.json
        json_path = os.path.join(os.path.join(folder, subdir_name), "scene_gt.json")
        # Load scene_gt.json if it exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                scene_gt = json.load(f)
        else:
            scene_gt = {}
        for image_name in data_dic[subdir_name]:
            # Check if image_name is in scene_gt
            if image_name not in scene_gt:
                # Add entry for image_name
                scene_gt[image_name] = [{
                    "cam_R_m2c": data_gt[image_name]["pose"]["cam_R_m2c"],
                    "cam_t_m2c": data_gt[image_name]["pose"]["cam_t_m2c"],
                    "obj_id": 11
                }]
        # Write updated scene_gt to file
        with open(json_path, "w") as f:
            json.dump(scene_gt, f)

        # Modify scene_gt_info
        # Construct path to scene_gt_info.json
        json_path = os.path.join(os.path.join(folder, subdir_name), "scene_gt_info.json")
        # Load scene_gt_info.json if it exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                scene_gt_info = json.load(f)
        else:
            scene_gt_info = {}
        for image_name in data_dic[subdir_name]:
            # Check if image_name is in scene_gt_info
            if image_name not in scene_gt_info:
                # Add entry for image_name
                bbox = data_gt[image_name]["bbox"]
                scene_gt_info[image_name] = [{
                    "bbox_obj": bbox,
                    "bbox_visib": bbox,
                    "px_count_all": bbox[2] * bbox[3],
                    "px_count_valid": bbox[2] * bbox[3],
                    "px_count_visib": bbox[2] * bbox[3],
                    "visib_fract": 1.0
                }]
        # Write updated scene_gt_info to file
        with open(json_path, "w") as f:
            json.dump(scene_gt_info, f)

        # Modify scene_camera
        # Construct path to scene_camera.json
        json_path = os.path.join(os.path.join(folder, subdir_name), "scene_camera.json")
        # Load scene_camera.json if it exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                scene_camera = json.load(f)
        else:
            scene_camera = {}
        for image_name in data_dic[subdir_name]:
            # Check if image_name is in scene_camera
            if image_name not in scene_camera:
                # Get the extrinsics
                extrinsics = np.array(data_gt["camera"]["cam_extrinsics"])
                # Extract rotation matrix
                rotation_matrix = extrinsics[:3, :3]
                # Extract translation vector
                translation_vector = extrinsics[:3, 3]
                # Inverse
                world_to_camera_rotation = rotation_matrix.T
                world_to_camera_translation = -np.dot(world_to_camera_rotation, translation_vector)
                # Add entry for image_name
                scene_camera[image_name] = {
                    "cam_K": data_gt[image_name]["cam_K"],
                    "cam_R_w2c": world_to_camera_rotation.flatten().tolist(),
                    "cam_t_w2c": world_to_camera_translation.tolist(),
                    "depth_scale": 0.1
                }
        # Write updated scene_camera to file
        with open(json_path, "w") as f:
            json.dump(scene_camera, f)

    else:

        # Modify scene_gt
        # Construct path to scene_gt.json
        json_path = os.path.join(os.path.join(folder, subdir_name), "scene_gt.json")
        # Load scene_gt.json if it exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                scene_gt = json.load(f)
        else:
            scene_gt = {}
        for image_name in data_dic[subdir_name]:
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
        # Construct path to scene_gt_info.json
        json_path = os.path.join(os.path.join(folder, subdir_name), "scene_gt_info.json")
        # Load scene_gt_info.json if it exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                scene_gt_info = json.load(f)
        else:
            scene_gt_info = {}
        for image_name in data_dic[subdir_name]:
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
        # Construct path to scene_camera.json
        json_path = os.path.join(os.path.join(folder, subdir_name), "scene_camera.json")
        # Load scene_camera.json if it exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                scene_camera = json.load(f)
        else:
            scene_camera = {}
        for image_name in data_dic[subdir_name]:
            image_name = int(str(image_name).lstrip("0"))
            # Check if image_name is in scene_camera
            if image_name not in scene_camera:
                # Add default entry for image_name
                scene_camera[image_name] = {
                    "cam_K": [1000, 0.0, 320, 0, 1000, 480, 0, 0, 1],
                    "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "cam_t_w2c": [0, 0, 0],
                    "depth_scale": 0.1
                }
        # Write updated scene_camera to file
        with open(json_path, "w") as f:
            json.dump(scene_camera, f)
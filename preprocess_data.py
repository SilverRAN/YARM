import argparse
import json
import math
import os
import shutil
import tqdm

def extract_extrinsics(extrinsic_matrix):
    assert len(extrinsic_matrix) == 4 and len(extrinsic_matrix[0]) == 4, "Input must be 4x4 matrix"
    R = [row[:3] for row in extrinsic_matrix[:3]]
    t = [[row[3]] for row in extrinsic_matrix[:3]]
    return R, t

def get_extrinsic_param(frame_i: dict):
    extrinsic_dict = {}
    extrinsic_dict['rotation'], extrinsic_dict['translation'] = extract_extrinsics(frame_i['transform_matrix'])
    return extrinsic_dict

def get_focal_length(camera_angle_x: float, width: int):
    return width / (2 * math.tan(camera_angle_x / 2))

def get_intrinsic_param(camera_angle_x: float):
    """Calculate intrinsic camera parameters based on the camera's horizontal field of view."""
    BOUNDS = [1.0, 8.0]
    H = W = 800.0 # Bounds and H, W are fixed for IM3D dataset, change it to corresponding values if using other datasets
    intrinsic_dict = {}
    intrinsic_dict['height'] = H
    intrinsic_dict['width'] = W
    intrinsic_dict['focal'] = get_focal_length(camera_angle_x, W)
    intrinsic_dict['bounds'] = BOUNDS
    return intrinsic_dict

def get_new_json_dict(json_data):
    new_dict = {}
    camera_angle_x = json_data['camera_angle_x']
    for frame_i in json_data['frames']:
        file_name = "{:0>6d}".format(int(frame_i['file_path'].split('/')[-1].split('_')[-1])) + '.png'
        ext_dict = get_extrinsic_param(frame_i)
        int_dict = get_intrinsic_param(camera_angle_x)
        params_dict = dict(extrinsic=ext_dict, intrinsic=int_dict)
        new_dict.update({file_name: params_dict})
        
    return new_dict

def load_json(json_path: str):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def main(args):
    print("Found {} objects in the dataset! \n".format(len(os.listdir(args.path))))
    for object in tqdm.tqdm(os.listdir(args.path)):
        os.makedirs(f'./data/{object}', exist_ok=True)

        for subfolder in ['train', 'val', 'test']:
            os.makedirs(f'./data/{object}/{subfolder}', exist_ok=True)
            # copy image files
            if subfolder == 'train':
                file_num = len(os.listdir(os.path.join(args.path, object, subfolder)))
            else: # For val and test
                file_num = len(os.listdir(os.path.join(args.path, object, subfolder)))
            for i in range(file_num):
                file_name = os.path.join(args.path, object, f"{subfolder}/r_{i}.png")
                new_file_name = os.path.join(f"./data/{object}/{subfolder}", f"{i:0>6}.png")
                shutil.copy(file_name, new_file_name)
        
            # copy new json file
            old_json_path = os.path.join(args.path, object, f"transforms_{subfolder}.json")
            json_data = load_json(old_json_path)
            new_dict = get_new_json_dict(json_data)
            with open(f'./data/{object}/{subfolder}_camera_params.json', 'w') as f:
                json.dump(new_dict, f, indent=4)
    print("Finished processing the dataset!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process IM3D dataset.')
    parser.add_argument('--path', type=str, required=True, help='Path to IM3D dataset')
    args = parser.parse_args()

    main(args)
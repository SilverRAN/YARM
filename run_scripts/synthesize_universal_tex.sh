#!/bin/bash
echo "Starting Optimize Universal Robust Texture!"

scene=$1
model=$2
label=$3

python -u learn_universal_texture.py \
	-d ./data/${scene}_01/ \
	-o logs/${scene}/universal_${model}/ \
	-i logs/${scene}_01/ref/saved_models/model_final.pth \
	--log_wandb=False \
	--proxy_model=$model \
	--label=$label

# Rendering Output Video:
echo "Starting Rendering..."
python -u render_sh_based_voxel_grid.py \
	-i logs/${scene}_10/ref/saved_models/model_final.pth \
	-d ./data/${scene}_10/ \
	--camera_path dataset \
	-o output_renders/${scene}_10/universal_${model}/ \
	-dp logs/${scene}/universal_${model}/saved_models/model_final.pth
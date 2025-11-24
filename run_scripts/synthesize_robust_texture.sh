#!/bin/bash
echo "Starting Optimize Robust Texture!"

scene=$1
model=$2
label=$3

python edit_pretrained_relu_field.py \
	-d ./data/${scene}/ \
	-o logs/${scene}/robust_${model}/ \
	-i logs/${scene}/ref/saved_models/model_final.pth \
	--log_wandb=False \
	--proxy_model=$model \
	--label=$label

# Rendering Output Video:
echo "Starting Rendering..."
python render_sh_based_voxel_grid.py \
	-i logs/${scene}/ref/saved_models/model_final.pth \
	-d ./data/${scene}/ \
	--camera_path dataset \
	-o output_renders/${scene}/robust_${model}/ \
	-dp logs/${scene}/robust_${model}/saved_models/model_final.pth
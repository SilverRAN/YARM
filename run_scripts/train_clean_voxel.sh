#!/bin/bash
echo "Starting Training..."

scene=$1

echo "Training scene: ${scene}"

python train_sh_based_voxel_grid_with_posed_images.py \
	-d ./data/${scene}/ \
	-o logs/${scene}/ \
	--fast_debug_mode=True \
	--sh_degree=0
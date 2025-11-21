# [You Always Recognize Me(YARM): Robust Texture Synthesis Against Multi-View Corruption](https://openreview.net/pdf?id=nLW7e7KjN0)
This is the official repository of ICML 2025 Paper **You Always Recognize Me: Robust Texture Synthesis Against Multi-View Corruption.**

## Data Preparation
The IM3D dataset can be downloaded from the [VIAT repository](https://github.com/Heathcliff-saku/VIAT).
Use the following script to convert the format of camera pose

    python preprocess_data.py --path /Path/To/IM3D/Dataset/

The processed data will be organized in ./data/ as following structure:
```
|-- data
    |-- airliner_01
        |-- train
            |-- 000000.png
            |-- 000001.png
            |-- ...
        |-- val
        |-- test
        |-- train_camera_params.json
        |-- val_camera_params.json
        |-- test_camera_params.json
    |-- airliner_02
    |-- ...
```

## Installation
    conda create --name yarm python=3.10.10 --yes
    conda activate yarm
    pip install -r requirements.txt

## Demo
In progress...

## Citation
```
@inproceedings{ranyou,
  title={You Always Recognize Me (YARM): Robust Texture Synthesis Against Multi-View Corruption},
  author={Ran, Weihang and Yuan, Wei and Zheng, Yinqiang},
  booktitle={Forty-second International Conference on Machine Learning}
}
```


## Acknowledgements
We would like to thank the maintainers of the following repositories.
- [Vox-E](https://github.com/TAU-VAILab/Vox-E)

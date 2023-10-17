## Installation
First initialize the conda environment
```shell
conda create -n FSF python=3.8 -y
conda activate FSF
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Then, install the mmdet3d
```shell
#mmcv
pip install mmcv-full==1.3.9
#mmdet
pip install mmdet==2.14.0
#mmdet 3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout -v 0.17.1
pip install -v -e .
```


## Data Preparation
First, make the data dir
```shell
mkdir data
```
Then, please download the nuScenes and Argoverse 2 dataset and organize the data dir as follow:
```
├── data
|   ├── nuscenes
|   |   ├── samples
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   ├── CAM_BACK_RIGHT
│   │   │   ├── CAM_FRONT
│   │   │   ├── CAM_FRONT_LEFT
│   │   │   ├── CAM_FRONT_RIGHT
│   │   │   ├── LIDAT_TOP
|   |   ├── sweeps
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   ├── CAM_BACK_RIGHT
│   │   │   ├── CAM_FRONT
│   │   │   ├── CAM_FRONT_LEFT
│   │   │   ├── CAM_FRONT_RIGHT
│   │   │   ├── LIDAT_TOP
|   |   ├── v1.0-train
|   |   ├── v1.0-val
|   |   ├── v1.0-trainval
|   |   ├── nuscenes_infos_train.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_trainval.pkl
│   ├── argo2
│   │   │── argo2_format
│   │   │   │   │──sensor
│   │   │   │   │   │──train
│   │   │   │   │   │   │──...
│   │   │   │   │   │──val
│   │   │   │   │   │   │──...
│   │   │   │   │   │──test
│   │   │   │   │   │   │──0c6e62d7-bdfa-3061-8d3d-03b13aa21f68
│   │   │   │   │   │   │──0f0cdd79-bc6c-35cd-9d99-7ae2fc7e165c
│   │   │   │   │   │   │──...
│   │   │   │   │   │──val_anno.feather
│   │   │── kitti_format
│   │   │   │   │──argo2_infos_train.pkl
│   │   │   │   │──argo2_infos_val.pkl
│   │   │   │   │──argo2_infos_test.pkl
│   │   │   │   │──argo2_infos_trainval.pkl
│   │   │   │   │──training
│   │   │   │   │──testing
│   │   │   │   │──argo2_gt_database
```
Please download the nuimage-pretrained HTC from mmdet3d and change the correspoining path in './tools/mask_tools/save_mask_argo2.py' and './tools/mask_tools/save_mask_nusc.py'
Then use our scripts for pre-infering and saving 2D mask
```shell
./tools/mask_tools/save_mask_nusc.sh
./tools/mask_tools/save_mask_argo2.sh
```

## Train and Test
### nuScenes
After the preparation, you can train our model with 8 GPUs on nuScenes using:
```shell
./tools/nusc_train.sh nuScenes/FSF_nuScenes_config 8
```
For testing, please run the command:
```shell
./tools/dist_test.sh projects/configs/nuScenes/FSF_nuScenes_config.py $CKPT_PATH$ 8
```

### Argoverse 2
For training on Argoverse 2 with 8 GPUs, please using:
```shell
./tools/argo_train.sh Argoverse2/FSF_AV2_config 8
```
For testing, please run:
```shell
./tools/dist_test.sh projects/configs/Argoverse2/FSF_AV2_config.py $CKPT_PATH$ 8
```

## Checkpoints and logs
The checkpoints, training logs and detailed evaluation results is [the github release](https://github.com/AnonymousUsersGithub/Anonymous/releases/tag/Checkpoints).


## TODO
- [ ] when generating masks, add the mmdet3d-based config and ckpts.
import os
from os import path as osp
import torch
from pyarrow import feather
import pandas as pd
import numpy as np
import multiprocessing as mp
import tqdm
import pickle as pkl
from pyquaternion import Quaternion
from utils import LABEL_ATTR, cuboid_to_vertices
from SO3 import quat_to_yaw, quat_to_mat

from vis_prj_lidar_pts import prj_pts_to_img, draw_lidar_pts
import cv2

def read_feather(path, columns = None) -> pd.DataFrame:
    """Read Apache Feather data from a .feather file.
    AV2 uses .feather to serialize much of its data. This function handles the deserialization
    process and returns a `pandas` DataFrame with rows corresponding to the records and the
    columns corresponding to the record attributes.
    Args:
        path: Source data file (e.g., 'lidar.feather', 'calibration.feather', etc.)
        columns: Tuple of columns to load for the given record. Defaults to None.
    Returns:
        (N,len(columns)) Apache Feather data represented as a `pandas` DataFrame.
    """
    with open(path, "rb") as f:
        data: pd.DataFrame = feather.read_feather(f, columns=columns)
    return data

def get_coor_trans_mat(info):
    """
    quat_array [4]
    tran_arary [3]
    """
    quat_array = info.values[1:5].astype(np.float64)
    trans_array = info.values[5:8].astype(np.float64)

    rot_mat = torch.tensor(Quaternion(quat_array).rotation_matrix)
    trans_mat = torch.from_numpy(trans_array).unsqueeze(-1)

    coor_mat = torch.cat([rot_mat, trans_mat], dim=-1)
    padding = torch.zeros((1, 4))
    padding[:, 3] = 1
    ext_mat_padding = torch.cat([coor_mat, padding], dim=0) #N, 4, 4
    return ext_mat_padding

def get_lidar2img(segment_calib_extri, segment_calib_intri, ego_motion, camera_frame_path_list, lidar_frame_path):
    cam_num = 7
    cam_names = segment_calib_extri.values[:cam_num, 0]
    quat_array_all = segment_calib_extri.values[:cam_num, 1:5].astype(np.float32)
    trans_array_all = segment_calib_extri.values[:cam_num, 5:8].astype(np.float32)

    lidar_time_stamp = int(osp.basename(lidar_frame_path).replace('.feather', ''))
    lidar_ego_info = ego_motion[ego_motion['timestamp_ns'] == lidar_time_stamp].iloc[0]
    lidar_ego2gloabl = get_coor_trans_mat(lidar_ego_info)

    ego2cam_list = []
    for cam_id in range(cam_num):     
        #1. lidar stamp ego -> global

        #2. global -> camera stamp ego
        #get camera timestamp
        cam_frame_path = camera_frame_path_list[cam_id]
        cam_time_stamp = int(osp.basename(cam_frame_path).replace('.jpg', ''))
        cam_ego_info = ego_motion[ego_motion['timestamp_ns'] == cam_time_stamp].iloc[0]
        cam_ego2gloabl = get_coor_trans_mat(cam_ego_info)
        global2cam_ego = torch.linalg.inv(cam_ego2gloabl)

        lidar_ego2cam_ego = torch.mm(global2cam_ego, lidar_ego2gloabl)
        
        #3. cam stamp ego -> cam coor
        calib_info = segment_calib_extri.iloc[cam_id]
        cam2cam_ego = get_coor_trans_mat(calib_info)
        cam_ego2cam = torch.linalg.inv(cam2cam_ego)
        lidar_ego2cam = torch.mm(cam_ego2cam, lidar_ego2cam_ego)

        ego2cam_list.append(lidar_ego2cam)

    ego2cam_all = torch.stack(ego2cam_list)  

    K_mat = torch.zeros((cam_num, 3, 4), dtype=torch.float64) #[N, 3, 4]
    K_infos = torch.from_numpy(segment_calib_intri.values[:cam_num, 1:8].astype(np.float64))
    K_mat[:, 0, 0] = K_infos[:, 0] #fx
    K_mat[:, 1, 1] = K_infos[:, 1] #fy
    K_mat[:, 0, 2] = K_infos[:, 2] #cx
    K_mat[:, 1, 2] = K_infos[:, 3] #cy
    K_mat[:, 0, 3] = K_infos[:, 4] #k1
    K_mat[:, 1, 3] = K_infos[:, 5] #k2
    K_mat[:, 2, 3] = K_infos[:, 6] #k3
    K_mat[:, 2, 2] = 1

    lidar2img = torch.bmm(K_mat, ego2cam_all).float()
    return lidar2img, K_mat, ego2cam_all

def get_ego_motion(segment_path):
    ego_info = read_feather(os.path.join(segment_path, 'city_SE3_egovehicle.feather'))
    return ego_info

def vis_prj_pts(lidar_frame_path, camera_frame_path_list, lidar2img, K_mat, ext_mat_padding):
    #get lidar pts
    lidar = read_feather(lidar_frame_path)
    pts = lidar.loc[:, ['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
    pts = torch.from_numpy(pts[:, :3])
    #load imgs
    img_list = [cv2.imread(camera_frame_path) for camera_frame_path in camera_frame_path_list ]

    ref_pts_cam, pts_mask = prj_pts_to_img(pts, lidar2img, img_list)
    masked_lidar_img_list = draw_lidar_pts(ref_pts_cam, pts_mask.squeeze(-1), img_list)
    
    #save vis
    lidar_ts = os.path.basename(lidar_frame_path)
    out_dir = f'vis/argo2/lidar_prj_pts/use_ego/{lidar_ts}'
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(masked_lidar_img_list):
        cv2.imwrite(os.path.join(out_dir, f'{idx}_{os.path.basename(camera_frame_path_list[idx])}.png'), img)

def get_closest_cam_frame(cam_dir, ts):
    cam_files_list = os.listdir(cam_dir)
    cam_ts_list = []
    for cam_file in cam_files_list:
        cam_ts = int(cam_file.replace(".jpg", ''))
        cam_ts_list.append(cam_ts)
    cam_ts_array = np.array(cam_ts_list)
    min_idx = np.abs(cam_ts_array - ts).argmin()
    return cam_files_list[min_idx]


def process_single_segment(segment_path, split, info_list, ts2idx, output_dir, save_bin):
    test_mode = 'test' in split
    if not test_mode:
        segment_anno = read_feather(osp.join(segment_path, 'annotations.feather'))
    ego_motion = get_ego_motion(segment_path)
    segment_calib_extri = read_feather(osp.join(segment_path, 'calibration', 'egovehicle_SE3_sensor.feather'))
    segment_calib_intri = read_feather(osp.join(segment_path, 'calibration', 'intrinsics.feather'))


    segname = segment_path.split('/')[-1]

    lidar_frame_path_list = os.listdir(osp.join(segment_path, 'sensors/lidar/'))
    lidar_frame_path_list.sort()
    camera_name_list = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left',
                        'ring_rear_right', 'ring_side_left', 'ring_side_right']

    for f_idx, lidar_frame_name in enumerate(lidar_frame_path_list):
        ts = int(osp.basename(lidar_frame_name).split('.')[0])   

        if not test_mode:
            frame_anno = segment_anno[segment_anno['timestamp_ns'] == ts]
        else:
            frame_anno = None

        lidar_frame_path = osp.join(segment_path, 'sensors/lidar/', lidar_frame_name)
        camera_frame_path_list = []
        for cam_name in camera_name_list:
            cam_dir = osp.join(segment_path, 'sensors/cameras/', cam_name,)
            cam_frame_name = get_closest_cam_frame(cam_dir, ts)

            camera_frame_path =  osp.join(cam_dir, cam_frame_name)
            camera_frame_path_list.append(camera_frame_path)
        
        lidar2img, K_mat, ext_mat_padding = get_lidar2img(segment_calib_extri, segment_calib_intri, ego_motion, camera_frame_path_list, lidar_frame_path)

        frame_info = process_and_save_frame(lidar_frame_path, camera_frame_path_list, lidar2img, frame_anno, ts2idx, segname, output_dir, save_bin)
        info_list.append(frame_info)

def process_and_save_frame(lidar_frame_path, camera_frame_path_list, lidar2img, frame_anno, ts2idx, segname, output_dir, save_bin):
    frame_info = {}
    frame_info['uuid'] = segname + '/' + lidar_frame_path.split('/')[-1].split('.')[0]
    frame_info['sample_idx'] = ts2idx[frame_info['uuid']]
    frame_info['image'] = camera_frame_path_list
    frame_info['point_cloud'] = dict(
        num_features=4,
        velodyne_path=None,
    )
    frame_info['lidar2img'] = lidar2img.numpy()
    frame_info['pose'] = dict() # not need for single frame
    frame_info['annos'] = dict(
        name=None,
        truncated=None,
        occluded=None,
        alpha=None,
        bbox=None, # not need for lidar-only
        dimensions=None,
        location=None,
        rotation_y=None,
        index=None,
        group_ids=None,
        camera_id=None,
        difficulty=None,
        num_points_in_gt=None,
    )
    frame_info['sweeps'] = [] # not need for single frame
    if frame_anno is not None:
        frame_anno = frame_anno[frame_anno['num_interior_pts'] > 0]
        cuboid_params = frame_anno.loc[:, list(LABEL_ATTR)].to_numpy()
        cuboid_params = torch.from_numpy(cuboid_params)
        yaw = quat_to_yaw(cuboid_params[:, -4:])
        xyz = cuboid_params[:, :3]
        wlh = cuboid_params[:, [4, 3, 5]]

        # waymo_yaw is equal to yaw
        # corners = cuboid_to_vertices(cuboid_params)
        # c0 = corners[:, 0, :]
        # c4 = corners[:, 4, :]
        # waymo_yaw = torch.atan2(c0[:, 1] - c4[:, 1], c0[:, 0] - c4[:, 0])
        yaw = -yaw - 0.5 * np.pi

        while (yaw < -np.pi).any():
            yaw[yaw < -np.pi] += 2 * np.pi

        while (yaw > np.pi).any():
            yaw[yaw > np.pi] -= 2 * np.pi
        
        # bbox = torch.cat([xyz, wlh, yaw.unsqueeze(1)], dim=1).numpy()
        
        cat = frame_anno['category'].to_numpy().tolist()
        cat = [c.lower().capitalize() for c in cat]
        cat = np.array(cat)

        num_obj = len(cat)

        annos = frame_info['annos']
        annos['name'] = cat
        annos['truncated'] = np.zeros(num_obj, dtype=np.float64)
        annos['occluded'] = np.zeros(num_obj, dtype=np.int64)
        annos['alpha'] = -10 * np.ones(num_obj, dtype=np.float64)
        annos['dimensions'] = wlh.numpy().astype(np.float64)
        annos['location'] = xyz.numpy().astype(np.float64)
        annos['rotation_y'] = yaw.numpy().astype(np.float64)
        annos['index'] = np.arange(num_obj, dtype=np.int32)
        annos['num_points_in_gt'] = frame_anno['num_interior_pts'].to_numpy().astype(np.int32)
    # frame_info['group_ids'] = np.arange(num_obj, dtype=np.int32)
    prefix2split = {'0': 'training', '1': 'training', '2': 'testing'}
    sample_idx = frame_info['sample_idx']
    split = prefix2split[sample_idx[0]]
    abs_save_path = osp.join(output_dir, split, 'velodyne', f'{sample_idx}.bin')
    rel_save_path = osp.join(split, 'velodyne', f'{sample_idx}.bin')
    frame_info['point_cloud']['velodyne_path'] = rel_save_path
    if save_bin:
        save_point_cloud(lidar_frame_path, abs_save_path)
    return frame_info

def save_point_cloud(frame_path, save_path):
    lidar = read_feather(frame_path)
    lidar = lidar.loc[:, ['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
    lidar.tofile(save_path)
    
def prepare(root):
    ts2idx = {}
    ts_list = []
    bin_idx_list = []
    seg_path_list = []
    seg_split_list = []
    assert root.split('/')[-1] == 'sensor'
    splits = ['train', 'val', 'test']
    # splits = ['train', ]
    num_train_samples = 0
    num_val_samples = 0
    num_test_samples = 0

    # 0 for training, 1 for validation and 2 for testing.
    prefixes = [0, 1, 2]

    for i in range(len(splits)):
        split = splits[i]
        prefix = prefixes[i]
        split_root = osp.join(root, split)
        seg_file_list = os.listdir(split_root)
        print(f'num of {split} segments:', len(seg_file_list))
        for seg_idx, seg_name in enumerate(seg_file_list):
            seg_path = osp.join(split_root, seg_name)
            seg_path_list.append(seg_path)
            seg_split_list.append(split)
            assert seg_idx < 1000
            frame_path_list = os.listdir(osp.join(seg_path, 'sensors/lidar/'))
            for frame_idx, frame_path in enumerate(frame_path_list):
                assert frame_idx < 1000
                bin_idx = str(prefix) + str(seg_idx).zfill(3) + str(frame_idx).zfill(3)
                ts = frame_path.split('/')[-1].split('.')[0]
                ts = seg_name + '/' + ts # ts is not unique, so add seg_name
                ts2idx[ts] = bin_idx
                ts_list.append(ts)
                bin_idx_list.append(bin_idx)
        if split == 'train':
            num_train_samples = len(ts_list)
        elif split == 'val':
            num_val_samples = len(ts_list) - num_train_samples
        else:
            num_test_samples = len(ts_list) - num_train_samples - num_val_samples
    # print three num samples
    print('num of train samples:', num_train_samples)
    print('num of val samples:', num_val_samples)
    print('num of test samples:', num_test_samples)

    assert len(ts_list) == len(set(ts_list))
    assert len(bin_idx_list) == len(set(bin_idx_list))
    return ts2idx, seg_path_list, seg_split_list

def main(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, token, num_process):
    for seg_i, seg_path in enumerate(seg_path_list):
        if seg_i % num_process != token:
            continue
        print(f'processing segment: {seg_i}/{len(seg_path_list)}')
        split = seg_split_list[seg_i]
        process_single_segment(seg_path, split, info_list, ts2idx, output_dir, save_bin)


if __name__ == '__main__':

    # please change to your data path
    root = 'data/argo2/argo2_format/sensor'
    output_dir = 'data/argo2/argo_pickle/'
    os.makedirs(output_dir, exist_ok=True)
    # save_bin = True
    save_bin = False
    ts2idx, seg_path_list, seg_split_list = prepare(root)

    num_process = 64
    # num_process = 1
    if num_process > 1:
        with mp.Manager() as manager:
            info_list = manager.list()
            pool = mp.Pool(num_process)
            for token in range(num_process):
                result = pool.apply_async(main, args=(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, token, num_process))
            pool.close()
            pool.join()
            info_list = list(info_list)
    else:
        info_list = []
        main(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, 0, 1)

    assert len(info_list) > 0
    
    train_info = [e for e in info_list if e['sample_idx'][0] == '0']
    val_info = [e for e in info_list if e['sample_idx'][0] == '1']
    test_info = [e for e in info_list if e['sample_idx'][0] == '2']
    trainval_info = train_info + val_info
    assert len(train_info) + len(val_info) + len(test_info) == len(info_list)

    # save info_list in under the output_dir as pickle file
    with open(osp.join(output_dir, 'argo2_infos_train.pkl'), 'wb') as f:
        pkl.dump(train_info, f)

    with open(osp.join(output_dir, 'argo2_infos_val.pkl'), 'wb') as f:
        pkl.dump(val_info, f)

    with open(osp.join(output_dir, 'argo2_infos_trainval.pkl'), 'wb') as f:
        pkl.dump(trainval_info, f)

    with open(osp.join(output_dir, 'argo2_infos_test.pkl'), 'wb') as f:
        pkl.dump(test_info, f)

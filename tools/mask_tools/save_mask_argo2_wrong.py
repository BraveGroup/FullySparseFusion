import mmcv
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import cv2
from shutil import copyfile
import pickle, torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
import json
import time

def load_img_list_argo(info):
    img_list = []
    img_name_list = info['image']
    for img_path in img_name_list:
        img = cv2.imread(img_path)
        img_list.append(img)
    return img_list

def paint_obj(obj, mask, anno_list, obj_id, all_score_map, cam_id, score_thre, img_size,):
    obj_score = obj['obj_score'] #result[0][i][j][-1]
    obj_mask = obj['obj_mask']
    obj_bbox = obj['obj_bbox'] #result[0][i][j][0:4]
    obj_class = obj['obj_class'] 
    if  obj_score > score_thre:
        #compute valid area: whose score is higher
        obj_score_map = np.zeros(img_size)
        obj_score_map[obj_mask] = obj_score
        valid_area = obj_score_map > all_score_map


        ##replace
        all_score_map[valid_area] = obj_score
        mask[valid_area] = obj_id
        anno = {
            'bbox' : obj_bbox.tolist(),
            'score' : obj_score.tolist(),
            'category' : obj_class,
            'cam_id' : cam_id,
            'obj_id' : obj_id,
        }
        anno_list.append(anno)
        obj_id += 1
    return mask, anno_list, obj_id, all_score_map

def collect_obj_list(result):
    obj_dict = {}
    num_classes = 10
    for i in range(num_classes):
        class_obj_list = []
        class_score_list = []
        class_name = nuim_class_names[i]
        class_id = name_to_num_nusc[nuim_class_names[i]]
        if len(result[1][i])>0:
            for j in range(len(result[0][i])):
                obj ={}
                obj['obj_score'] = result[0][i][j][-1]
                obj['obj_mask'] = result[1][i][j]
                obj['obj_bbox'] = result[0][i][j][0:4]
                obj['obj_class'] = class_id
                class_obj_list.append(obj)
                class_score_list.append(result[0][i][j][-1])
        #sort by score
        idx_list = np.argsort(class_score_list).tolist()
        idx_list.reverse()
        class_obj_list_sorted = []
        for idx in idx_list:
            class_obj_list_sorted.append(class_obj_list[idx])
        
        obj_dict[class_name] = class_obj_list_sorted
    return obj_dict

def get_instance_mask(obj_list, img, obj_id, cam_id, score_thre):
    #obj id: begin idx
    img_size = img.shape[0:2]
    mask = np.zeros(img_size)
    all_score_map = np.zeros(img_size)
    anno_list = []

    for obj in obj_list:
        if bbox_only:
            mask, anno_list, obj_id, all_score_map = paint_obj_bbox_only(obj, mask, anno_list, obj_id, all_score_map, cam_id, score_thre, img_size,)
        else:
            mask, anno_list, obj_id, all_score_map = paint_obj(obj, mask, anno_list, obj_id, all_score_map, cam_id, score_thre, img_size,)

    return mask, anno_list, obj_id

def get_score_thre_topk(result_list, topk=250):
    score_list = []
    num_classes = 10
    for result_list_cam in result_list:
        for i in range(num_classes):
            if len(result_list_cam[1][i])>0:
                for j in range(len(result_list_cam[0][i])):
                    score_list.append(result_list_cam[0][i][j][-1].tolist())
    score_list.sort(reverse=True)
    if len(score_list) < topk:
        return 0.
    else:
        return score_list[topk-1]

def save_result_format(result_list, img_list, info):
    """
    save mask and annotation from mask-rcnn output
    """
    obj_id = 1
    score_thre_topk = get_score_thre_topk(result_list, topk=250)
    score_thre = max(score_thre_topk, score_thre_init)

    mask_list = []
    anno_list = []
    for cam_id, result_list_cam in enumerate(result_list):
        mask_list_single_cls = []
        anno_list_single_cls = {}
        obj_dict = collect_obj_list(result_list_cam)
        for cls_name in name_nusc:
            obj_list = obj_dict[cls_name]
            mask_cam, anno_list_cam, obj_id = get_instance_mask(obj_list, img_list[cam_id], obj_id, cam_id, score_thre)
            mask_list_single_cls.append(mask_cam)
            anno_list_single_cls[cls_name] = anno_list_cam
        mask_list.append(mask_list_single_cls)
        anno_list.append(anno_list_single_cls)
    
    sample_idx = info['uuid']
    sample_dir = os.path.join(out_path, sample_idx) 
    os.makedirs(sample_dir, exist_ok=True)

    anno_path = os.path.join(sample_dir, 'anno.json')
    json.dump(anno_list, open(anno_path, 'w'), indent=2)
    for cam_id, single_mask_list in enumerate(mask_list):
        for cls_id, mask in enumerate(single_mask_list):
            mask_path = os.path.join(sample_dir, '{}_{}.png'.format(str(cam_id), name_nusc[cls_id]))
            assert mask.max() < 255, 'for uint8'
            cv2.imwrite(mask_path, mask.astype('uint8'))
    return

def save_data(model, args):
    data_infos = pickle.load(open(info_path, 'rb'))

    infos = data_infos
    num_infos = len(infos)

    for idx in tqdm(range(num_infos)):
        if idx % args.num_gpus != args.split_id:
            continue
        info = infos[idx]
        img_list = load_img_list_argo(info)
        result_list = inference_detector(model, img_list) 

        save_result_format(result_list, img_list, info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--split_id', type=int, default=0)
    args = parser.parse_args()

    #HTC
    config ='/mnt/weka/scratch/yingyan.li/repo/frustum-query-fusion/projects/configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_trainval.py'
    checkpoint = '/mnt/weka/scratch/yingyan.li/repo/frustum-query-fusion/work_dirs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_trainval/epoch_20.pth'
    # config ='mmdetection3d/configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py'
    # checkpoint = 'ckpt/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_trainval/epoch_20.pth' 

    info_path = f'data/argo2/argo_pickle/argo2_infos_{args.split}.pkl'
    out_path = 'data/frustum_mask/AV2'
    os.makedirs(out_path, exist_ok=True)

    score_thre_init = 0.05
    bbox_only = False

    num_classes = 10
    nuim_class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    name_to_num_nusc = {
        'car' : 0, 'truck' : 1, 'construction_vehicle' : 2, 'bus' : 3, 'trailer' : 4, 'barrier' : 5,
        'motorcycle' : 6, 'bicycle' : 7, 'pedestrian' : 8, 'traffic_cone' :9
    }
    name_nusc = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    model = init_detector(config, checkpoint, device=f'cuda:{args.split_id}')

    save_data(model, args)
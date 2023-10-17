import mmcv
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import torch
import cv2
from mmdet3d.core.points.lidar_points import LiDARPoints
import os, time
import numpy as np
from mmcv.parallel import DataContainer as DC
import json, copy
from mmcv.utils import build_from_cfg
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_cpu

import torchvision
from torchvision.transforms.functional import resize as Resize
from torchvision.transforms import InterpolationMode
@PIPELINES.register_module()
class LoadMaskFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                    data_path, 
                    class_names=[
                                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                                    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
                                ], 
                    obj_max_num=250, 
                    is_argo=False, 
                    is_waymo=False
                ):
        self.data_path = data_path
        self.obj_max_num = obj_max_num
        self.class_names = class_names
        self.is_argo = is_argo
        self.is_waymo = is_waymo

    def resize_img(self, results, mask_data, anno, resize_idx=0, resize_shape=[1550, 2048]):
        #resize project mat
        ori_shape = mask_data[resize_idx].shape
        h_resize_factor = resize_shape[0] / ori_shape[0]
        w_resize_factor = resize_shape[1] / ori_shape[1]

        lidar2img = results['lidar2img'][resize_idx]
        lidar2img[0] *= w_resize_factor
        lidar2img[1] *= h_resize_factor
        results['lidar2img'][resize_idx] = lidar2img

        #resize image
        mask_data[resize_idx] = \
            Resize(mask_data[resize_idx].unsqueeze(0), resize_shape, interpolation=InterpolationMode.NEAREST).squeeze(0)
        
        #resize bbox
        for obj_anno in anno[resize_idx]:
            bbox = obj_anno['bbox']
            bbox_resized = [
                bbox[0] * w_resize_factor, 
                bbox[1] * h_resize_factor, 
                bbox[2] * w_resize_factor, 
                bbox[3] * h_resize_factor,
                ]
            obj_anno['bbox'] = bbox_resized
    
    def resize_img_argo(self, results, mask_data, anno, resize_idx=0, resize_shape=[1550, 2048]):
        #resize project mat
        ori_shape = mask_data[resize_idx].shape
        h_resize_factor = resize_shape[0] / ori_shape[0]
        w_resize_factor = resize_shape[1] / ori_shape[1]

        lidar2img = results['lidar2img'][resize_idx]
        lidar2img[0] *= w_resize_factor
        lidar2img[1] *= h_resize_factor
        results['lidar2img'][resize_idx] = lidar2img

        cls_num = len(self.class_names)
        #resize image
        # mask_data[cls_num * resize_idx : cls_num * (resize_idx + 1)] = \
        #     Resize(mask_data[cls_num * resize_idx : cls_num * (resize_idx + 1)].unsqueeze(0), resize_shape, interpolation=InterpolationMode.NEAREST).squeeze(0)
        
        # 循环处理每一类
        for i in range(cls_num * resize_idx, cls_num * (resize_idx + 1)):
            # 对单个类别的数据进行缩放
            resized = Resize(mask_data[i].unsqueeze(0), resize_shape, interpolation=InterpolationMode.NEAREST)
            mask_data[i] = resized.squeeze(0)

        #resize bbox
        for cls_name, obj_annos in anno[resize_idx].items():
            for obj_anno in obj_annos:
                bbox = obj_anno['bbox']
                bbox_resized = [
                    bbox[0] * w_resize_factor, 
                    bbox[1] * h_resize_factor, 
                    bbox[2] * w_resize_factor, 
                    bbox[3] * h_resize_factor,
                ]
                obj_anno['bbox'] = bbox_resized

    def resize_img_waymo(self, results, mask_data, anno, resize_cam_id, resize_idx, resize_shape=[1550, 2048], anno_is_dict=False):
        #resize project mat
        ori_shape = mask_data[resize_idx][0].shape
        h_resize_factor = resize_shape[0] / ori_shape[0]
        w_resize_factor = resize_shape[1] / ori_shape[1]
        
        for cam_id in resize_cam_id:
            lidar2img = results['lidar2img'][cam_id]
            lidar2img[0] *= w_resize_factor
            lidar2img[1] *= h_resize_factor
            results['lidar2img'][cam_id] = lidar2img

        #resize image
        for resize_id in range(resize_idx.start, resize_idx.stop):
            mask_data[resize_id] = \
                Resize(mask_data[resize_id].unsqueeze(0), resize_shape, interpolation=InterpolationMode.NEAREST).squeeze(0)
        
        #resize bbox
        for cam_id in resize_cam_id:
            for cls_name, obj_annos in anno[cam_id].items():
                for obj_anno in obj_annos:
                    bbox = obj_anno['bbox']
                    bbox_resized = [
                        bbox[0] * w_resize_factor, 
                        bbox[1] * h_resize_factor, 
                        bbox[2] * w_resize_factor, 
                        bbox[3] * h_resize_factor,
                        ]
                    obj_anno['bbox'] = bbox_resized
        
        return

    def load_waymo(self, results):
        sample_idx = results['pts_filename'].split('/')[-1].replace('.bin', '')
        sample_dir = os.path.join(self.data_path, sample_idx)
        mask_data = []
        num_cams = 5

        waymo_cls_name = ['vehicle', 'pedestrian', 'cyclist']
        num_cls = len(waymo_cls_name)
        #load image
        for cam_id in range(num_cams):
            for name in waymo_cls_name:
                file_name = f"{cam_id}_{name}.png"
                img_path = os.path.join(sample_dir, file_name)
                img = cv2.imread(img_path, -1)
                mask_data.append(torch.from_numpy(img))
                
        #load anno
        anno_path = os.path.join(sample_dir, 'anno.json')
        anno = json.load(open(anno_path, 'r'))
        
        #resize the image of two back cameras
        self.resize_img_waymo(results, mask_data, anno, resize_cam_id=[3, 4], resize_idx=slice(3 * num_cls, 5 * num_cls), resize_shape=[1280, 1920])

        results['mask_anno'] = self.reorg_anno_multi_cls(anno)
        results['mask_data'] = torch.stack(mask_data, dim=0).reshape(num_cams, len(waymo_cls_name), mask_data[0].shape[0], mask_data[0].shape[1])

        return results

    # def load_argo_old(self, results):
    #     uuid = results['img_info']['uuid']
    #     sample_dir = os.path.join(self.data_path, uuid)
    #     mask_data = []
    #     file_list = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', 'anno.json']
    #     for file in file_list:
    #         if '.png' in file:
    #             img_path = os.path.join(sample_dir, file)
    #             img = cv2.imread(img_path, -1).astype(np.int32)
    #             mask_data.append(torch.from_numpy(img))
    #         elif '.json' in file:
    #             anno_path = os.path.join(sample_dir, file)
    #             anno = json.load(open(anno_path, 'r'))
    #     #resize the image of ring_front_camera
    #     self.resize_img(results, mask_data, anno, resize_idx=0, resize_shape=[1550, 2048]) #check whether mask data change
    #     results['mask_anno'] = self.reorg_anno_single_cls(anno)
    #     results['mask_data'] = torch.stack(mask_data, dim=0).unsqueeze(1)
    #     return results
    
    def load_argo(self, results):
        uuid = results['img_info']['uuid']
        sample_dir = os.path.join(self.data_path, uuid)
        mask_data = []
        num_cams = 7

        #load anno
        anno_path = os.path.join(sample_dir, 'anno.json')
        anno = json.load(open(anno_path, 'r'))

        #load image
        for cam_id in range(num_cams):
            for name in self.class_names:
                file_name = f"{cam_id}_{name}.png"
                img_path = os.path.join(sample_dir, file_name)
                img = cv2.imread(img_path, -1)
                mask_data.append(torch.from_numpy(img))
        
        self.resize_img_argo(results, mask_data, anno, resize_idx=0, resize_shape=[1550, 2048]) #check whether mask data change
        
        results['mask_anno'] = self.reorg_anno_multi_cls(anno)
        results['mask_data'] = torch.stack(mask_data, dim=0).reshape(num_cams, len(self.class_names), mask_data[0].shape[0], mask_data[0].shape[1])

        return results

    def load_nusc(self, results):
        sample_idx = results['sample_idx']
        sample_dir = os.path.join(self.data_path, sample_idx)
        mask_data = []
        num_cams = 6

        #load image
        for cam_id in range(num_cams):
            for name in self.class_names:
                file_name = f"{cam_id}_{name}.png"
                img_path = os.path.join(sample_dir, file_name)
                img = cv2.imread(img_path, -1)
                mask_data.append(torch.from_numpy(img))
        
        #load anno
        anno_path = os.path.join(sample_dir, 'anno.json')
        anno = json.load(open(anno_path, 'r'))

        results['mask_anno'] = self.reorg_anno_multi_cls(anno)
        results['mask_data'] = torch.stack(mask_data, dim=0).reshape(num_cams, len(self.class_names), mask_data[0].shape[0], mask_data[0].shape[1])

        return results

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.is_argo:
            results = self.load_argo(results)
        elif self.is_waymo:
            results = self.load_waymo(results)
        else:
            results = self.load_nusc(results)
        return results
    
    def pad_tensor(self, data_list):
        num_obj = len(data_list)
        res_num = self.obj_max_num - num_obj
        data_tensor = torch.tensor(data_list)
        if len(data_tensor.shape) > 1:
            num_chan = data_tensor.shape[1]
            pad_tensor = data_tensor.new_zeros((res_num, num_chan))
        else:
            pad_tensor = data_tensor.new_zeros((res_num,))
        return torch.cat((data_tensor, pad_tensor), dim=0)

    def reorg_anno_single_cls(self, annos):
        result_list = []
        for anno in annos:
            for obj_anno in anno:
                obj_anno['bbox'].extend(
                    [
                        obj_anno['score'],
                        obj_anno['category'],
                        obj_anno['cam_id'],
                        obj_anno['obj_id']
                    ]
                )
                result_list.append(obj_anno['bbox'])
        
        num_obj = len(result_list)
        if num_obj == 0:
            anno_tensor = torch.zeros((self.obj_max_num, 8))
        else:
            anno_tensor = torch.tensor(result_list) #num_obj, 8
            anno_tensor = self.pad_tensor(result_list)
        valid_mask = torch.zeros((self.obj_max_num, 1), dtype=torch.bool)
        valid_mask[:num_obj] = True
        final_tensor = torch.cat([anno_tensor, valid_mask], dim=-1)
        ## num_obj, 9
        ##[0: 4],    4,        5,      6,      7,         8,
        ## bbox, score, category, cam_id, obj_id, valid_flag
        return final_tensor
    
    def reorg_anno_multi_cls(self, annos):
        """
        output: K, 9 tensor
        """
        result_list = []
        obj_id_list = []
        for anno_cam in annos:
            for cls_name , cls_annos in anno_cam.items():
                for obj_anno in cls_annos:
                    obj_id_list.append(obj_anno['obj_id'])
                    obj_anno['bbox'].extend(
                        [
                            obj_anno['score'],
                            obj_anno['category'],
                            obj_anno['cam_id'],
                            obj_anno['obj_id']
                        ]
                    )
                    result_list.append(obj_anno['bbox'])

        _, sort_idx = torch.sort(torch.tensor(obj_id_list))
        result_list_new = []
        for obj_idx in sort_idx:
            result_list_new.append(result_list[obj_idx])
        result_list = result_list_new

        num_obj = len(result_list)
        if num_obj == 0:
            anno_tensor = torch.zeros((self.obj_max_num, 8))
        else:
            anno_tensor = torch.tensor(result_list) #num_obj, 8
            anno_tensor = self.pad_tensor(result_list)
        valid_mask = torch.zeros((self.obj_max_num, 1), dtype=torch.bool)
        valid_mask[:num_obj] = True
        final_tensor = torch.cat([anno_tensor, valid_mask], dim=-1)
        ## num_obj, 9
        ##[0: 4],    4,        5,      6,      7,         8,
        ## bbox, score, category, cam_id, obj_id, valid_flag
        return final_tensor


@PIPELINES.register_module()
class SaveNoAugPoints(object):
    def __init__(self,):
        return

    def __call__(self, results):
        points = results['points'].tensor.clone()
        results['points'].tensor \
            = torch.cat([results['points'].tensor, points[:, :3]], -1)
        if 'gt_bboxes_3d' in results.keys():
            results['no_aug_gt_bboxes_3d'] = results['gt_bboxes_3d'].clone()
            results['no_aug_gt_labels_3d'] = torch.from_numpy(results['gt_labels_3d'])
        return results

@PIPELINES.register_module()
class MyObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        if 'no_aug_gt_bboxes_3d' in input_dict.keys():
            if 'copy_mask' in input_dict.keys():
                copy_mask = input_dict['copy_mask']
                mask = mask[~copy_mask]
            no_aug_gt_bboxes_3d = input_dict['no_aug_gt_bboxes_3d'][mask]
            no_aug_gt_labels_3d = input_dict['no_aug_gt_labels_3d'][mask]
            no_aug_gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
            input_dict['no_aug_gt_bboxes_3d'] = DC(no_aug_gt_bboxes_3d, cpu_only=True)
            input_dict['no_aug_gt_labels_3d'] = DC(no_aug_gt_labels_3d, cpu_only=True)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@PIPELINES.register_module()
class MyObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, pad_value=10000, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.pad_value = pad_value

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def encode_sampled_pts(self, sampled_pts, gt_bboxes_3d, gt_labels_3d):
        """cat last 3 dimension as [10000, 10000+gt_id, gt_labels]
        
        """
        match_mat = points_in_boxes_cpu(sampled_pts.tensor[:, :3], gt_bboxes_3d[:, :7])
        max_val, pts_gt_id = match_mat.max(0)
        pts_mask = max_val > 0
        #> raw gt id
        pts_gt_label = gt_labels_3d[pts_gt_id].float()
        pad_value_tensor = torch.ones_like(pts_gt_id) * self.pad_value
        gt_tensor = torch.cat([
                        pad_value_tensor.unsqueeze(-1), 
                        pts_gt_id.unsqueeze(-1) + self.pad_value, 
                        pts_gt_label.unsqueeze(-1)], dim=-1)
        sampled_pts.tensor = torch.cat([sampled_pts.tensor, gt_tensor], dim=1)
        return sampled_pts[pts_mask]

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            
            copy_mask = torch.zeros((gt_labels_3d.shape[0]), dtype=torch.bool)
            copy_num = sampled_gt_labels.shape[0]
            copy_mask[-copy_num:] = True
            input_dict['copy_mask'] = copy_mask
            sampled_points_plus = self.encode_sampled_pts(
                                        sampled_points, 
                                        torch.from_numpy(sampled_gt_bboxes_3d), 
                                        torch.from_numpy(sampled_gt_labels)
                                    )
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points_plus])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict


@PIPELINES.register_module()
class NormalizePoints(object):

    def __init__(self,
                 std=[255,],
                 mean=[0,],
                 dims=[3,]):
        self.dims = dims
        self.std = std
        self.mean = mean

    def __call__(self, input_dict):
        """Call function to jitter all the points in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each point, \
                'points' key is updated in the result dict.
        """
        points = input_dict['points']
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)

        points.tensor[:, self.dims] = (points.tensor[:, self.dims] - mean[None, :]) / std[None, :]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(jitter_std={self.jitter_std},'
        repr_str += f' clip_range={self.clip_range})'
        return repr_str


@PIPELINES.register_module()
class MyLoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 virtual_path=None):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        self.virtual_path = virtual_path

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def load_vpts(self, pts_filename):
        token = pts_filename.split('/')
        vpts_filename = os.path.join(self.virtual_path, token[-3], token[-2]+'_VIRTUAL', token[-1]+'.pkl.npy')

        #load pts
        if os.path.isfile(vpts_filename):
            data_dict = np.load(vpts_filename, allow_pickle=True).item()
            vpts = data_dict['virtual_points']
            num_vpts = vpts.shape[0]
            vpts_final = np.ones((num_vpts, 5)) * -1
            vpts_final[:, :3] = vpts[:, :3]
        else:
            vpts_final = np.ones((0, 5)) * -1
            # print('No vpts')
        
        return vpts_final

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.virtual_path is not None:
            vpts = self.load_vpts(pts_filename)
            final_points = np.vstack([points, vpts])
            points = final_points

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@PIPELINES.register_module()
class MyLoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 virtual_path=None,
                 ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.virtual_path = virtual_path

    def load_vpts(self, pts_filename):
        token = pts_filename.split('/')
        vpts_filename = os.path.join(self.virtual_path, token[-3], token[-2]+'_VIRTUAL', token[-1]+'.pkl.npy')

        #load pts
        # if 'n015-2018-07-18-11-50-34+0800__LIDAR_TOP__1531886039449152.pcd.bin' in vpts_filename:
        #     print()
        if os.path.isfile(vpts_filename):
            data_dict = np.load(vpts_filename, allow_pickle=True).item()
            vpts = data_dict['virtual_points']
            num_vpts = vpts.shape[0]
            vpts_final = np.ones((num_vpts, 5)) * -1
            vpts_final[:, :3] = vpts[:, :3]
        else:
            vpts_final = np.ones((0, 5)) * -1
            # print('No vpts')
        
        return vpts_final
        
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.virtual_path is not None:
                    vpts_sweep = self.load_vpts(sweep['data_path'])
                    points_sweep = np.vstack([points_sweep, vpts_sweep])
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


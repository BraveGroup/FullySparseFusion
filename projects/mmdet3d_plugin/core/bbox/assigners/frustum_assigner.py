import torch
import cv2, os
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet3d.core import LiDARInstance3DBoxes as LB
from mmdet.core import (multi_apply, build_assigner, build_sampler, build_bbox_coder, reduce_mean)
from mmdet.core import PseudoSampler
from shapely.geometry import MultiPoint
from shapely.geometry import box as shapely_box
import time
import copy
@BBOX_ASSIGNERS.register_module()
class FrustumAssigner(BaseAssigner):
    def __init__(self,
                assigner_2d=None,
                assigner_3d=None,
                assigner_dist=None,
                num_cams=6,
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                ],
                tasks=None,
                vis_dir=None,
                ignore_bev_dist=None,
                is_frustum=False,
                ):
        self.assigner_2d = build_assigner(assigner_2d) if assigner_2d is not None else None
        self.assigner_3d = build_assigner(assigner_3d) if assigner_3d is not None else None
        self.sampler = PseudoSampler()
        self.num_cams = num_cams
        self.class_names = class_names
        self.tasks=tasks
        self.vis_dir=vis_dir
        self.ignore_bev_dist = ignore_bev_dist
        self.is_frustum = is_frustum
        if assigner_dist is not None:
            self.assigner_dist = build_assigner(assigner_dist)
        else:
            self.assigner_dist = None
    
    def merge_3d_2d_assign_result(self, assign_result_3d, assign_result_2d):
        """if 2d assign some objs that 3d not assign, replace it.
        vis the 2d supplement.
        """     
        num_gts = assign_result_3d.num_gts

        neg_3d_mask = assign_result_3d.gt_inds <= 0
        pos_2d_mask = assign_result_2d.gt_inds > 0
        replace_mask = neg_3d_mask & pos_2d_mask

        assigned_gt_inds = assign_result_3d.gt_inds.clone()
        assigned_labels = assign_result_3d.labels.clone()

        assigned_gt_inds[replace_mask] = assign_result_2d.gt_inds[replace_mask]
        assigned_labels[replace_mask] = assign_result_2d.labels[replace_mask] 
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    def assign(self,
               preds_2d,
               no_aug_gt_bboxes_3d,
               no_aug_gt_labels_3d,
               cluster_xyz,
               old_cluster_logits,
               old_reg_preds,
               gt_bboxes_3d,
               gt_labels_3d,
               img_metas,
               task_id,
               ):
        """
        1. if cluster_xyz in 3d bbox, assign it
        2. the rest preds, use 2d assign. 2d assign only allows not too far obj
        """
        no_aug_num = no_aug_gt_labels_3d.shape[0]
        
        if self.assigner_3d:
            assign_result_3d = self.assigner_3d.assign(
                cluster_xyz,
                old_reg_preds,
                gt_bboxes_3d.tensor[:, :7] if not self.is_frustum else gt_bboxes_3d.tensor[:no_aug_num, :7],
                gt_labels=gt_labels_3d if not self.is_frustum else gt_labels_3d[:no_aug_num],
            )

        if self.assigner_2d:
            assign_result_2d = self.assign_2d(
                preds_2d,
                no_aug_gt_bboxes_3d,
                no_aug_gt_labels_3d,
                gt_bboxes_3d,
                gt_labels_3d,
                img_metas,
                task_id,
            )
        
        if self.assigner_3d and self.assigner_2d:
            assign_result_final = self.merge_3d_2d_assign_result(assign_result_3d, assign_result_2d)
        elif self.assigner_3d and not self.assigner_2d:
            assign_result_final = assign_result_3d
            assign_result_2d = assign_result_3d
        elif not self.assigner_3d and self.assigner_2d:
            assign_result_final = assign_result_2d
            assign_result_3d = assign_result_2d

        if self.assigner_dist is not None:
            assign_result_dist = self.assigner_dist.assign(
                                                    cluster_xyz,
                                                    old_cluster_logits,
                                                    gt_bboxes_3d, #tensor
                                                    gt_labels_3d,
                                                )
            assign_result_final = self.merge_3d_2d_assign_result(assign_result_final, assign_result_dist)
        return assign_result_final, assign_result_3d, assign_result_2d

    def assign_2d(self,
               preds_2d,
               gt_bboxes_3d,
               gt_labels_3d,
               augged_gt_bboxes_3d,
               augged_gt_labels_3d,
               img_metas,
               task_id,
               ):
        num_preds = preds_2d.shape[0]
        device = gt_labels_3d.device
        num_gts = gt_bboxes_3d.tensor.shape[0]
        assigned_gt_inds = gt_labels_3d.new_zeros((num_preds, ), dtype=torch.long) # 0 indicates assign to backgroud
        assigned_labels = gt_labels_3d.new_full((num_preds, ), -1, dtype=torch.long)

        num_gts = gt_bboxes_3d.tensor.size(0)
        if num_gts == 0 or num_preds == 0:
            # No ground truth or cluster, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
                
        for cam_id in range(self.num_cams):
            lidar2img = torch.tensor(img_metas['lidar2img'][cam_id], device=device).float()
            gt_bboxes_2d, gt_idx_tensor = get_gt_bboxes_2d(gt_bboxes_3d, lidar2img)
            if gt_bboxes_2d.shape[0] == 0:
                continue
            else:
                cam_id_of_preds = preds_2d[:, 6]
                cam_mask = cam_id_of_preds == cam_id
                preds_2d_img = preds_2d[cam_mask]

                dt_bboxes_2d = preds_2d_img[:, :4]
                assign_result_img = self.assigner_2d.assign(dt_bboxes_2d, gt_bboxes_2d)
                sample_result = self.sampler.sample(assign_result_img, dt_bboxes_2d, gt_bboxes_2d)

                if self.vis_dir is not None:
                    self.vis_2d_assign(dt_bboxes_2d, gt_bboxes_2d, sample_result, cam_id, img_metas, task_id, obj_ids=preds_2d_img[:, 7])

                pos_inds = sample_result.pos_inds
                pos_assigned_gt_inds = gt_idx_tensor[sample_result.pos_assigned_gt_inds]

                assigned_gt_inds_img = assigned_gt_inds[cam_mask]
                assigned_labels_img = assigned_labels[cam_mask]

                assigned_gt_inds_img[pos_inds] = pos_assigned_gt_inds + 1
                assigned_labels_img[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]

                assigned_gt_inds[cam_mask] = assigned_gt_inds_img
                assigned_labels[cam_mask] = assigned_labels_img

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
    
    def vis_2d_assign(self, dt_bboxes_2d, gt_bboxes_2d, sample_result, cam_id, img_metas, task_id, obj_ids):
        """
        task-specific
        1. assigned gt/dt pair
        2. missed gt
        3. redundancy dt
        """
        print('vis 2d assign')
        img_path = img_metas['filename'][cam_id]
        sample_idx = img_metas['sample_idx']
        img = cv2.imread(img_path, -1)
        vis_dir = self.vis_dir
        task_name = self.tasks[task_id]['class_names']
        out_dir = os.path.join(vis_dir, sample_idx, 'cam_' + str(cam_id), str(task_name))
        missed_gt_dir = os.path.join(vis_dir, '2d_assign_missed_gt')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(missed_gt_dir, exist_ok=True)

        pos_inds = sample_result.pos_inds
        pos_assigned_gt_inds = sample_result.pos_assigned_gt_inds

        #assigned pair
        obj_id_assigned = obj_ids[pos_inds].cpu().numpy()
        dt_assigned = dt_bboxes_2d[pos_inds].int().cpu().numpy()
        gt_assigned = gt_bboxes_2d[pos_assigned_gt_inds].int().cpu().numpy()
        for dt, gt, obj_id, gt_id in zip(dt_assigned, gt_assigned, obj_id_assigned, pos_assigned_gt_inds):
            one_img = img.copy()
            cv2.rectangle(one_img, pt1=dt[:2], pt2=dt[2:], color=(255, 0, 0))
            cv2.rectangle(one_img, pt1=gt[:2], pt2=gt[2:], color=(0, 255, 0))
            cv2.imwrite(os.path.join(out_dir, f'gt_{int(gt_id)}_obj_{int(obj_id)}.png'), one_img)

        #missed gt
        all_gt_idx = list(range(gt_bboxes_2d.shape[0]))
        used_gt_idx = pos_assigned_gt_inds.unique().cpu().numpy().tolist()
        missed_gt_idx = list(set(all_gt_idx).difference(set(used_gt_idx)))
        missed_gt_img = img.copy()
        for idx in missed_gt_idx:
            gt = gt_bboxes_2d[idx].int().cpu().numpy()
            cv2.rectangle(missed_gt_img, pt1=gt[:2], pt2=gt[2:], color=(0, 255, 0))
        if len(missed_gt_idx) > 0:
            cv2.imwrite(os.path.join(out_dir, 'missed_gt.png'), missed_gt_img)
            cv2.imwrite(os.path.join(missed_gt_dir, str(sample_idx) + '.png'), missed_gt_img)

        #all dt
        all_dt_idx = list(range(dt_bboxes_2d.shape[0]))
        dt_img = img.copy()
        for idx in all_dt_idx:
            dt = dt_bboxes_2d[idx].int().cpu().numpy()
            cv2.rectangle(dt_img, pt1=dt[:2], pt2=dt[2:], color=(255, 0, 0))
        if len(all_dt_idx) > 0:
            cv2.imwrite(os.path.join(out_dir, 'all_dt.png'), dt_img)
        
        #unused dt
        all_dt_idx = list(range(dt_bboxes_2d.shape[0]))
        used_dt_idx = pos_inds.unique().cpu().numpy().tolist()
        missed_dt_idx = list(set(all_dt_idx).difference(set(used_dt_idx)))
        missed_dt_img = img.copy()
        for idx in missed_dt_idx:
            dt = dt_bboxes_2d[idx].int().cpu().numpy()
            cv2.rectangle(missed_dt_img, pt1=dt[:2], pt2=dt[2:], color=(255, 0, 0))
        if len(missed_dt_idx) > 0:
            cv2.imwrite(os.path.join(out_dir, 'missed_dt.png'), missed_dt_img)

def get_gt_bboxes_2d(gt_bboxes_3d, lidar2img):
    #img size must be (1600, 900)
    gt_2d_prj, valid_mask = prj_lidar_bbox3d_on_img(gt_bboxes_3d, lidar2img)

    device = gt_bboxes_3d.tensor.device
    num_gts = gt_bboxes_3d.tensor.shape[0]
    gt_idx = torch.tensor(
                    list(range(num_gts)), 
                    dtype=torch.long, 
                    device=device,
            )

    gt_2d_prj = gt_2d_prj[valid_mask]
    gt_idx = gt_idx[valid_mask]

    num_gts = gt_2d_prj.shape[0]

    gt_bboxes_2d_list = []
    gt_idx_list = []
    for i in range(num_gts):
        gt_bboxes_2d = post_process_coords(gt_2d_prj[i])
        if (gt_bboxes_2d is not None):
            gt_bboxes_2d_list.append(gt_bboxes_2d)
            gt_idx_list.append(gt_idx[i])
    gt_bboxes_2d_tensor = torch.tensor(gt_bboxes_2d_list, device=device)
    gt_idx_tensor = torch.tensor(gt_idx_list, device=device)
    return gt_bboxes_2d_tensor, gt_idx_tensor


def prj_lidar_bbox3d_on_img(bboxes3d,
                            lidar2img_rt,
                            ):
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = torch.cat(
        [corners_3d.reshape(-1, 3),
        corners_3d.new_ones((num_bbox * 8, 1))], dim=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    valid_mask = pts_2d[:, 2] > 1e-5
    valid_mask = valid_mask.reshape(num_bbox, 8)
    valid_mask_bbox = valid_mask.sum(-1) > 0

    pts_2d[:, 2] = torch.clip(pts_2d[:, 2], min=1e-5, max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    num_bbox = valid_mask_bbox.sum()
    return imgfov_pts_2d, valid_mask_bbox

def post_process_coords(
        corner_coords, 
        imsize = (1600, 900)
    ):
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = shapely_box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = torch.tensor(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

### 3rd assign ###

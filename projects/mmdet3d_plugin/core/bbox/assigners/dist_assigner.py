import torch
import cv2, os
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet3d.core import LiDARInstance3DBoxes as LB

@BBOX_ASSIGNERS.register_module()
class DistAssigner(BaseAssigner):
    def __init__(self, assign_tasks, class_names, max_dist):
        self.tasks = assign_tasks
        self.num_tasks = len(assign_tasks)
        self.class_names = class_names
        self.max_dist = max_dist
        return 

    def assign(self,
            cluster_xyz,
            cluster_logits,
            gt_bboxes_3d, #tensor
            gt_labels,
            ):
        gt_bboxes_3d = gt_bboxes_3d.tensor
        _, cluster_labels = cluster_logits.max(-1)

        assign_result_list = []
        for task_id in range(self.num_tasks):
            gt_bboxes_3d_single, gt_labels_single = self.modify_gt_for_single_task([gt_bboxes_3d], [gt_labels], task_id)
            cluster_xyz_single, cluster_labels_single = self.modify_gt_for_single_task([cluster_xyz], [cluster_labels], task_id)
            assign_result_single = self.assign_by_dist_single(cluster_xyz_single[0], gt_bboxes_3d_single[0], gt_labels_single[0], task_id)
            assign_result_list.append(assign_result_single)

        assign_result = self.combine_assign_result(cluster_labels, gt_labels, assign_result_list)
        return assign_result
                
    def combine_assign_result(self, cluster_labels, gt_labels, assign_result_list):
        num_gts = gt_labels.shape[0]
        gt_idx_tensor = torch.tensor(list(range(num_gts)), dtype=torch.long, device=gt_labels.device)
        assigned_gt_inds_all = torch.zeros_like(cluster_labels)
        assigned_labels_all = torch.zeros_like(cluster_labels)

        for task_id, assign_ruslt in enumerate(assign_result_list):
            class_names_this_task = self.tasks[task_id]['class_names']
            cls_mask_gt = torch.zeros_like(gt_labels).bool()
            cls_mask_pred = torch.zeros_like(cluster_labels).bool()

            label_mapping_list = []
            for i, name in enumerate(class_names_this_task):
                cls_id = self.class_names.index(name)
                this_cls_mask_gt = gt_labels == cls_id
                this_cls_mask_pred = cluster_labels == cls_id
                cls_mask_gt = cls_mask_gt | this_cls_mask_gt
                cls_mask_pred = cls_mask_pred | this_cls_mask_pred
                label_mapping_list.append(cls_id)
            label_mapping_list.append(-1)#map -1 to -1
            label_mapping_tensor = torch.tensor(label_mapping_list, dtype=torch.long, device=gt_labels.device)

            pos_assign_mask = assign_ruslt.gt_inds > 0
            cluster_assigned_gt_inds = torch.zeros_like(assign_ruslt.gt_inds)
            cluster_assigned_gt_inds[pos_assign_mask] = gt_idx_tensor[cls_mask_gt][assign_ruslt.gt_inds[pos_assign_mask] - 1] + 1

            assigned_gt_inds_all[cls_mask_pred] = cluster_assigned_gt_inds
            assigned_labels_all[cls_mask_pred] = label_mapping_tensor[assign_ruslt.labels]

        return AssignResult(num_gts, assigned_gt_inds_all, None, labels=assigned_labels_all)
    
    def assign_by_dist_single(self,
                      cluster_xyz,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      task_id,
                      ):
        """Generate targets of vote head for single batch.

        """

        num_cluster = cluster_xyz.size(0)
        num_gts = gt_bboxes_3d.size(0)

        # initialize as all background
        assigned_gt_inds = cluster_xyz.new_zeros((num_cluster, ), dtype=torch.long) # 0 indicates assign to backgroud
        assigned_labels = cluster_xyz.new_full((num_cluster, ), -1, dtype=torch.long)

        if num_gts == 0 or num_cluster == 0:
            # No ground truth or cluster, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        gt_centers = gt_bboxes_3d[None, :, :2]
        pd_xy = cluster_xyz[None, :, :2]
        dist_mat = torch.cdist(pd_xy, gt_centers).squeeze(0)
        max_dist_per_cls = self.max_dist[task_id]
        min_dist_v, matched_gt_inds = torch.min(dist_mat, dim=1)

        max_dist_per_cls = torch.tensor(max_dist_per_cls, dtype=torch.float, device=pd_xy.device)
        matched_gt_labels = gt_labels_3d[matched_gt_inds]
        # assert matched_gt_labels.max().item() < max_dist_per_cls
        matched_max_dist = max_dist_per_cls[matched_gt_labels]

        matched_gt_inds[min_dist_v >= matched_max_dist] = -1
        pos_cluster_mask = matched_gt_inds > -1

        if pos_cluster_mask.any():
            assigned_gt_inds[pos_cluster_mask] = matched_gt_inds[pos_cluster_mask] + 1
            assigned_labels[pos_cluster_mask] = gt_labels_3d[matched_gt_inds[pos_cluster_mask]]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    def modify_gt_for_single_task(self, gt_bboxes_3d, gt_labels_3d, task_id):
        out_bboxes_list, out_labels_list = [], []
        for gts_b, gts_l in zip(gt_bboxes_3d, gt_labels_3d):
            out_b, out_l = self.modify_gt_for_single_task_single_sample(gts_b, gts_l, task_id)
            out_bboxes_list.append(out_b)
            out_labels_list.append(out_l)
        return out_bboxes_list, out_labels_list
    
    def modify_gt_for_single_task_single_sample(self, gt_bboxes_3d, gt_labels_3d, task_id):
        assert gt_bboxes_3d.size(0) == gt_labels_3d.size(0)
        if gt_labels_3d.size(0) == 0:
            return gt_bboxes_3d, gt_labels_3d
        assert (gt_labels_3d >= 0).all() # I don't want -1 in gt_labels_3d

        class_names_this_task = self.tasks[task_id]['class_names']
        num_classes_this_task = len(class_names_this_task)
        out_gt_bboxes_list = []
        out_labels_list = []
        for i, name in enumerate(class_names_this_task):
            cls_id = self.class_names.index(name)
            this_cls_mask = gt_labels_3d == cls_id
            out_gt_bboxes_list.append(gt_bboxes_3d[this_cls_mask])
            out_labels_list.append(gt_labels_3d.new_ones(this_cls_mask.sum()) * i)
        out_gt_bboxes_3d = torch.cat(out_gt_bboxes_list)
        out_labels = torch.cat(out_labels_list, dim=0)
        if len(out_labels) > 0:
            assert out_labels.max().item() < num_classes_this_task
        return out_gt_bboxes_3d, out_labels
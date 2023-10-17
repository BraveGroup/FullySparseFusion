import torch
import cv2, os
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet3d.core import LiDARInstance3DBoxes as LB
from mmdet.core import build_bbox_coder

@BBOX_ASSIGNERS.register_module()
class PointInBoxAssigner(BaseAssigner):
    def __init__(self, extra_height=0.0):
        self.extra_height = extra_height

    def assign(self,
            cluster_xyz,
            old_reg_preds,
            gt_bboxes_3d,
            gt_labels,
            ):
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

        enlarged_box = self.enlarge_box_height(gt_bboxes_3d[:, :7])
        #inbox_inds : the box idx of N pts, shape [N]
        inbox_inds = enlarged_box.points_in_boxes(cluster_xyz).long()
        pos_cluster_mask = inbox_inds > -1

        if pos_cluster_mask.any():
            assigned_gt_inds[pos_cluster_mask] = inbox_inds[pos_cluster_mask] + 1
            assigned_labels[pos_cluster_mask] = gt_labels[inbox_inds[pos_cluster_mask]]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
    
    def enlarge_box_height(self, bbox):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = bbox.clone()
        enlarged_boxes[:, 5] += self.extra_height * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= self.extra_height
        return LB(enlarged_boxes[:, :7])
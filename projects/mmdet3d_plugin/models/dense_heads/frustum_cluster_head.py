import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F

from mmdet3d.models import builder
from mmdet3d.models.builder import build_loss
from projects.mmdet3d_plugin.ops import build_mlp
from mmdet3d.core import AssignResult, PseudoSampler, xywhr2xyxyr, box3d_multiclass_nms, bbox_overlaps_3d, LiDARInstance3DBoxes
from mmdet.core import build_bbox_coder, build_assigner, multi_apply, reduce_mean
from mmdet.models import HEADS

from .sparse_cluster_head_v2 import SparseClusterHeadV2
import copy
import os, cv2, shutil


@HEADS.register_module()
class FrustumClusterHead(SparseClusterHeadV2):

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 loss_cls,
                 loss_center,
                 loss_size,
                 loss_rot,
                 in_channel,
                 shared_mlp_dims,
                 tasks,
                 class_names,
                 common_attrs,
                 num_cls_layer,
                 cls_hidden_dim,
                 separate_head,
                 cls_mlp=None,
                 reg_mlp=None,
                 iou_mlp=None,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 norm_cfg=dict(type='LN'),
                 loss_iou=None,
                 act='relu',
                 corner_loss_cfg=None,
                 enlarge_width=None,
                 as_rpn=False,
                 init_cfg=None,
                 shared_dropout=0,
                 loss_vel=None,
                 assigner=None,
                 num_objs=250,
                 vis_dir=None,
                 use_one_to_one=False,
                 ):
        super().__init__(
            num_classes,
            bbox_coder,
            loss_cls,
            loss_center,
            loss_size,
            loss_rot,
            in_channel,
            shared_mlp_dims,
            tasks,
            class_names,
            common_attrs,
            num_cls_layer,
            cls_hidden_dim,
            separate_head,
            cls_mlp,
            reg_mlp,
            iou_mlp,
            train_cfg,
            test_cfg,
            norm_cfg,
            loss_iou,
            act,
            corner_loss_cfg,
            enlarge_width,
            as_rpn,
            init_cfg,
            shared_dropout,
            loss_vel,
        )
        self.assigner = build_assigner(assigner)
        self.sampler = PseudoSampler()
        self.num_objs = num_objs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.task_info = {}
        self.vis_dir = vis_dir
        self.use_one_to_one = use_one_to_one


    @force_fp32(apply_to=('cls_logits', 'reg_preds',))
    def loss(
        self,
        cls_logits,
        reg_preds,
        cluster_xyz,
        cluster_inds,
        no_aug_gt_bboxes_3d,
        no_aug_gt_labels_3d,
        gt_bboxes_3d,
        gt_labels_3d,
        preds_2d=None,
        img_metas=None,
        iou_logits=None,
        old_cls_logits=None,
        old_reg_preds=None,
        gt_bboxes_ignore=None,
        ):
        assert isinstance(cls_logits, list)
        assert isinstance(reg_preds, list)
        assert len(cls_logits) == len(reg_preds) == len(self.tasks)
        all_task_losses = {}
        self.task_info = {}
        for i in range(len(self.tasks)):
            losses_this_task = self.loss_single_task(
                i,
                cls_logits[i],
                reg_preds[i],
                cluster_xyz, #191, 3
                cluster_inds, #191, 3
                no_aug_gt_bboxes_3d,
                no_aug_gt_labels_3d,
                gt_bboxes_3d,
                gt_labels_3d,
                preds_2d,
                img_metas,
                iou_logits[i] if iou_logits is not None else None,
                old_cls_logits[i] if old_cls_logits is not None else None,
                old_reg_preds[i]  if old_reg_preds is not None else None,
            )
            all_task_losses.update(losses_this_task)
        return all_task_losses

    def loss_single_task(
            self,
            task_id,
            cls_logits,
            reg_preds,
            cluster_xyz,
            cluster_inds,
            no_aug_gt_bboxes_3d,
            no_aug_gt_labels_3d,
            gt_bboxes_3d,
            gt_labels_3d,
            preds_2d,
            img_metas,
            iou_logits=None,
            old_cls_logits=None,
            old_reg_preds=None,
        ):
        # assert task_id == 0
        #TODO: if use 2d-3d hybrid assign, make sure the order don't change in modify_gt_for_single_task function
        no_aug_gt_bboxes_3d, no_aug_gt_labels_3d = self.modify_gt_for_single_task(no_aug_gt_bboxes_3d, no_aug_gt_labels_3d, task_id)
        gt_bboxes_3d, gt_labels_3d = self.modify_gt_for_single_task(gt_bboxes_3d, gt_labels_3d, task_id)
        
        num_total_samples = len(reg_preds)
        batch_size = len(gt_bboxes_3d)

        num_task_classes = len(self.tasks[task_id]['class_names'])
        targets = self.get_targets(
                            num_task_classes, 
                            no_aug_gt_bboxes_3d,
                            no_aug_gt_labels_3d,
                            gt_bboxes_3d, 
                            gt_labels_3d, 
                            preds_2d, 
                            cluster_xyz,
                            cluster_inds,
                            task_id=task_id,
                            img_metas_list=img_metas,
                            reg_preds=reg_preds,
                            new_cls_logits=cls_logits,
                            old_cls_logits=old_cls_logits,
                            old_reg_preds=old_reg_preds,
                        )
        labels, label_weights, bbox_targets, bbox_weights, iou_labels = targets

        cls_avg_factor = num_total_samples * 1.0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                bbox_weights.new_tensor([cls_avg_factor]))

        if cls_logits.shape[0] > 0:
            loss_cls = self.loss_cls(
                cls_logits, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = cls_logits.sum() * 0.0

        # regression loss
        pos_inds = ((labels >= 0) & (labels < num_task_classes)).nonzero(as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_reg_preds = reg_preds[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        reg_avg_factor = num_pos * 1.0
        if self.sync_reg_avg_factor:
            reg_avg_factor = reduce_mean(
                bbox_weights.new_tensor([reg_avg_factor]))

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            
            loss_center = self.loss_center(
                pos_reg_preds[:, :3],
                pos_bbox_targets[:, :3],
                pos_bbox_weights[:, :3],
                avg_factor=reg_avg_factor)
            loss_size = self.loss_size(
                pos_reg_preds[:, 3:6],
                pos_bbox_targets[:, 3:6],
                pos_bbox_weights[:, 3:6],
                avg_factor=reg_avg_factor)
            loss_rot = self.loss_rot(
                pos_reg_preds[:, 6:8],
                pos_bbox_targets[:, 6:8],
                pos_bbox_weights[:, 6:8],
                avg_factor=reg_avg_factor)
            if self.loss_vel is not None:
                loss_vel = self.loss_vel(
                    pos_reg_preds[:, 8:10],
                    pos_bbox_targets[:, 8:10],
                    pos_bbox_weights[:, 8:10],
                )
        else:
            loss_center = pos_reg_preds.sum() * 0
            loss_size = pos_reg_preds.sum() * 0
            loss_rot = pos_reg_preds.sum() * 0
            if self.loss_vel is not None:
                loss_vel = pos_reg_preds.sum() * 0

        losses = dict(
            loss_cls=loss_cls,
            loss_center=loss_center,
            loss_size=loss_size,
            loss_rot=loss_rot,
        )
        losses.update(
            self.task_info[str(task_id)]
        )
        if self.loss_vel is not None:
            losses['loss_vel'] = loss_vel

        if self.corner_loss_cfg is not None:
            losses['loss_corner'] = self.get_corner_loss(pos_reg_preds, pos_bbox_targets, cluster_xyz[pos_inds], reg_avg_factor)

        if self.loss_iou is not None:
            losses['loss_iou'] = self.loss_iou(iou_logits.reshape(-1), iou_labels, label_weights, avg_factor=cls_avg_factor)
            losses['max_iou'] = iou_labels.max()
            losses['mean_iou'] = iou_labels.mean()

        # assert task_id == 0
        # return losses
        losses_with_task_id = {k + f'{self.tasks[task_id]["class_names"]}': v \
                    for k, v in losses.items()}

        return losses_with_task_id

    def get_targets(self,
                    num_task_classes,
                    no_aug_gt_bboxes_3d,
                    no_aug_gt_labels_3d,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    preds_2d,
                    cluster_xyz,
                    cluster_inds,
                    task_id=None,
                    img_metas_list=None,
                    reg_preds=None,
                    new_cls_logits=None,
                    old_cls_logits=None,
                    old_reg_preds=None,
                    ):
        batch_size = len(gt_bboxes_3d)
        batch_idx = cluster_inds[:, 0]
        cluster_xyz_list = self.split_by_batch(cluster_xyz, batch_idx, batch_size)
        preds_2d_list = self.split_by_batch(preds_2d, batch_idx, batch_size)

        if reg_preds is not None:
            reg_preds_list = self.split_by_batch(reg_preds, batch_idx, batch_size)
        else:
            reg_preds_list = [None,] * len(cluster_xyz_list)

        if old_cls_logits is not None:
            new_cls_logits_list = self.split_by_batch(new_cls_logits, batch_idx, batch_size)
            old_cls_logits_list = self.split_by_batch(old_cls_logits, batch_idx, batch_size)
            old_reg_preds_list = self.split_by_batch(old_reg_preds, batch_idx, batch_size)
        else:
            new_cls_logits_list = [None,] * len(cluster_xyz_list)
            old_cls_logits_list = [None,] * len(cluster_xyz_list)
            old_reg_preds_list = [None,] * len(cluster_xyz_list)

        num_task_class_list = [num_task_classes,] * batch_size
        task_id_list = [task_id,] * batch_size
        
        target_list_per_sample = multi_apply(
                                    self.get_targets_single, 
                                    num_task_class_list, 
                                    preds_2d_list, 
                                    cluster_xyz_list,
                                    no_aug_gt_bboxes_3d,
                                    no_aug_gt_labels_3d,
                                    gt_bboxes_3d, 
                                    gt_labels_3d, 
                                    reg_preds_list, 
                                    new_cls_logits_list,
                                    old_cls_logits_list,
                                    old_reg_preds_list,
                                    task_id_list, 
                                    img_metas_list
                                )
        
        targets = [self.combine_by_batch(t, batch_idx, batch_size) for t in target_list_per_sample]
        return targets

    def highlight_ignore_obj(self, preds_2d_pos, ignore_mask, img_metas):
        print('highlight_ignore_obj')
        if ignore_mask.sum() == 0:
            return
        sample_idx = img_metas['sample_idx']
        sample_dir = os.path.join(self.vis_dir, sample_idx)
        out_dir = os.path.join(sample_dir, 'far_case')
        os.makedirs(out_dir, exist_ok=True)

        ignore_num = int(ignore_mask.sum())
        ignore_obj_id = preds_2d_pos[ignore_mask][:, 7]
        ignore_obj_cam_id = preds_2d_pos[ignore_mask][:, 6]
        for i in range(ignore_num):
            obj_id = int(ignore_obj_id[i])
            cam_id = int(ignore_obj_cam_id[i])
            #copy mask
            mask_in_path = os.path.join(sample_dir, f'{obj_id}_cam_{cam_id}.png')
            mask_out_path = os.path.join(out_dir, f'{obj_id}_cam_{cam_id}_mask.png')
            shutil.copy(mask_in_path, mask_out_path)
            #copy obj pts
            all_pts_in_path = os.path.join(sample_dir, f'{obj_id}_points.obj')
            fg_pts_in_path = os.path.join(sample_dir, f'{obj_id}_fg_points.obj')
            bg_pts_in_path = os.path.join(sample_dir, f'{obj_id}_bg_points.obj')
            if os.path.isfile(all_pts_in_path):
                pts_out_path = os.path.join(out_dir, f'{obj_id}_points.obj')
                shutil.copy(all_pts_in_path, pts_out_path)
            if os.path.isfile(fg_pts_in_path):
                pts_out_path = os.path.join(out_dir, f'{obj_id}_fg_points.obj')
                shutil.copy(fg_pts_in_path, pts_out_path)
            if os.path.isfile(bg_pts_in_path):
                pts_out_path = os.path.join(out_dir, f'{obj_id}_bg_points.obj')
                shutil.copy(bg_pts_in_path, pts_out_path)
            #copy 2d assign
            cam_dir = os.path.join(sample_dir, f'cam_{cam_id}')
            for cls_name in os.listdir(cam_dir):
                for file in os.listdir(os.path.join(cam_dir, cls_name)):
                    if f'obj_{obj_id}.png' in file:
                        assign_in_path = os.path.join(cam_dir, cls_name, file)
                        assign_out_path = os.path.join(out_dir, f'{obj_id}_assign.png')
                        shutil.copy(assign_in_path, assign_out_path)
                        return
            
            
    def get_targets_single(self,
                           num_task_classes,
                           preds_2d,
                           cluster_xyz,
                           no_aug_gt_bboxes_3d,
                           no_aug_gt_labels_3d,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           reg_preds=None,
                           new_cls_logits=None,
                           old_cls_logits=None,
                           old_reg_preds=None,
                           task_id=None,
                           img_metas=None,
                        ):
        """Generate targets of mask for single batch.

        """
        ##[0: 4],    4,        5,      6,      7,         8,
        ## bbox, score, category, cam_id, obj_id, valid_flag
        num_cluster = cluster_xyz.shape[0]
        labels = gt_labels_3d.new_full((num_cluster, ), num_task_classes, dtype=torch.long)
        
        label_weights = preds_2d.new_ones(num_cluster)
        bbox_targets = preds_2d.new_zeros((num_cluster, self.box_code_size))
        bbox_weights = preds_2d.new_zeros((num_cluster, self.box_code_size))
        if num_cluster == 0:
            self.task_info[str(task_id)] = dict(
                num_preds=torch.tensor(0, dtype=torch.float32, device=preds_2d.device),
                num_pos_preds=torch.tensor(0, dtype=torch.float32, device=preds_2d.device),
                num_gts=torch.tensor(int(gt_bboxes_3d.tensor.shape[0]), dtype=torch.float32, device=preds_2d.device),
                assigned_gts=torch.tensor(0, dtype=torch.float32, device=preds_2d.device),
            )
            iou_labels = None
            if self.loss_iou is not None:
                iou_labels = preds_2d.new_zeros(0)
            return labels, label_weights, bbox_targets, bbox_weights, iou_labels

        no_aug_gt_bboxes_3d = no_aug_gt_bboxes_3d.to(preds_2d.device)
        no_aug_gt_labels_3d = no_aug_gt_labels_3d.to(preds_2d.device)
        gt_bboxes_3d = gt_bboxes_3d.to(preds_2d.device)
        gt_labels_3d = gt_labels_3d.to(preds_2d.device)

        if self.use_one_to_one:
            assign_result = self.assigner.assign(reg_preds, new_cls_logits, gt_bboxes_3d.tensor,
                                             gt_labels_3d)
        else:
            assign_result, assign_result_3d, assign_result_2d = self.assigner.assign(
                                        preds_2d,
                                        no_aug_gt_bboxes_3d,
                                        no_aug_gt_labels_3d, 
                                        cluster_xyz,
                                        old_cls_logits,
                                        old_reg_preds,
                                        gt_bboxes_3d,
                                        gt_labels_3d, 
                                        img_metas,
                                        task_id,
                                )
        sample_result = self.sampler.sample(
                                assign_result, 
                                gt_bboxes_3d.tensor.new_zeros((num_cluster, 9)), 
                                gt_bboxes_3d.tensor
                            )  # Pseudo Sampler
        self.task_info[str(task_id)] = dict(
            num_preds=torch.tensor(cluster_xyz.shape[0], dtype=torch.float32, device=preds_2d.device),
            num_pos_preds=torch.tensor(sample_result.pos_inds.shape[0], dtype=torch.float32, device=preds_2d.device),
            num_gts=torch.tensor(sample_result.num_gts, dtype=torch.float32, device=preds_2d.device),
            assigned_gts=torch.tensor(sample_result.pos_assigned_gt_inds.unique().shape[0], dtype=torch.float32, device=preds_2d.device),
        )
        pos_inds = sample_result.pos_inds

        # label targets
        labels[pos_inds] = gt_labels_3d[sample_result.pos_assigned_gt_inds]
        assert (labels >= 0).all()
        bbox_weights[pos_inds] = 1.0
        if len(pos_inds) > 0:
            bbox_targets[pos_inds] = self.bbox_coder.encode(
                                    sample_result.pos_gt_bboxes,
                                    cluster_xyz[pos_inds].detach(),
                                    )

            if sample_result.pos_gt_bboxes.size(1) == 10: 
                # zeros velocity loss weight for pasted objects
                assert sample_result.pos_gt_bboxes[:, 9].max().item() in (0, 1)
                assert sample_result.pos_gt_bboxes[:, 9].min().item() in (0, 1)
                assert bbox_weights.size(1) == 10, 'It is not safe to use -2: as follows if size(1) != 10'
                bbox_weights[pos_inds, -2:] = sample_result.pos_gt_bboxes[:, [9]]

        if self.loss_iou is not None:
            iou_labels = self.get_dist_labels(cluster_xyz, sample_result)
        else:
            iou_labels = None

        return labels, label_weights, bbox_targets, bbox_weights, iou_labels

    def get_far_away_mask(self, bbox_offset, task_id):
        """filter too large offset by threshold
        bbox_offset: K, 10

        return 
        ignore_mask: K
        """
        dis_bev = bbox_offset[:, :2].norm(dim=-1)
        ignore_mask = dis_bev > self.train_cfg.ignore_bev_dist[task_id]
        return ignore_mask

    def get_dist_labels(self, cluster_xyz, sample_result):
        pos_inds = sample_result.pos_inds
        num_pos = len(pos_inds)
        num_preds = cluster_xyz.shape[0]
        dist_labels = cluster_xyz.new_zeros(num_preds)
        if num_pos == 0:
            return dist_labels
        
        dist_labels_pos = dist_labels[pos_inds]
        preds_bev = cluster_xyz[pos_inds][:, :2]
        gt_bev = sample_result.pos_gt_bboxes[:, :2]

        dist = (preds_bev - gt_bev).norm(dim=-1)

        dist_min_thre = self.train_cfg.dist_min_thre
        dist_max_thre = self.train_cfg.dist_max_thre

        pos_mask = dist < dist_min_thre
        neg_mask = dist > dist_max_thre
        mid_mask = (pos_mask == 0) & (neg_mask == 0)

        dist_labels_pos[pos_mask] = 1.0
        dist_labels_pos[mid_mask] = (dist_max_thre - dist[mid_mask]) / (dist_max_thre - dist_min_thre)

        dist_labels[pos_inds] = dist_labels_pos
        return dist_labels

    @torch.no_grad()
    def get_bboxes(self,
                   cls_logits,
                   reg_preds,
                   preds_2d,
                   cluster_xyz,
                   cluster_inds,
                   input_metas,
                   iou_logits=None,
                   rescale=False,
                   ):

        assert isinstance(cls_logits, list)
        assert isinstance(reg_preds, list)

        assert len(cls_logits) == len(reg_preds) == len(self.tasks)
        alltask_result_list = []
        for i in range(len(self.tasks)):
            res_this_task = self.get_bboxes_single_task(
                i,
                cls_logits[i],
                reg_preds[i],
                preds_2d,
                cluster_xyz,
                cluster_inds,
                input_metas,
                iou_logits[i] if iou_logits is not None else None,
                rescale,
            )
            alltask_result_list.append(res_this_task)
        
        # concat results, I guess len of return list should equal to batch_size
        batch_size = len(input_metas)
        real_batch_size = len(alltask_result_list[0])
        assert  real_batch_size <= batch_size # may less than batch_size if no 
        concat_list = [] 

        for b_idx in range(batch_size):
            boxes = LiDARInstance3DBoxes.cat([task_res[b_idx][0] for task_res in alltask_result_list])
            score = torch.cat([task_res[b_idx][1] for task_res in alltask_result_list], dim=0)
            label = torch.cat([task_res[b_idx][2] for task_res in alltask_result_list], dim=0)
            concat_list.append((boxes, score, label))

        return concat_list


    @torch.no_grad()
    def get_bboxes_single_task(
        self,
        task_id,
        cls_logits,
        reg_preds,
        preds_2d,
        cluster_xyz,
        cluster_inds,
        input_metas,
        iou_logits=None,
        rescale=False,
        ):

        if cluster_inds.ndim == 1:
            batch_inds = cluster_inds
        else:
            batch_inds = cluster_inds[:, 0]

        batch_size = len(input_metas)
        # assert batch_size == 1, 'test bz is 1'
        cls_logits_list = self.split_by_batch(cls_logits, batch_inds, batch_size)
        reg_preds_list = self.split_by_batch(reg_preds, batch_inds, batch_size)
        preds_2d_list = self.split_by_batch(preds_2d, batch_inds, batch_size)
        cluster_xyz_list = self.split_by_batch(cluster_xyz, batch_inds, batch_size)

        if iou_logits is not None:
            iou_logits_list = self.split_by_batch(iou_logits, batch_inds, batch_size)
        else:
            iou_logits_list = [None,] * len(cls_logits_list)

        task_id_repeat = [task_id, ] * len(cls_logits_list)
        multi_results = multi_apply(
            self._get_bboxes_single,
            task_id_repeat,
            cls_logits_list,
            iou_logits_list,
            reg_preds_list,
            preds_2d_list,
            cluster_xyz_list,
            input_metas
        )
        # out_bboxes_list, out_scores_list, out_labels_list = multi_results
        results_list = [(b, s, l) for b, s, l in zip(*multi_results)]
        return results_list

    
    def _get_bboxes_single(
            self,
            task_id,
            cls_logits,
            iou_logits,
            reg_preds,
            preds_2d,
            cluster_xyz,
            input_meta,
        ):
        '''
        Get bboxes of a single sample
        '''

        if self.as_rpn:
            cfg = self.train_cfg.rpn if self.training else self.test_cfg.rpn
        else:
            cfg = self.test_cfg

        assert cls_logits.size(0) == reg_preds.size(0) == cluster_xyz.size(0)
        assert cls_logits.size(1) == len(self.tasks[task_id]['class_names'])
        assert reg_preds.size(1) == self.box_code_size

        if len(cls_logits) == 0:
            out_bboxes = reg_preds.new_zeros((0, 9))
            out_bboxes = input_meta['box_type_3d'](out_bboxes)
            out_scores = reg_preds.new_zeros(0)
            out_labels = reg_preds.new_zeros(0)
            return (out_bboxes, out_scores, out_labels)

        scores = cls_logits.sigmoid()

        if iou_logits is not None:
            iou_scores = iou_logits.sigmoid()
            a = cfg.get('iou_score_weight', 0.5)
            scores = (scores ** (1 - a)) * (iou_scores ** a)

        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            reg_preds = reg_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            cluster_xyz = cluster_xyz[topk_inds, :]

        bboxes = self.bbox_coder.decode(reg_preds, cluster_xyz)
        if self.vis_dir is not None:
            obj_id_tensor = preds_2d[:, 7:8]
            bboxes = torch.cat([bboxes, obj_id_tensor], dim=-1)

        bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](bboxes, box_dim=bboxes.size(1)).bev)

        # Add a dummy background class to the front when using sigmoid
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        # # test for class-agnostic nms
        #
        # single_bbox_num = bboxes.shape[0]
        # cls_num = 10
        # cls_labels = bboxes.new_zeros((cls_num, single_bbox_num))
        # for cls_idx in range(cls_num):
        #     cls_labels[cls_idx] = cls_idx
        # cls_labels = cls_labels.reshape(-1, 1)
        # bboxes = bboxes.repeat(10, 1)
        # bboxes = torch.cat([bboxes, cls_labels], dim=-1)

        # bboxes_for_nms = bboxes_for_nms.repeat(10, 1)
        # scores = scores[:, :10].permute((1, 0)).reshape(-1, 1)
        # padding = scores.new_zeros(scores.shape[0], 1)
        # scores = torch.cat([scores, padding], dim=1)
        # results = box3d_multiclass_nms(bboxes, bboxes_for_nms,
        #                             scores, score_thr, cfg.max_num,
        #                             cfg)
                                
        # out_bboxes, out_scores, out_labels = results
        # out_bboxes, out_labels = out_bboxes[:, :9], out_bboxes[:, -1:]
        #
        # # end of class-agnostic nms

        results = box3d_multiclass_nms(bboxes, bboxes_for_nms,
                                        scores, score_thr, cfg.max_num,
                                        cfg)


        out_bboxes, out_scores, out_labels = results
        if self.vis_dir is not None:
            out_bboxes, out_obj_id = out_bboxes[:, :-1], out_bboxes[:, -1]
            out_scores += out_obj_id #a smart way to encode out_obj_id
        out_bboxes = input_meta['box_type_3d'](out_bboxes, out_bboxes.size(1))

        # modify task labels to global label indices
        new_labels = torch.zeros_like(out_labels) - 1 # all -1 
        if len(out_labels) > 0:
            for i, name in enumerate(self.tasks[task_id]['class_names']):
                global_cls_ind = self.class_names.index(name)
                new_labels[out_labels == i] = global_cls_ind

            assert (new_labels >= 0).all()

        out_labels = new_labels

        return (out_bboxes, out_scores, out_labels)
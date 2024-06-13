import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, Box3DMode, Coord3DMode
from mmdet.models import build_detector
from mmdet3d.models.builder import build_backbone, build_head, build_neck, build_roi_extractor
from projects.mmdet3d_plugin.ops import build_mlp

from mmseg.models import SEGMENTORS
from mmdet3d.models import builder
from mmdet3d.ops import Voxelization, furthest_point_sample
from projects.mmdet3d_plugin.ops import scatter_v2, get_inner_win_inds
from scipy.sparse.csgraph import connected_components
from mmdet.core import multi_apply, build_bbox_coder
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.models.segmentors.base import Base3DSegmentor
from .single_stage_fsd import SingleStageFSD
import cv2, os, copy
import time, shutil
from torch import nn as nn
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
from mmdet3d.core import show_result as show_o3d_result
try:
    from torchex import connected_components as cc_gpu
except ImportError:
    cc_gpu = None
import numpy as np

@DETECTORS.register_module()
class FSF(SingleStageFSD):  
    def __init__(self,
                backbone,
                segmentor,
                voxel_layer=None,
                voxel_encoder=None,
                middle_encoder=None,
                neck=None,
                frustum_obj_head=None,
                frustum_sir=None,
                bbox_head=None,
                roi_head=None,
                train_cfg=None,
                test_cfg=None,
                cluster_assigner=None,
                pretrained=None,
                tanh_dims=3,
                init_cfg=None,
                encode_2d_mlp_cfg=dict(
                    in_channel=16,
                    mlp_channel=[128, 128],
                    norm_cfg=dict(type='LN', eps=1e-3),
                    act='gelu',
                ),
                refine_encode_2d_mlp_cfg=None,
                num_classes=10,
                num_cams=6,
                vis_dir=None,
                encode_label_only=False,
                class_names=None,
                min_pts=5,
                bbox_coder=None,
                roi_extractor=None,
                single_refine_sir_layer=None,
                mlp_cfg=dict(
                embed_dims=256,
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
                lidar_img_input_dim=128 * 3 * 2 + 128, 
                ),
                fsd_begin_idx=1000,
                refined_obj_head=None,
                segmentor_updated_mlp=dict(
                    in_channel=10, 
                    mlp_channel=[128, 67 + 64],
                    norm_cfg=dict(type='LN', eps=1e-3),
                    act='gelu',
                ),
                tta_test_cfg={},
                use_frustum=True,
                use_fsd=True,
                voxel_downsampling_size=None,
                is_argo=False,
                ):
        super().__init__(
            backbone=backbone,
            segmentor=segmentor,
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            cluster_assigner=cluster_assigner,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.runtime_info = dict()
        self.tanh_dims = tanh_dims
        self.num_classes = num_classes
        self.num_cams = num_cams
        self.vis_dir = vis_dir
        self.encode_label_only = encode_label_only
        self.class_names = class_names
        self.min_pts = min_pts
        
        self.mlp_cfg = mlp_cfg
        self.embed_dims = self.mlp_cfg.get('embed_dims', 256)
        self.norm_cfg = self.mlp_cfg.get('norm_cfg', dict(type='LN', eps=1e-3))
        self.act = self.mlp_cfg.get('act', 'gelu')
        self.lidar_img_input_dim = self.mlp_cfg.get('lidar_img_input_dim', 128 * 3 * 2 + 128)
        self.lidar_input_dim = self.mlp_cfg.get('lidar_input_dim', 128 * 3 * 2)

        self.use_fsd = use_fsd

        self.use_frustum = use_frustum
        self.frustum_obj_head = build_head(frustum_obj_head)
        self.frustum_sir = build_head(frustum_sir) 
        self.combine_frustum_feat_mlp = build_mlp(self.lidar_img_input_dim, [self.embed_dims, ], self.norm_cfg, act=self.act)

        self.encode_2d_mlp_cfg = encode_2d_mlp_cfg
        self.encode_2d_mlp = build_mlp(
                                self.encode_2d_mlp_cfg['in_channel'], 
                                self.encode_2d_mlp_cfg['mlp_channel'], 
                                self.encode_2d_mlp_cfg['norm_cfg'], 
                                is_head=False, 
                                act=self.encode_2d_mlp_cfg['act']
                            )
                            
        self.combine_fsd_feat_mlp = build_mlp(self.lidar_input_dim, [self.embed_dims, ], self.norm_cfg, act=self.act)
        
        
        self.segmentor_updated_mlp = build_mlp(segmentor_updated_mlp['in_channel'], 
                                                segmentor_updated_mlp['mlp_channel'], 
                                                segmentor_updated_mlp['norm_cfg'], 
                                                is_head=True, 
                                                act=segmentor_updated_mlp['act']
                                        )

        nn.init.constant_(self.segmentor_updated_mlp[-1].weight, 0.)
        nn.init.constant_(self.segmentor_updated_mlp[-1].bias, 0.)

        self.fsd_begin_idx = fsd_begin_idx
        self.num_extra_stages = len(refined_obj_head)
        if self.num_extra_stages > 0:
            self.bbox_coder = build_bbox_coder(bbox_coder)
            self.roi_extractor = build_roi_extractor(roi_extractor) 
            self.refine_sir_layers = nn.ModuleList([build_head(single_refine_sir_layer) for _ in range(self.num_extra_stages)])

            self.refine_encode_2d_mlp_cfg = refine_encode_2d_mlp_cfg
            self.refine_img_mlp = nn.ModuleList([build_mlp(
                                                    self.refine_encode_2d_mlp_cfg['in_channel'], 
                                                    self.refine_encode_2d_mlp_cfg['mlp_channel'], 
                                                    self.refine_encode_2d_mlp_cfg['norm_cfg'], 
                                                    is_head=False, 
                                                    act=self.refine_encode_2d_mlp_cfg['act']
                                                ) for _ in range(self.num_extra_stages)])
                                                    
            self.lidar_img_mlp = nn.ModuleList([build_mlp(self.lidar_input_dim, [self.embed_dims, self.embed_dims], self.norm_cfg, act=self.act) for _ in range(self.num_extra_stages)])
            self.position_encoder = nn.ModuleList([build_mlp(3, [self.embed_dims, self.embed_dims], self.norm_cfg, act=self.act) for _ in range(self.num_extra_stages)])
            self.out_proj = nn.ModuleList([build_mlp(self.embed_dims, [self.embed_dims, self.embed_dims], self.norm_cfg, act=self.act, is_head=True) for _ in range(self.num_extra_stages)])
            self.frustum_refined_head = nn.ModuleList([build_head(refined_obj_head[idx]) for idx in range(self.num_extra_stages)])
        self.tta_test_cfg = tta_test_cfg
        self.voxel_downsampling_size = voxel_downsampling_size
        self.is_argo = is_argo

    def prj_points_2d(self, points, lidar2img, img_h, img_w):
        """points: N, 3
           lidar2img: 6, 4, 4

           return: 
           pts_2d: 6, N, 2
        """
        #N, 3 -> N, 4
        pts_4d = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)
        #pts_2d: 6, N, 4
        pts_2d = pts_4d @ lidar2img.permute(0, 2, 1)
        depth_valid_mask = pts_2d[..., 2] > 1e-3

        pts_2d[..., 2] = torch.clip(pts_2d[..., 2], min=1e-5, max=1e5)
        pts_2d[..., 0] /= pts_2d[..., 2]
        pts_2d[..., 1] /= pts_2d[..., 2]

        pts_2d[..., 0] /= img_w
        pts_2d[..., 1] /= img_h

        pts_2d = pts_2d[..., :2]
        pts_2d = (pts_2d - 0.5) * 2

        img_valid_mask = ((pts_2d[..., 0:1] > -1.0) 
                 & (pts_2d[..., 0:1] < 1.0) 
                 & (pts_2d[..., 1:2] > -1.0) 
                 & (pts_2d[..., 1:2] < 1.0)).squeeze(-1)

        valid_mask = depth_valid_mask & img_valid_mask
        # assert (img_valid_mask & ~depth_valid_mask).sum() == 0
        pts_2d[~valid_mask] = -2.0
        return pts_2d

    def points_in_mask(self, points, mask_data, lidar2img):
        """design for a single batch with 6 cams
        mask_data: 6, H, W
        return:
        obj_id_of_pts: 6, N
        """

        mask_tensor = mask_data.float()
        num_cams, num_classes, img_h, img_w = mask_tensor.shape

        #6, N, 2
        pts_2d = self.prj_points_2d(points, lidar2img, img_h, img_w)

        obj_id_list = []
        for cam_id in range(num_cams):
            pts_2d_cam = pts_2d[cam_id].unsqueeze(0).unsqueeze(1)
            mask_cam = mask_tensor[cam_id].unsqueeze(0)
            #mask_cam: 1, 10, 900, 1600
            #pts_2d_cam: 1, 1, N, 2
            obj_id_cam = F.grid_sample(mask_cam, pts_2d_cam, mode='nearest')
            obj_id_cam = obj_id_cam.squeeze(2).long()
            obj_id_list.append(obj_id_cam)
        #6, 10, N
        obj_id_tensor = torch.cat(obj_id_list, dim=0)
        return obj_id_tensor.permute(2, 0, 1) #N, 6, 10

    def frustum_gather(self, 
                        batch_idx, 
                        points, 
                        mask_data, 
                        mask_anno, 
                        img_metas):
        """get each pt's obj id
        """
        # pooling_func = F.max_pool1d
        device = batch_idx.device
        bz, num_cams, num_classes = mask_data.shape[0:3]
        obj_id_tensor = batch_idx.new_zeros((batch_idx.shape[0], num_cams, num_classes))
        for bidx in range(bz):
            ##split for batch
            bz_mask = (batch_idx == bidx)
            points_bz = points[bz_mask]
            mask_data_bz = mask_data[bidx]
            img_metas_bz = img_metas[bidx]

            ##several cams 
            lidar2img_bz = torch.tensor(
                [data for data in img_metas_bz['lidar2img']], 
                device=device,
                dtype=torch.float32
            )
            obj_id_tensor_bz = self.points_in_mask(points_bz[:, :3], 
                                mask_data_bz, 
                                lidar2img_bz,
            )
            obj_id_tensor[bz_mask] = obj_id_tensor_bz
        return obj_id_tensor

    def double_overlap_pts(self, 
                        pts_feat, 
                        bz_coor, 
                        points, 
                        obj_id_tensor, 
                        point_fg_weights
                    ):
        obj_id_tensor = obj_id_tensor.reshape(obj_id_tensor.shape[0], -1)
        overlaps_tensor = (obj_id_tensor > 0).sum(-1)
        max_overlap_num = overlaps_tensor.max() + 1
        pts_feat_clone = pts_feat.clone()
        bz_coor_clone = bz_coor.clone()
        points_clone = points.clone()
        point_fg_weights_clone = point_fg_weights.clone()

        raw_obj_id_tensor = obj_id_tensor.max(-1)[0]
        for overlap_num in range(2, max_overlap_num):
            overlaps_mask = overlaps_tensor == overlap_num
            if overlaps_mask.sum() == 0:
                continue

            pad_pts_feat = pts_feat_clone[overlaps_mask].clone().repeat(overlap_num -1, 1)
            pad_bz_coor = bz_coor_clone[overlaps_mask].clone().repeat(overlap_num -1, 1)
            pad_points = points_clone[overlaps_mask].clone().repeat(overlap_num -1, 1)
            pad_point_fg_weights = point_fg_weights_clone[overlaps_mask].clone().repeat(overlap_num -1)

            pts_feat = torch.cat([pts_feat, pad_pts_feat], dim=0)
            bz_coor = torch.cat([bz_coor, pad_bz_coor], dim=0)
            point_fg_weights = torch.cat([point_fg_weights, pad_point_fg_weights], dim=0)
            points = torch.cat([points, pad_points], dim=0)

            #generate pad_obj_id_tensor
            sort_value = obj_id_tensor[overlaps_mask].topk(overlap_num, dim=-1)[0]
            for pad_idx in range(1, overlap_num):
                pad_obj_id_tensor = sort_value[:, pad_idx].clone()
                raw_obj_id_tensor = torch.cat([raw_obj_id_tensor, pad_obj_id_tensor], dim=0)

        return pts_feat, bz_coor, points, raw_obj_id_tensor, point_fg_weights

    def extract_fg_pts(self, 
                    pts_feat, 
                    bz_coor, 
                    points, 
                    obj_id_tensor, 
                    point_fg_weights
                ):
        fg_mask = obj_id_tensor.sum((-2, -1)) > 0
        return pts_feat[fg_mask], bz_coor[fg_mask], points[fg_mask], \
            obj_id_tensor[fg_mask], point_fg_weights[fg_mask]

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):
        return voxel_mean[voxel2point_inds]

    def get_cluster_delta_weighted(self, points, sir_coors, point_weights):
        #compute all frames
        point_weights = point_weights.clamp(min=1e-5).detach()
        input_feat = torch.cat([points[:, :3] * point_weights,
                                point_weights], dim=-1)
        voxel_mean_feat, voxel_mean_coors, unq_inv = scatter_v2(
                                            input_feat, 
                                            sir_coors, 
                                            mode='avg', 
                                        )
        voxel_center = voxel_mean_feat[:, :3] / voxel_mean_feat[:, 3:4]

        points_center = self.map_voxel_center_to_point(
            voxel_center, unq_inv)

        f_cluster = (points[:, :3] - points_center[:, :3])
        return f_cluster, voxel_center, voxel_mean_coors
    
    def get_cluster_delta_avg(self, points, sir_coors, point_weights):
        input_feat = points[:, :3]
        voxel_mean_feat, voxel_mean_coors, unq_inv = scatter_v2(
                                            input_feat, 
                                            sir_coors, 
                                            mode='avg', 
                                        )
        voxel_center = voxel_mean_feat[:, :3]
        points_center = self.map_voxel_center_to_point(
            voxel_center, unq_inv)
        
        #visualize the obj of the big delta 
        f_cluster = (points[:, :3] - points_center[:, :3])
        return f_cluster, voxel_center, voxel_mean_coors
    
    def get_point_fg_weights(self, seg_logits):
        """
        seg_logits : N, 11
        return 
        point_weights: N, 1
        """
        seg_logits_softmax = seg_logits.softmax(1)
        seg_prob_bg = seg_logits_softmax[:, -1]
        seg_prob_fg = 1 - seg_prob_bg
        return seg_prob_fg

    def get_sir_coors(self,
                      bz_coor,
                      obj_id_tensor,
                      point_fg_weights,
                      ):
        sir_coors = torch.cat([bz_coor, 
                               torch.zeros_like(bz_coor), 
                               obj_id_tensor.unsqueeze(-1),], dim=-1)
        return sir_coors, obj_id_tensor     

    def get_cluster_delta_from_center(self,
                                    points,
                                    sir_coors,
                                    cluster_center
                                    ):
        pts_mean_feat, pts_mean_coors, unq_inv = scatter_v2(
                                                points, 
                                                sir_coors, 
                                                mode='avg', 
                                            )

        points_center = self.map_voxel_center_to_point(
            cluster_center, unq_inv)

        points_delta = (points[:, :3] - points_center[:, :3])
        return points_delta

    def frustum_pooling(self, 
                        pts_feat, 
                        bz_coor, 
                        points, 
                        obj_id_tensor, 
                        point_fg_weights,
                        img_metas=None,
                        cluster_center=None,
                    ):
        """use sir to pooling the pts' feature
        pts_feat: N, C
        obj_id_tensor: N, 6, 10

        return
        obj_feat: K, C   (K means obj_num, set to 256)
        """
        pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights = \
                self.extract_fg_pts(pts_feat, 
                                    bz_coor, 
                                    points, 
                                    obj_id_tensor, 
                                    point_fg_weights,
                                    )
        if obj_id_tensor.sum() ==0:
            #fake an obj here if frustum branch no output
            fake_num = 1
            points = points.new_zeros(fake_num, points.shape[-1])
            pts_feat = pts_feat.new_zeros(fake_num, pts_feat.shape[-1])
            sir_coors = bz_coor.new_zeros(fake_num, 3)
            points_delta = points.new_zeros(fake_num, 3)
            cluster_center = points.new_zeros(fake_num, 3)
        else:
            pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights = \
                self.double_overlap_pts(pts_feat, 
                                        bz_coor, 
                                        points, 
                                        obj_id_tensor, 
                                        point_fg_weights)

            sir_coors, obj_id_tensor = self.get_sir_coors(bz_coor,
                                        obj_id_tensor,
                                        point_fg_weights)
            
            if cluster_center is None:
                points_delta, cluster_center, cluster_coors\
                                = self.get_cluster_delta_weighted(
                                            points, 
                                            sir_coors, 
                                            point_fg_weights.unsqueeze(-1)
                                        )
            else:
                points_delta = self.get_cluster_delta_from_center(
                                        points,
                                        sir_coors,
                                        cluster_center
                                    )
            

        out_feats, final_cluster_feats, out_coors = \
            self.frustum_sir(points, pts_feat, sir_coors, f_cluster=points_delta)
        
        if out_coors.shape[0] == 0:
            out_coors = out_coors.new_zeros((0, 3))
        return final_cluster_feats, out_coors, cluster_center

    def encode_preds_2d(self, preds_2d, img_w, img_h, encode_single_cls=True):
        """select and encode preds_2d
        preds_2d: K, 9
        return 
        feats_2d: K, 128,
        """
        ##[0: 4],    4,        5,      6,      7,         8,
        ## bbox, score, category, cam_id, obj_id, valid_flag
        bbox_2d, score, category, cam_id = \
            preds_2d[:, :4], preds_2d[:, 4:5], preds_2d[:, 5], preds_2d[:, 6]
        en_bbox_2d = bbox_2d.clone()
        en_bbox_2d[:, 0::2] /= img_w
        en_bbox_2d[:, 1::2] /= img_h
        en_score = score
        en_category = F.one_hot(category.long(), num_classes=self.num_classes + 1)
        en_cam_id = F.one_hot(cam_id.long(), num_classes=self.num_cams)
        if self.encode_label_only:
            en_feat = en_category.float()
        elif encode_single_cls:
            en_feat = torch.cat(
                [en_bbox_2d, en_score, en_category.float()], dim=-1
            )
        else:
            #encode ten cls score
            en_feat = en_score
        return en_feat

    def get_single_cls_preds_2d(self, mask_anno, obj_coors):
        """
        mask_anno: bz, 250, 9
        obj_coors, K, 12
        """
        batch_tensor, obj_id_tensor = obj_coors[:, 0], obj_coors[:, 2]
        num_objs = obj_coors.shape[0]
        num_mask_annos = mask_anno.shape[-1]
        batch_size = mask_anno.shape[0]
        preds_2d_flat = torch.zeros(
                        (num_objs, num_mask_annos), 
                        dtype=torch.float32, 
                        device=obj_coors.device
                    )

        for bidx in range(batch_size):
            bz_mask = batch_tensor == bidx
            
            preds_2d_flat_bz = preds_2d_flat[bz_mask]
            obj_id_tensor_bz = obj_id_tensor[bz_mask] - 1

            valid_mask = obj_id_tensor_bz >= 0
            preds_2d_flat_bz[valid_mask] = mask_anno[bidx][obj_id_tensor_bz[valid_mask]]
            preds_2d_bz_invalid = preds_2d_flat_bz[~valid_mask]
            preds_2d_bz_invalid[:, 5] = self.num_classes
            preds_2d_flat_bz[~valid_mask] = preds_2d_bz_invalid

            preds_2d_flat[bz_mask] = preds_2d_flat_bz
        return preds_2d_flat
    
    def get_all_cls_preds_2d(self, mask_anno, batch_tensor, obj_id_tensor):
        """
        mask_anno: bz, 250, 9
        obj_coors, K, 10
        """
        num_objs = batch_tensor.shape[0]
        num_mask_annos = mask_anno.shape[-1]
        num_classes = obj_id_tensor.shape[-1]
        batch_size = mask_anno.shape[0]
        preds_2d_flat = torch.zeros(
                        (num_objs, num_classes, num_mask_annos), 
                        dtype=torch.float32, 
                        device=batch_tensor.device
                    )

        for bidx in range(batch_size):
            bz_mask = batch_tensor == bidx
            
            preds_2d_flat_bz = preds_2d_flat[bz_mask]
            obj_id_tensor_bz = obj_id_tensor[bz_mask] - 1

            valid_mask = obj_id_tensor_bz >= 0
            preds_2d_flat_bz[valid_mask] = mask_anno[bidx][obj_id_tensor_bz[valid_mask]]
            preds_2d_bz_invalid = preds_2d_flat_bz[~valid_mask]
            preds_2d_bz_invalid[:, 5] = num_classes
            preds_2d_flat_bz[~valid_mask] = preds_2d_bz_invalid


            preds_2d_flat[bz_mask] = preds_2d_flat_bz
        return preds_2d_flat #(num_objs, num_classes, num_mask_annos)

    def encode_2d_feats(self, preds_2d, img_w, img_h, encode_mlp):
        #preds_2d   (num_objs, num_classes, num_mask_annos)
        if len(preds_2d.shape) == 3:
            num_objs, num_classes, num_mask_annos = preds_2d.shape
            encoded_2d = self.encode_preds_2d(
                                preds_2d.reshape(-1, num_mask_annos),
                                img_w, 
                                img_h, 
                                encode_single_cls=self.is_argo
                            )
            if not self.is_argo:
                encoded_2d = encoded_2d.reshape(num_objs, num_classes)
        else:
            encoded_2d = self.encode_preds_2d(preds_2d, img_w, img_h)
        encoded_2d_feat = encode_mlp(encoded_2d)
        return encoded_2d_feat

    def split_points_last_3dim(self, points):
        no_aug_points = []
        new_points = []
        for point in points:
            no_aug_points.append(point[:, -3:])
            new_points.append(point[:, :-3])
        return new_points, no_aug_points

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        data_list_flat = data_list[0].new_zeros((batch_idx.shape[0], data_list[0].shape[-1]))
        for bidx in range(batch_size):
            bz_mask = batch_idx == bidx
            data_list_flat[bz_mask] = data_list[bidx]
        return data_list_flat   

    def fsd_forward(self,
                    seg_out_dict,
                    img_metas
                    ):
        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'].detach(),
            seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),
            seg_feats=seg_out_dict['seg_feats'],
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets=seg_out_dict['offsets'].detach(),
        )
        if self.cfg.get('pre_voxelization_size', None) is not None:
            dict_to_sample = self.pre_voxelize(dict_to_sample)
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets']) # per cls list in sampled_out ## to extract the fg by the classification score

        # we filter almost empty voxel in clustering, so here is a valid_mask
        cluster_inds_list, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], origin_points=sampled_out['seg_points']) # per cls list
        pts_cluster_inds = torch.cat(cluster_inds_list, dim=0) #[N, 3], (cls_id, batch_idx, cluster_id)

        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])

        points = combined_out['seg_points']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)

        extracted_outs = self.extract_feat(points, pts_feats, pts_cluster_inds, img_metas, combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds'] # [class, batch, groups]

        assert (cluster_inds[:, 0]).max().item() < self.num_classes
        obj_feat = cluster_feats
        outs = self.bbox_head(obj_feat)
        return obj_feat, cluster_xyz, cluster_inds, outs

    def frustum_forward(self,
                        seg_out_dict,
                        mask_anno,
                        mask_data,
                        point_infos,
                        img_metas,
                        cluster_center=None,
                        ):
        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']
        seg_logits = seg_out_dict['seg_logits']

        point_fg_weights = self.get_point_fg_weights(seg_logits)

        batch_size = mask_anno.shape[0]
        points_info_flat = self.combine_by_batch(point_infos, batch_idx, batch_size)

        #grouping
        obj_id_tensor = self.frustum_gather(batch_idx, 
                                            points_info_flat, #no_aug_points
                                            mask_data, 
                                            mask_anno, 
                                            img_metas)
                
        #pooling
        lidar_feat, obj_coors, obj_centers = self.frustum_pooling(pts_feat, 
                                                    batch_idx.unsqueeze(-1), 
                                                    points, 
                                                    obj_id_tensor,
                                                    point_fg_weights,
                                                    img_metas,
                                                    cluster_center
                                                    )

        #get preds 2d
        preds_2d = self.get_single_cls_preds_2d(mask_anno, obj_coors)
       
        img_feat = self.encode_2d_feats(
            preds_2d,
            img_w=mask_data.shape[-1],
            img_h=mask_data.shape[-2],
            encode_mlp=self.encode_2d_mlp,
        )
        lidar_img_feat = torch.cat([lidar_feat, img_feat], dim=-1)
        obj_feat = lidar_img_feat
        frustum_obj_result = self.frustum_obj_head(obj_feat)
        
        return obj_feat, obj_centers, obj_coors, frustum_obj_result, preds_2d

    def combine_frustum_and_fsd(self,
                                frustum_obj_centers,
                                frustum_obj_coors,
                                frustum_obj_result,
                                frustum_obj_feats,
                                frustum_preds_2d,
                                fsd_obj_centers,
                                fsd_obj_coors,
                                fsd_obj_result,
                                fsd_obj_feats,
                            ):
        obj_centers = torch.cat([frustum_obj_centers, fsd_obj_centers], dim=0)

        fsd_obj_coors_re = fsd_obj_coors.clone()
        fsd_obj_coors_re[:, 0] = fsd_obj_coors[:, 1]
        fsd_obj_coors_re[:, 1] = fsd_obj_coors[:, 0]
        fsd_obj_coors_re[:, 2] += self.fsd_begin_idx
        obj_coors = torch.cat([frustum_obj_coors, fsd_obj_coors_re], dim=0)

        obj_result = {}
        for key in frustum_obj_result.keys():
            batch_size = len(frustum_obj_result[key])
            obj_result[key] = []
            for bidx in range(batch_size):
                data = torch.cat([frustum_obj_result[key][bidx], fsd_obj_result[key][bidx]], dim=0)
                obj_result[key].append(data)
        
        frustum_obj_feats_com = self.combine_frustum_feat_mlp(frustum_obj_feats)
        fsd_obj_feats_com = self.combine_fsd_feat_mlp(fsd_obj_feats)

        obj_feats = torch.cat([frustum_obj_feats_com, fsd_obj_feats_com], dim=0)

        fsd_preds_2d = frustum_preds_2d.new_zeros((fsd_obj_feats.shape[0], frustum_preds_2d.shape[1]))
        preds_2d = torch.cat([frustum_preds_2d, fsd_preds_2d], dim=0)

        return obj_centers, obj_coors, obj_result, obj_feats, preds_2d

    def img_cross_attn(self,
                      point_infos,
                      batch_idx,
                      mask_anno,
                      mask_data,
                      img_metas,
                      encode_mlp,
                      ext_pts_inds=None,
                      ):
        batch_size = mask_anno.shape[0]
        points_info_flat = self.combine_by_batch(point_infos, batch_idx, batch_size)
        if ext_pts_inds is not None:
            points_info_flat = points_info_flat[ext_pts_inds]
            batch_idx = batch_idx[ext_pts_inds]
        #grouping
        obj_id_tensor = self.frustum_gather(batch_idx, 
                                            points_info_flat, #no_aug_points
                                            mask_data, 
                                            mask_anno, 
                                            img_metas)
        #N, 6, 10
        _, num_cams, num_classes = obj_id_tensor.shape
        cam_select_value = obj_id_tensor.sum(-1).max(-1)[1] 
        cam_select_mask = F.one_hot(cam_select_value, num_cams).bool().unsqueeze(-1) 
        points_obj_id_multi_cls = obj_id_tensor.masked_select(cam_select_mask).reshape(-1, num_classes)   #N, 10      
        preds_2d = self.get_all_cls_preds_2d(mask_anno, batch_idx, points_obj_id_multi_cls)

        img_feat = self.encode_2d_feats(
            preds_2d,
            img_w=mask_data.shape[-1],
            img_h=mask_data.shape[-2],
            encode_mlp=encode_mlp,
        )

        return img_feat

    def segmentor_feat_inhance_train(self, seg_out_tuple, point_infos, mask_anno, mask_data, img_metas):
        (neck_out, pts_coors, points, labels, vote_targets, vote_mask) = seg_out_tuple
        batch_idx = pts_coors[:, 0]
        losses = dict()

        pts_lidar_feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]
        labels = labels[valid_pts_mask]
        vote_targets = vote_targets[valid_pts_mask]
        vote_mask = vote_mask[valid_pts_mask]
        assert pts_lidar_feats.size(0) == labels.size(0)

        pts_updated_feats = self.img_cross_attn(point_infos,
                            batch_idx,
                            mask_anno,
                            mask_data,
                            img_metas,
                            encode_mlp=self.segmentor_updated_mlp,
                            )
        
        pts_feats = pts_lidar_feats + pts_updated_feats

        loss_decode, preds_dict = self.segmentor.segmentation_head.forward_train(pts_feats, img_metas, labels, vote_targets, vote_mask, return_preds=True)
        losses.update(loss_decode)

        vote_preds = preds_dict['vote_preds']

        offsets = self.segmentor.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,
            seg_logits=preds_dict['seg_logits'],
            seg_vote_preds=preds_dict['vote_preds'],
            offsets=offsets,
            seg_feats=pts_feats,
            batch_idx=pts_coors[:, 0],
            losses=losses
        )
        return output_dict

    def segmentor_feat_inhance_test(self, seg_out_tuple, point_infos, mask_anno, mask_data, img_metas):
        (neck_out, pts_coors, points) = seg_out_tuple
        batch_idx = pts_coors[:, 0]
        
        
        pts_lidar_feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]

        pts_updated_feats = self.img_cross_attn(point_infos,
                            batch_idx,
                            mask_anno,
                            mask_data,
                            img_metas,
                            encode_mlp=self.segmentor_updated_mlp,
                            )
        
        pts_feats = pts_lidar_feats + pts_updated_feats

        seg_logits, vote_preds = self.segmentor.segmentation_head.forward_test(pts_feats, img_metas, self.segmentor.test_cfg)

        offsets = self.segmentor.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,
            seg_logits=seg_logits,
            seg_vote_preds=vote_preds,
            offsets=offsets,
            seg_feats=pts_feats,
            batch_idx=pts_coors[:, 0],
        )
        return output_dict

    def forward_train(self,
                      points,
                      img_metas,
                      no_aug_gt_bboxes_3d,
                      no_aug_gt_labels_3d,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      mask_data,
                      mask_anno,
                      gt_bboxes_ignore=None,
                      img=None,):
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)
        points, point_infos = self.split_points_last_3dim(points)
        
        no_aug_gt_bboxes_3d = [b[l>=0] for b, l in zip(no_aug_gt_bboxes_3d, no_aug_gt_labels_3d)]
        no_aug_gt_labels_3d = [l[l>=0] for l in no_aug_gt_labels_3d]
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]

        losses = {}

        ##1. segmentation
        seg_out_tuple = self.segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True, extract_feat_only=True)
        seg_out_dict = self.segmentor_feat_inhance_train(seg_out_tuple, point_infos, mask_anno, mask_data, img_metas)

        seg_loss = seg_out_dict['losses']
        losses.update(seg_loss)

        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']
        
        ##2. camera queries
        frustum_obj_feats, frustum_obj_centers, frustum_obj_coors, frustum_obj_result, frustum_preds_2d \
                                                        = self.frustum_forward(seg_out_dict,
                                                                                mask_anno,
                                                                                mask_data,
                                                                                point_infos,
                                                                                img_metas,
                                                                                cluster_center=None)
        frustum_query_losses = self.frustum_obj_head.loss(
                                frustum_obj_result['cls_logits'],
                                frustum_obj_result['reg_preds'],
                                frustum_obj_centers,
                                frustum_obj_coors,
                                no_aug_gt_bboxes_3d,
                                no_aug_gt_labels_3d,
                                gt_bboxes_3d,
                                gt_labels_3d,
                                frustum_preds_2d,
                                img_metas,
                                )
        for loss_key, loss_value in frustum_query_losses.items():
            losses[f'frustum_' + loss_key] = loss_value

        ##3. LiDAR queries
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = self.fsd_forward(seg_out_dict, img_metas)
        fsd_loss_inputs = (fsd_obj_result['cls_logits'], fsd_obj_result['reg_preds']) + (fsd_obj_centers, fsd_obj_coors) + (gt_bboxes_3d, gt_labels_3d, img_metas)
        fsd_query_loss = self.bbox_head.loss(
            *fsd_loss_inputs, iou_logits=fsd_obj_result.get('iou_logits', None), gt_bboxes_ignore=gt_bboxes_ignore)
        for loss_key, loss_value in fsd_query_loss.items():
            losses[f'fsd_' + loss_key] = loss_value

        ##4. Query Refinement
        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = \
            self.combine_frustum_and_fsd(
                frustum_obj_centers,
                frustum_obj_coors,
                frustum_obj_result,
                frustum_obj_feats,
                frustum_preds_2d,
                fsd_obj_centers,
                fsd_obj_coors,
                fsd_obj_result,
                fsd_obj_feats,
            )

        if self.num_extra_stages > 0:
            multi_stage_losses = self.multi_stage_refine_train(obj_centers,
                                                    obj_coors,
                                                    obj_result,
                                                    points,
                                                    point_infos,
                                                    pts_feat,
                                                    batch_idx,
                                                    mask_data,
                                                    mask_anno,
                                                    no_aug_gt_bboxes_3d,
                                                    no_aug_gt_labels_3d,
                                                    gt_bboxes_3d,
                                                    gt_labels_3d,
                                                    preds_2d,
                                                    img_metas,
                                                    obj_feats,
                                                    )
            losses.update(multi_stage_losses)
        return losses
    
    def multi_stage_refine_train(self,
                            obj_centers,
                            obj_coors,
                            obj_result,
                            points,
                            point_infos,
                            pts_feat,
                            batch_idx,
                            mask_data,
                            mask_anno,
                            no_aug_gt_bboxes_3d,
                            no_aug_gt_labels_3d,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            preds_2d,
                            img_metas,
                            res_query_feat,
                            ):
        mutli_stage_losses = {}
        for i_stage in range(self.num_extra_stages):
            old_obj_result = obj_result
            obj_centers, obj_result, res_query_feat\
                                        = self.each_stage_refine(i_stage,
                                                            obj_centers,
                                                            obj_coors,
                                                            obj_result,
                                                            points,
                                                            point_infos,
                                                            pts_feat,
                                                            batch_idx,
                                                            mask_data,
                                                            mask_anno,
                                                            img_metas,
                                                            res_query_feat,
                                                            )
            refined_query_losses = self.frustum_refined_head[i_stage].loss(
                                obj_result['cls_logits'],
                                obj_result['reg_preds'],
                                obj_centers,
                                obj_coors,
                                no_aug_gt_bboxes_3d,
                                no_aug_gt_labels_3d,
                                gt_bboxes_3d,
                                gt_labels_3d,
                                preds_2d,
                                img_metas,
                                obj_result.get('iou_logits', None),
                                old_obj_result['cls_logits'],
                                old_obj_result['reg_preds'],
                                )
            stage_refined_query_losses = {}
            for loss_key, loss_value in refined_query_losses.items():
                stage_refined_query_losses[f'stage_{i_stage}_' + loss_key] = loss_value
            mutli_stage_losses.update(stage_refined_query_losses)
        return mutli_stage_losses

    def multi_stage_refine_test(self,
                            obj_centers,
                            obj_coors,
                            obj_result,
                            points,
                            point_infos,
                            pts_feat,
                            batch_idx,
                            mask_data,
                            mask_anno,
                            preds_2d,
                            img_metas,
                            res_query_feat,
                            ):
        all_stage_bbox_list = []
        obj_centers_list = []
        obj_result_list = []
        for i_stage in range(self.num_extra_stages):
            old_obj_result = copy.deepcopy(obj_result)
            obj_centers, obj_result, res_query_feat\
                                        = self.each_stage_refine(i_stage,
                                                            obj_centers,
                                                            obj_coors,
                                                            obj_result,
                                                            points,
                                                            point_infos,
                                                            pts_feat,
                                                            batch_idx,
                                                            mask_data,
                                                            mask_anno,
                                                            img_metas,
                                                            res_query_feat,
                                                            )
            obj_centers_list.append(obj_centers)
            obj_result_list.append(obj_result)

            each_stage_bbox_list = self.frustum_refined_head[i_stage].get_bboxes(
                                            obj_result['cls_logits'],
                                            obj_result['reg_preds'],
                                            preds_2d, 
                                            obj_centers,
                                            obj_coors, 
                                            img_metas,
                                            iou_logits=obj_result.get('iou_logits', None)
                                            )
            all_stage_bbox_list.append(each_stage_bbox_list)
        return all_stage_bbox_list[-1]

    def query_feat_refine(self,
                        points,
                        pts_feat,
                        batch_idx,
                        input_bbox_rois,
                        i_stage,
                        point_infos,
                        mask_anno,
                        mask_data,
                        img_metas,
                        ):
        ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
                                                    points[:, :3], # intensity might be in pts_xyz
                                                    batch_idx,
                                                    input_bbox_rois[:, :8],  # [22,8]
                                                )

        extracted_points = points[ext_pts_inds]
        extracted_points_feats = pts_feat[ext_pts_inds]

        pts_img_feat = self.img_cross_attn(point_infos,
                                            batch_idx,
                                            mask_anno,
                                            mask_data,
                                            img_metas,
                                            self.refine_img_mlp[i_stage],
                                            ext_pts_inds,
                                            )
    
        ext_pts_feats_updated = torch.cat([extracted_points_feats, pts_img_feat], dim=-1)
        lidar_feat, lidar_mask = self.refine_sir_layers[i_stage](extracted_points,
                                        ext_pts_feats_updated,
                                        ext_pts_info,
                                        ext_pts_roi_inds,
                                        input_bbox_rois)
        return lidar_feat

    def each_stage_refine(self, 
                        i_stage,
                        old_obj_centers,
                        obj_coors,
                        old_obj_result,
                        points,
                        point_infos,
                        pts_feat,
                        batch_idx,
                        mask_data,
                        mask_anno,
                        img_metas,
                        res_query_feat,
                        ):
        if len(old_obj_centers) == 0:
            lidar_img_feat = old_obj_centers.new_zeros((0, self.lidar_input_dim))
            obj_centers = old_obj_centers.clone()
        else:
            input_bbox_rois = self.decode_stage_bboxes(old_obj_centers, obj_coors[:, 0], old_obj_result['reg_preds'])
            obj_centers = input_bbox_rois[:, 1:4]
    
            lidar_img_feat = self.query_feat_refine(points, 
                                                    pts_feat, 
                                                    batch_idx, 
                                                    input_bbox_rois, 
                                                    i_stage, 
                                                    point_infos,
                                                    mask_anno,
                                                    mask_data,
                                                    img_metas,)

        cur_query_feat = self.lidar_img_mlp[i_stage](lidar_img_feat)
        pos_feat = self.position_encoder[i_stage](obj_centers.detach())
        query_feat = self.out_proj[i_stage](cur_query_feat + res_query_feat + pos_feat)

        obj_result = self.frustum_refined_head[i_stage](query_feat)

        return obj_centers, obj_result, query_feat

    def decode_stage_bboxes(self, obj_centers, bz_coors, reg_preds):
        batch_size = len(reg_preds)
        decode_size = reg_preds[0].shape[-1] - 1
        bboxes_tensor = reg_preds[0].new_zeros((bz_coors.shape[0], decode_size))
        for bidx in range(batch_size):
            bz_mask = bz_coors == bidx
            bboxes_bz = self.bbox_coder.decode(reg_preds[bidx], obj_centers[bz_mask])
            bboxes_tensor[bz_mask] = bboxes_bz
        bboxes_tensor_roi = torch.cat([bz_coors.unsqueeze(-1), bboxes_tensor], dim=-1)
        return bboxes_tensor_roi

    def forward_test(self,
                    points,
                    img_metas,
                    mask_data,
                    mask_anno,
                    **kwargs,
                ):
        num_augs = len(points)
        if num_augs == 1:
            return self.simple_test(points[0],
                           img_metas[0],
                           mask_data[0],
                           mask_anno[0],
                           **kwargs,
                    )
        else:
            return self.aug_test(points, img_metas, mask_data, mask_anno, **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    mask_data,
                    mask_anno,
                    **kwargs,
                ):
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)
        points, point_infos = self.split_points_last_3dim(points)
            
        ##1. segmentation
        seg_out_tuple = self.segmentor.simple_test(points, img_metas, extract_feat_only=True, rescale=False) 
        seg_out_dict = self.segmentor_feat_inhance_test(seg_out_tuple, point_infos, mask_anno, mask_data, img_metas)
        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']

        ##2. camera queries
        frustum_obj_feats, frustum_obj_centers, frustum_obj_coors, frustum_obj_result, frustum_preds_2d \
                                                        = self.frustum_forward(seg_out_dict,
                                                                                mask_anno,
                                                                                mask_data,
                                                                                point_infos,
                                                                                img_metas,
                                                                                cluster_center=None,)

        ##3. LiDAR queries
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = self.fsd_forward(seg_out_dict, img_metas)

        ##4. Query Refinement
        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = \
            self.combine_frustum_and_fsd(
                frustum_obj_centers,
                frustum_obj_coors,
                frustum_obj_result,
                frustum_obj_feats,
                frustum_preds_2d,
                fsd_obj_centers,
                fsd_obj_coors,
                fsd_obj_result,
                fsd_obj_feats,
            )
            
        if self.num_extra_stages >= 0:
            bbox_list = self.multi_stage_refine_test(obj_centers,
                                                obj_coors,
                                                obj_result,
                                                points,
                                                point_infos,
                                                pts_feat,
                                                batch_idx,
                                                mask_data,
                                                mask_anno,
                                                preds_2d,
                                                img_metas,
                                                obj_feats,
                                                )
     
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
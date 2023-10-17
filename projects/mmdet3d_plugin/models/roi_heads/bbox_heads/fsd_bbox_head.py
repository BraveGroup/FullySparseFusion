import numpy as np
import torch
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn

from mmdet3d.core.bbox.structures import (LiDARInstance3DBoxes,
                                          rotation_3d_in_axis, xywhr2xyxyr)
from mmdet3d.models.builder import build_loss
from projects.mmdet3d_plugin.ops import build_mlp
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet.models import HEADS

from mmdet3d.models import builder

# from ipdb import set_trace
# from projects.mmdet3d_plugin.core.bbox.util import TorchTimer
# timer = TorchTimer(-1)

@HEADS.register_module()
class FullySparseBboxHead(BaseModule):
    def __init__(self,
                 num_classes,
                 num_blocks,
                 in_channels, 
                 feat_channels,
                 with_distance,
                 with_cluster_center,
                 with_rel_mlp,
                 rel_mlp_hidden_dims,
                 rel_mlp_in_channels,
                 reg_mlp,
                 cls_mlp,
                 mode='max',
                 xyz_normalizer=[20, 20, 4],
                 cat_voxel_feats=True,
                 pos_fusion='mul',
                 fusion='cat',
                 act='gelu',
                 geo_input=True,
                 use_middle_cluster_feature=True,
                 norm_cfg=dict(type='LN', eps=1e-3, momentum=0.01),
                 dropout=0,
                 unique_once=False,
                 init_cfg=None,
                 no_head=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.geo_input = geo_input

        self.num_blocks = num_blocks
        self.use_middle_cluster_feature = use_middle_cluster_feature
        self.print_info = {}
        self.unique_once = unique_once
        
        block_list = []
        for i in range(num_blocks):
            return_point_feats = i != num_blocks-1
            kwargs = dict(
                type='DynamicClusterVFE',
                in_channels=in_channels[i],
                feat_channels=feat_channels[i],
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_rel_mlp=with_rel_mlp,
                rel_mlp_hidden_dims=rel_mlp_hidden_dims[i],
                rel_mlp_in_channel=rel_mlp_in_channels[i],
                with_voxel_center=False,
                voxel_size=[0.1, 0.1, 0.1], # not used, placeholder
                point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4], # not used, placeholder
                norm_cfg=norm_cfg,
                mode=mode,
                fusion_layer=None,
                return_point_feats=return_point_feats,
                return_inv=False,
                rel_dist_scaler=10.0,
                fusion=fusion,
                pos_fusion=pos_fusion,
                xyz_normalizer=xyz_normalizer,
                cat_voxel_feats=cat_voxel_feats,
                act=act,
                dropout=dropout,
            )
            encoder = builder.build_voxel_encoder(kwargs)
            block_list.append(encoder)
        self.block_list = nn.ModuleList(block_list)


    def init_weights(self):
        super().init_weights()
        # normal_init(self.conv_reg[-1].conv, mean=0, std=0.001)

    @force_fp32(apply_to=('pts_features', 'rois'))
    def forward(self, pts_xyz, pts_features, pts_info, roi_inds, rois):
        """Forward pass.

        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.

        Returns:
            tuple[torch.Tensor]: Score of class and bbox predictions.
        """
        assert pts_features.size(0) > 0

        rois_batch_idx = rois[:, 0]
        real_batch_size = rois_batch_idx.max().item() + 1
        rois = rois[:, 1:]
        roi_centers = rois[:, :3]
        rel_xyz = pts_xyz[:, :3] - roi_centers[roi_inds] 

        if self.unique_once:
            new_coors, unq_inv = torch.unique(roi_inds, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv = None


        point_feat_list = []
        out_feats = pts_features
        f_cluster = torch.cat([pts_info['local_xyz'], pts_info['boundary_offset'], pts_info['is_in_margin'][:, None], rel_xyz], dim=-1)

        cluster_feat_list = []
        for i, block in enumerate(self.block_list):

            in_feats = torch.cat([pts_xyz, out_feats], 1)

            if self.geo_input:
                in_feats = torch.cat([in_feats, f_cluster / 10], 1)

            if i < self.num_blocks - 1:
                # return point features
                # in_feats:  torch.Size([1026, 274])
                out_feats, out_cluster_feats = block(in_feats, roi_inds, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                if self.use_middle_cluster_feature:
                    cluster_feat_list.append(out_cluster_feats)
            if i == self.num_blocks - 1:
                # return group features
                out_cluster_feats, out_coors = block(in_feats, roi_inds, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            
        final_cluster_feats = torch.cat(cluster_feat_list, dim=1)

        if self.training and (out_coors == -1).any():
            assert out_coors[0].item() == -1, 'This should be hold due to sorted=True in torch.unique'

        nonempty_roi_mask = self.get_nonempty_roi_mask(out_coors, len(rois))
        final_cluster_feats = self.align_roi_feature_and_rois(final_cluster_feats, out_coors, len(rois))

        return final_cluster_feats, nonempty_roi_mask
    
    def get_nonempty_roi_mask(self, out_coors, num_rois):
        if self.training:
            assert out_coors.max() + 1 <= num_rois
            assert out_coors.ndim == 1
            assert torch.unique(out_coors).size(0) == out_coors.size(0)
            assert (out_coors == torch.sort(out_coors)[0]).all()
        out_coors = out_coors[out_coors >= 0]
        nonempty_roi_mask = torch.zeros(num_rois, dtype=torch.bool, device=out_coors.device)
        nonempty_roi_mask[out_coors] = True
        return nonempty_roi_mask
    
    def pad_pred_to_roi_size(self, cls_score, bbox_pred, nonempty_roi_mask, out_coors):
        num_all_rois = nonempty_roi_mask.size(0)
        new_cls_score = cls_score.new_zeros((num_all_rois, cls_score.size(1)))
        out_coors = out_coors[out_coors >= 0]
        # a tricky pytorch feature: a[all_false_mask] = b does not raise any error
        try:
            new_cls_score[nonempty_roi_mask] = cls_score
        except RuntimeError:
            set_trace()

        new_bbox_pred = bbox_pred.new_ones((num_all_rois, bbox_pred.size(1))) * -1
        new_bbox_pred[nonempty_roi_mask] = bbox_pred
        return new_cls_score, new_bbox_pred

    def align_roi_feature_and_rois(self, features, out_coors, num_rois):
        """
        1. The order of roi features obtained by dynamic pooling may not align with rois
        2. Even if we set sorted=True in torch.unique, the empty group (with idx -1) will be the first feature, causing misaligned
        So here we explicitly align them to make sure the sanity
        """
        new_feature = features.new_zeros((num_rois, features.size(1)))
        coors_mask = out_coors >= 0

        if not coors_mask.any():
            new_feature[:len(features), :] = features * 0 # pseudo gradient, avoid unused_parameters
            return new_feature

        nonempty_coors = out_coors[coors_mask]
        nonempty_feats = features[coors_mask]

        new_feature[nonempty_coors] = nonempty_feats

        return new_feature


import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class ABSPointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.
    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 post_center_range=None,
                 num_classes=10,
                 max_num=500,
                 code_size=10,
                 xy_factor=20.0,
                 z_factor=4.0):

        self.post_center_range = post_center_range
        self.code_size = code_size
        self.EPS = 1e-6
        self.num_classes = num_classes
        self.max_num = max_num
        self.xy_factor = xy_factor
        self.z_factor = z_factor

    def encode(self, bboxes):
        """
        Get regress target given bboxes and corresponding base_points
        """
        dtype = bboxes.dtype
        device = bboxes.device

        assert bboxes.size(1) in (7, 9, 10), f'bboxes shape: {bboxes.shape}'
        xyz = bboxes[:,:3].clone()
        dims = bboxes[:, 3:6]
        yaw = bboxes[:, 6:7]

        xyz[:, :2] /= self.xy_factor
        xyz[:, 2] /= self.z_factor
        log_dims = (dims + self.EPS).log()

        delta = xyz # / self.window_size_meter
        reg_target = torch.cat([delta, log_dims, yaw.sin(), yaw.cos()], dim=1)
        if bboxes.size(1) in (9, 10): # with velocity or copypaste flag
            assert self.code_size == 10
            reg_target = torch.cat([reg_target, bboxes[:, [7, 8]]], dim=1)
        return reg_target

    def decode(self, reg_preds, detach_yaw=False):

        assert reg_preds.size(1) in (8, 10)
        assert reg_preds.size(1) == self.code_size

        if self.code_size == 10:
            velo = reg_preds[:, -2:]
            reg_preds = reg_preds[:, :8] # remove the velocity

        xyz = reg_preds[:, :3].clone()
        xyz[:, :2] *= self.xy_factor
        xyz[:, 2] *= self.z_factor

        dims = reg_preds[:, 3:6].exp() - self.EPS

        sin = reg_preds[:, 6:7]
        cos = reg_preds[:, 7:8]
        yaw = torch.atan2(sin, cos)
        if detach_yaw:
            yaw = yaw.clone().detach()
        bboxes = torch.cat([xyz, dims, yaw], dim=1)
        if self.code_size == 10:
            bboxes = torch.cat([bboxes, velo], dim=1)
        return bboxes

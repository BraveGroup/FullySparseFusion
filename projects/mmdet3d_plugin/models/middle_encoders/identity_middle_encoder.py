import torch.nn as nn
from mmdet3d.models.builder import MIDDLE_ENCODERS

@MIDDLE_ENCODERS.register_module()
class IdentityMiddleEncoder(nn.Module):
    """
    Keep consistency with DynamicVoxelNet and for future usage.
    """

    def __init__(self):
        super().__init__()

    def forward(self, voxel_feats, voxel_coors, batch_size):
        '''
        '''
        return voxel_feats, voxel_coors, batch_size
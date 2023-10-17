import torch
from torch import Tensor
from math import pi as PI

LABEL_ATTR = (
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
)

@torch.jit.script
def xyz_to_quat(xyz_rad: Tensor) -> Tensor:
    """Convert euler angles (xyz - pitch, roll, yaw) to scalar first quaternions.

    Args:
        xyz_rad: (...,3) Tensor of roll, pitch, and yaw in radians.

    Returns:
        (...,4) Scalar first quaternions (wxyz).
    """
    x_rad = xyz_rad[..., 0]
    y_rad = xyz_rad[..., 1]
    z_rad = xyz_rad[..., 2]

    cy = torch.cos(z_rad * 0.5)
    sy = torch.sin(z_rad * 0.5)
    cp = torch.cos(y_rad * 0.5)
    sp = torch.sin(y_rad * 0.5)
    cr = torch.cos(x_rad * 0.5)
    sr = torch.sin(x_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat_wxyz = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_wxyz
    
@torch.jit.script
def yaw_to_quat(yaw_rad: Tensor) -> Tensor:
    """Convert yaw (rotation about the vertical axis) to scalar first quaternions.

    Args:
        yaw_rad: (...,1) Rotations about the z-axis.

    Returns:
        (...,4) scalar first quaternions (wxyz).
    """
    xyz_rad = torch.zeros_like(yaw_rad)[..., None].repeat_interleave(3, dim=-1)
    xyz_rad[..., -1] = yaw_rad
    quat_wxyz: Tensor = xyz_to_quat(xyz_rad)
    return quat_wxyz

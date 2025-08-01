#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def transform_rotation_scaling(_rotation, _scaling, transform):
    """
    Apply SE(3) transform to Gaussian _rotation and _scaling parameters.

    Args:
        _rotation: (N, 4) quaternion
        _scaling: (N, 3) log-scale
        transform: (4, 4) SE(3) matrix

    Returns:
        new_rotation: (N, 4) quaternion
        new_scaling: (N, 3) log-scale
    """
    # Step 1: convert quaternion to rotation matrix
    R_local = quaternion_to_matrix(_rotation)  # (N, 3, 3)

    # Step 2: build diagonal scale matrix
    scaling = torch.exp(_scaling)  # (N, 3)
    S = torch.diag_embed(scaling**2)  # (N, 3, 3)

    # Step 3: local covariance
    cov_local = R_local @ S @ R_local.transpose(1, 2)  # (N, 3, 3)

    # Step 4: extract global rotation
    R_global = transform[:3, :3]  # (3, 3)

    # Step 5: apply global rotation: Σ' = Rg * Σ * Rg^T
    cov_global = R_global @ cov_local @ R_global.T  # (N, 3, 3)

    # Step 6: SVD decomposition
    U, S_diag, _ = torch.linalg.svd(cov_global)
    print(U, S_diag, cov_global)
    new_rotation_mat = U  # (N, 3, 3)
    new_scaling = torch.sqrt(S_diag)  # (N, 3)

    # Step 7: convert rotation matrix to quaternion
    new_rotation = matrix_to_quaternion(new_rotation_mat)  # (N, 4)
    new_scaling_log = torch.log(new_scaling)

    return new_rotation, new_scaling_log


def geom_transform_quat(quats, transf_matrix):
    assert quats.shape[1] == 4 and transf_matrix.shape == (4, 4)

    # 1. 提取旋转部分
    R_np = transf_matrix[:3, :3].cpu().numpy()
    R_quat_np = R.from_matrix(R_np).as_quat()  # [x, y, z, w]
    R_quat = torch.from_numpy(R_quat_np).to(quats.device).float()

    # 2. 四元数运算函数
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    def quat_conjugate(q):
        return torch.cat([-q[:, :3], q[:, 3:]], dim=-1)

    # 3. 进行共轭变换：q' = R * q * R⁻¹
    R_quat_batch = R_quat.unsqueeze(0).expand(quats.shape[0], 4)
    R_conj = quat_conjugate(R_quat_batch)
    q_out = quat_multiply(quat_multiply(R_quat_batch, quats), R_conj)

    # 4. 归一化结果
    q_out = q_out / q_out.norm(dim=-1, keepdim=True)
    return q_out


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(
    znear,
    zfar,
    fovX,
    fovY,
    w=None,
    h=None,
    cx=None,
    cy=None,
    allow_principle_point_shift=True,
):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    if (
        w != None
        and h != None
        and cx != None
        and cy != None
        and allow_principle_point_shift
    ):
        # shift the frame window due to the non-zero principle point offsets
        focal_x = w / (2 * math.tan(fovX / 2))
        focal_y = h / (2 * math.tan(fovY / 2))

        offset_x = cx - (w / 2)
        offset_x = (offset_x / focal_x) * znear
        offset_y = cy - (h / 2)
        offset_y = (offset_y / focal_y) * znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

        # aspect_ratio = w / h
        # cy_offset = (h / 2 - cy) / (h / 2) * tanHalfFovY * znear
        # cx_offset = (w / 2 - cx) / (w / 2) * tanHalfFovX * znear

        # top = tanHalfFovY * znear + cy_offset
        # bottom = -tanHalfFovY * znear + cy_offset
        # right = tanHalfFovX * znear + cx_offset
        # left = -tanHalfFovX * znear + cx_offset

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

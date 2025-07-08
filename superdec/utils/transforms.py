import torch

def transform_to_primitive_frame(pc_or_normals, trans, rotate):
    B, N, _ = pc_or_normals.shape
    P = trans.shape[1]

    centered = pc_or_normals.unsqueeze(1).repeat(1, P, 1, 1) - trans.unsqueeze(2).repeat(1,1,N,1)
    centered = centered.permute(0,1,3,2)
    rotate_T = rotate.permute(0, 1, 3, 2)
    transformed = torch.einsum('abcd,abde->abce', rotate_T, centered).permute(0,1,3,2)
    return transformed

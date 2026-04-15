import torch

H = 986
W = 1400

def compute_ADE(pred, gt):
    # normalize coordinates to [0,1]
    pred_norm = pred.clone()
    gt_norm = gt.clone()

    pred_norm[..., 0] /= W
    pred_norm[..., 1] /= H
    gt_norm[..., 0] /= W
    gt_norm[..., 1] /= H

    return torch.norm(pred_norm - gt_norm, dim=-1).mean()

def compute_FDE(pred, gt):
    pred_norm = pred.clone()
    gt_norm = gt.clone()

    pred_norm[..., 0] /= W
    pred_norm[..., 1] /= H
    gt_norm[..., 0] /= W
    gt_norm[..., 1] /= H

    return torch.norm(pred_norm[:, -1] - gt_norm[:, -1], dim=-1).mean()

def compute_velocity_error(pred, gt, reduction="mean"):
    """
    pred, gt: (B, P, N, 2)

    returns scalar (default) or tensor depending on reduction
    """
    pred_vel = pred[:, 1:] - pred[:, :-1]   # (B, P-1, N, 2)
    gt_vel   = gt[:, 1:] - gt[:, :-1]

    vel_err = torch.norm(pred_vel - gt_vel, dim=-1)  # (B, P-1, N)

    if reduction == "mean":
        return vel_err.mean()
    elif reduction == "none":
        return vel_err
    else:
        raise ValueError
    
def compute_direction_error(pred, gt, eps=1e-8, reduction="mean", motion_thresh=1e-4):
    pred_vel = pred[:, 1:] - pred[:, :-1]   # (B, P-1, N, 2)
    gt_vel   = gt[:, 1:] - gt[:, :-1]

    pred_norm = torch.norm(pred_vel, dim=-1)
    gt_norm   = torch.norm(gt_vel, dim=-1)

    valid = gt_norm > motion_thresh

    pred_unit = pred_vel / (pred_norm.unsqueeze(-1) + eps)
    gt_unit   = gt_vel / (gt_norm.unsqueeze(-1) + eps)

    cos = (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0, 1.0)
    dir_err = 1.0 - cos

    if reduction == "mean":
        if valid.any():
            return dir_err[valid].mean()
        else:
            return torch.tensor(0.0, device=pred.device)
    elif reduction == "none":
        return dir_err, valid
    else:
        raise ValueError

def compute_path_length_error(pred, gt, reduction="mean"):
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel   = gt[:, 1:] - gt[:, :-1]

    pred_len = torch.norm(pred_vel, dim=-1).sum(dim=1)  # (B,N)
    gt_len   = torch.norm(gt_vel, dim=-1).sum(dim=1)

    err = torch.abs(pred_len - gt_len)

    if reduction == "mean":
        return err.mean()
    elif reduction == "none":
        return err
    else:
        raise ValueError
    
def _as_delta_pred(model_out):
    """
    model_out can be:
      (B,T,N,2) or (B,T,N,4) or (B,T,1,N,2/4)
    return mu: (B,T,N,2)
    """
    if model_out.ndim == 5:
        model_out = model_out.squeeze(2)  # (B,T,N,C)
    C = model_out.shape[-1]
    if C == 2:
        return model_out
    elif C == 4:
        return model_out[..., 0:2]  # mu only
    else:
        raise RuntimeError(f"Unexpected model output last dim C={C}")
    

def add_virtual_root_from_xy(feat):
    """
    feat: (B,T,M,5,3) where last dim = [x, y, valid]
    returns: (B,T,M,6,3)
    root = mean of valid keypoints
    """
    xy = feat[..., :2]                    # (B,T,M,5,2)
    valid = feat[..., 2] > 0.5            # (B,T,M,5)

    valid_f = valid.unsqueeze(-1).float() # (B,T,M,5,1)
    denom = valid_f.sum(dim=3, keepdim=True).clamp_min(1.0)
    root_xy = (xy * valid_f).sum(dim=3, keepdim=True) / denom   # (B,T,M,1,2)
    root_valid = valid.any(dim=3, keepdim=True).float().unsqueeze(-1)  # (B,T,M,1,1)

    root = torch.cat([root_xy, root_valid], dim=-1)             # (B,T,M,1,3)
    return torch.cat([feat, root], dim=3)                       # (B,T,M,6,3)

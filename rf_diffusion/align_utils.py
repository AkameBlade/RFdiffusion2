import torch
import time
import logging
from rf_diffusion.chemical import ChemicalData as ChemData

logger = logging.getLogger(__name__)


def kabsch_rotation_translation(P, Q):
    """
    Compute optimal rotation (R) and translation (t) that aligns P->Q using Kabsch.
    P, Q: (N, 3) torch tensors.
    Returns R (3,3) and t (3,).
    """
    assert P.shape == Q.shape and P.shape[-1] == 3
    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    H = P_centered.T.mm(Q_centered)
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = (Q_mean - P_mean @ R).squeeze(0)
    return R, t, P_mean, Q_mean


def residue_centroids(xyz, idxs):
    """
    Compute per-residue centroids over heavy atoms for given residue indices.
    xyz: (L, N, 3); idxs: 1D Long/np array of residue indices
    Returns: (K, 3)
    """
    if len(idxs) == 0:
        return torch.empty((0, 3), dtype=xyz.dtype, device=xyz.device)
    sel = xyz[idxs, :, :]
    return torch.nanmean(sel, dim=1)


def scheduled_alpha(it, total_steps, alpha0=1.0, mode='linear'):
    """
    Return fraction in [0, alpha0] that typically decays as time advances.
    it: 0..total_steps-1, progress increases; mode: 'linear' or 'cosine'.
    """
    if total_steps <= 1:
        return alpha0
    progress = it / float(total_steps - 1)
    if mode == 'cosine':
        import math
        frac = (1.0 - math.sin(math.pi /2.0 * progress))
    else:  # linear decay
        frac = 1.0 - progress
    return float(alpha0) * max(0.0, min(1.0, frac))





def compute_alignment_params(px0, indep, contig_map, match_idx, gp_idx, it, total_steps, align_conf):
    """
    Return (R, t, alpha, move_mask) if alignment is enabled and valid; else None.
    """
    if not (align_conf and getattr(align_conf, 'active', False)):
        return None
    if len(match_idx) == 0 or len(gp_idx) == 0:
        return None

    # Use current indep.xyz frame to compute alignment
    frame = indep.xyz
    # Use only the first 4 atoms per residue (typically backbone: N, CA, C, O)
    gp_sel = indep.xyz[torch.as_tensor(gp_idx, device=frame.device), :5, :]
    mt_sel = px0[torch.as_tensor(match_idx, device=frame.device), :5, :]
    valid = (~torch.isnan(gp_sel).any(dim=-1)) & (~torch.isnan(mt_sel).any(dim=-1))
    if valid.sum().item() < 3:
        return None
    P = gp_sel[valid].reshape(-1, 3)
    Q = mt_sel[valid].reshape(-1, 3)
    R, tt, P_mean, Q_mean = kabsch_rotation_translation(P, Q)
    alpha = scheduled_alpha(it, total_steps, getattr(align_conf, 'alpha0', 1.0), getattr(align_conf, 'schedule', 'linear'))
    if alpha <= 0:
        return None
    gp_mask = torch.zeros(indep.length(), dtype=bool, device=indep.xyz.device)
    gp_mask[torch.as_tensor(gp_idx, device=gp_mask.device)] = True

    substrate_mask = contig_map.substrate_mask
    expanded_substrate_mask = torch.cat([substrate_mask, torch.zeros(gp_mask.size(0) - substrate_mask.size(0), dtype=torch.bool)])
    move_mask = gp_mask | expanded_substrate_mask
    return R, tt, float(alpha), move_mask, P_mean, Q_mean


def apply_transform_fractional(xyz, move_mask, R, t, alpha, P_mean, Q_mean, use_quaternions=True):
    """
    Apply fractional rigid transform with proper rotation/translation parameterization.
    
    Args:
        xyz: (L, N, 3) coordinates
        move_mask: (L,) boolean mask for residues to move
        R: (3, 3) rotation matrix
        t: (3,) translation vector (already rotated)
        alpha: interpolation factor [0, 1]
        P_mean: (3,) source center
        Q_mean: (3,) target center
        use_quaternions: if True, use quaternion SLERP; if False, use axis-angle
    
    Returns:
        Modified xyz tensor
    """
    if alpha <= 0:
        return xyz
    if move_mask is None or (~move_mask).all():
        return xyz
    xyz_sel = xyz[move_mask, :5,:]
    xyz_flat = xyz_sel.reshape(-1, 3)
    
    # Center coordinates around P_mean for proper rotation
    xyz_centered = xyz_flat - P_mean
    
    if use_quaternions:
        # Quaternion-based interpolation (SLERP)
        R_alpha = _rotation_matrix_slerp(R, alpha)
    else:
        # Axis-angle based interpolation
        R_alpha = _rotation_matrix_axis_angle(R, alpha)
    
    # Apply fractional rotation and translation
    xyz_rotated = xyz_centered @ R_alpha
    xyz_final = xyz_rotated + alpha * Q_mean + (1 - alpha) * P_mean
    
    xyz[move_mask, :5,:] = xyz_final.reshape_as(xyz_sel)
    return xyz


def _rotation_matrix_axis_angle(R, alpha):
    """
    Compute R_alpha = exp(alpha * log(R)) using axis-angle representation.
    """
    # Compute axis-angle representation of rotation
    # R = exp([w]_×) where w is axis-angle vector
    trace = torch.trace(R)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)  # Numerical stability
    
    if torch.abs(cos_angle - 1) < 1e-6:
        # Identity rotation case
        return torch.eye(3, device=R.device, dtype=R.dtype)
    
    # Compute axis-angle vector
    angle = torch.acos(cos_angle)
    axis = torch.stack([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0], 
        R[1, 0] - R[0, 1]
    ]) / (2 * torch.sin(angle))
    
    # Scale rotation by alpha: R_alpha = exp(alpha * angle * axis)
    alpha_angle = alpha * angle
    alpha_axis = axis
    
    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -alpha_axis[2], alpha_axis[1]],
        [alpha_axis[2], 0, -alpha_axis[0]],
        [-alpha_axis[1], alpha_axis[0], 0]
    ], device=R.device, dtype=R.dtype)
    
    R_alpha = torch.eye(3, device=R.device, dtype=R.dtype) + \
              torch.sin(alpha_angle) * K + \
              (1 - torch.cos(alpha_angle)) * K @ K
    
    return R_alpha


def _rotation_matrix_slerp(R, alpha):
    """
    Compute R_alpha using quaternion SLERP (Spherical Linear Interpolation).
    """
    # Convert rotation matrix to quaternion
    q = _matrix_to_quaternion(R)
    
    # Identity quaternion
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=R.device, dtype=R.dtype)
    
    # SLERP between identity and target quaternion
    q_alpha = _quaternion_slerp(q_identity, q, alpha)
    
    # Convert back to rotation matrix
    return _quaternion_to_matrix(q_alpha)


def _matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    """
    trace = torch.trace(R)
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return torch.stack([w, x, y, z])


def _quaternion_to_matrix(q):
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Normalize quaternion
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = torch.tensor([
        [1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
    ], device=q.device, dtype=q.dtype)
    
    return R


def _quaternion_slerp(q1, q2, t):
    """
    Spherical Linear Interpolation between two quaternions.
    """
    # Normalize quaternions
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)
    
    # Compute dot product
    dot = torch.dot(q1, q2)
    
    # If dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / torch.norm(result)
    
    # Calculate angle between quaternions
    theta_0 = torch.acos(torch.abs(dot))
    sin_theta_0 = torch.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2



import torch

def get_coef_dict(motion_coef, shape_coef=None, denorm_stats=None, with_global_pose=False, rot_repr='aa'):
    coef_dict = {
        'exp': motion_coef[..., :50]
    }
    if rot_repr == 'aa':
        if with_global_pose:
            coef_dict['pose'] = motion_coef[..., 50:]
        else:
            placeholder = torch.zeros_like(motion_coef[..., :3])
            coef_dict['pose'] = torch.cat([placeholder, motion_coef[..., -1:]], dim=-1)
        # Add back rotation around y, z axis
        coef_dict['pose'] = torch.cat([coef_dict['pose'], torch.zeros_like(motion_coef[..., :2])], dim=-1)
    else:
        raise ValueError(f'Unknown rotation representation {rot_repr}!')

    if shape_coef is not None:
        if motion_coef.ndim == 3:
            if shape_coef.ndim == 2:
                shape_coef = shape_coef.unsqueeze(1)
            if shape_coef.shape[1] == 1:
                shape_coef = shape_coef.expand(-1, motion_coef.shape[1], -1)

        coef_dict['shape'] = shape_coef

    if denorm_stats is not None:
        coef_dict = {k: coef_dict[k] * denorm_stats[f'{k}_std'].to(shape_coef.device) + denorm_stats[f'{k}_mean'].to(shape_coef.device) for k in coef_dict}

    if not with_global_pose:
        if rot_repr == 'aa':
            coef_dict['pose'][..., :3] = 0
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')

    return coef_dict

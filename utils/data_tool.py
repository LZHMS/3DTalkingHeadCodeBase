import torch


def _truncate_audio(audio, end_idx, pad_mode='zero'):
    batch_size = audio.shape[0]
    audio_trunc = audio.clone()
    if pad_mode == 'replicate':
        for i in range(batch_size):
            audio_trunc[i, end_idx[i]:] = audio_trunc[i, end_idx[i] - 1]
    elif pad_mode == 'zero':
        for i in range(batch_size):
            audio_trunc[i, end_idx[i]:] = 0
    else:
        raise ValueError(f'Unknown pad mode {pad_mode}!')

    return audio_trunc


def _truncate_coef_dict(coef_dict, end_idx, pad_mode='zero'):
    batch_size = coef_dict['exp'].shape[0]
    coef_dict_trunc = {k: v.clone() for k, v in coef_dict.items()}
    if pad_mode == 'replicate':
        for i in range(batch_size):
            for k in coef_dict_trunc:
                coef_dict_trunc[k][i, end_idx[i]:] = coef_dict_trunc[k][i, end_idx[i] - 1]
    elif pad_mode == 'zero':
        for i in range(batch_size):
            for k in coef_dict:
                coef_dict_trunc[k][i, end_idx[i]:] = 0
    else:
        raise ValueError(f'Unknown pad mode: {pad_mode}!')

    return coef_dict_trunc


def truncate_coef_dict_and_audio(audio, coef_dict, n_motions, audio_unit=640, pad_mode='zero'):
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
    audio_end_idx = (end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio
    audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # truncate coef dict
    coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)

    return audio_trunc, coef_dict_trunc, end_idx


def truncate_motion_coef_and_audio(audio, motion_coef, n_motions, audio_unit=640, pad_mode='zero'):
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
    audio_end_idx = (end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio
    audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # prepare coef dict and stats
    coef_dict = {'exp': motion_coef[..., :50], 'pose_any': motion_coef[..., 50:]}

    # truncate coef dict
    coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)
    motion_coef_trunc = torch.cat([coef_dict_trunc['exp'], coef_dict_trunc['pose_any']], dim=-1)

    return audio_trunc, motion_coef_trunc, end_idx
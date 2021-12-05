def mask_padding(x, pad_idx=0):
    mask = x.eq(pad_idx)
    return mask.unsqueeze(1).expand(-1, x.shape[1], -1)

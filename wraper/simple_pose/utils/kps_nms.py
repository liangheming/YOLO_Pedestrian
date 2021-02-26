import torch


def oks_iou(pick_kps, candi_kps, pick_area, candi_area, sigmas=None, in_vis_thresh=None):
    """
    :param pick_kps:[kp_num,3]
    :param candi_kps:[gt_num,kp_num,3]
    :param pick_area:
    :param candi_area:
    :param sigmas:
    :param in_vis_thresh:
    :return:
    """
    if not isinstance(sigmas, torch.Tensor):
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89],
                              device=pick_kps.device) / 10.0
    var = (sigmas * 2) ** 2
    xg = pick_kps[:, 0]
    yg = pick_kps[:, 1]
    vg = pick_kps[:, 2]
    xds = candi_kps[..., 0]
    yds = candi_kps[..., 1]
    vds = candi_kps[..., 2]

    dx = xds - xg
    dy = yds - yg
    e = (dx ** 2 + dy ** 2) / var / ((pick_area + candi_area)[:, None] / 2 + 1e-12) / 2
    vd_vis = torch.ones_like(vds)
    if in_vis_thresh is not None:
        vg_vis = ((vg > in_vis_thresh)[None, :]).repeat(vds.shape[0], 1)
        vd_vis = ((vds > in_vis_thresh) & vg_vis).float()
    ious = ((-e).exp() * vd_vis).sum(-1) / (vd_vis.sum(-1) + 1e-12)
    return ious


def oks_nms(kps, scores, areas, thresh, sigmas=None, in_vis_thresh=None):
    """
    :param kps:[gt_num,kp_num,3] (x1,y1,conf)
    :param scores:[gt_num]
    :param areas:[gt_num]
    :param thresh:
    :param sigmas:
    :param in_vis_thresh:
    :return:
    """
    order = scores.argsort(descending=True)
    keep = list()
    while len(order) > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]
        if len(order) == 0:
            break
        oks_ovr = oks_iou(kps[pick_idx], kps[order], areas[pick_idx], areas[order], sigmas, in_vis_thresh)
        order = order[oks_ovr <= thresh]
    return torch.stack(keep)

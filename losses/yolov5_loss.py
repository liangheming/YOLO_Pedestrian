import torch
from torch import nn
from losses.commons import BoxSimilarity


class PedYOLOv5Builder(object):
    def __init__(self, ratio_thresh=4., expansion_bias=0.5):
        super(PedYOLOv5Builder, self).__init__()
        self.ratio_thresh = ratio_thresh
        self.expansion_bias = expansion_bias

    def __call__(self, predicts, targets, wh_anchors):
        """
        :param predicts:[bs,anchor_per_grid,ny,nx,5]
        :param targets:[bid,xc,yc,w,h]
        :param wh_anchors:[layer_num,anchor_per_grid,2]
        :return:
        """
        device = predicts[0].device
        num_layer, anchor_per_grid = wh_anchors.shape[:2]
        num_gt = targets.shape[0]
        target_box, target_indices, target_anchors = list(), list(), list()
        gain = torch.ones(6, device=device).float()
        # [anchor_per_grid,gt_num]
        anchor_idx = torch.arange(anchor_per_grid, device=device).float().view(anchor_per_grid, 1).repeat(1, num_gt)
        # [anchor_per_grid,gt_num,5]
        targets = torch.cat((targets.repeat(anchor_per_grid, 1, 1), anchor_idx[:, :, None]), dim=2)
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * self.expansion_bias
        for i in range(num_layer):
            anchors = wh_anchors[i]
            gain[1:-1] = torch.tensor(predicts[i].shape)[[3, 2, 3, 2]]
            t = targets * gain
            if num_gt:
                r = t[:, :, 3:5] / anchors[:, None, :]
                valid_idx = torch.max(r, 1. / r).max(2)[0] < self.ratio_thresh
                t = t[valid_idx]
                gxy = t[:, 1:3]
                gxy_flip = gain[1:3] - gxy
                j, k = ((gxy % 1. < self.expansion_bias) & (gxy > 1.)).T
                l, m = ((gxy_flip % 1. < self.expansion_bias) & (gxy_flip > 1.)).T
                gain_valid_idx = torch.stack([torch.ones_like(j), j, k, l, m])
                t = t.repeat((5, 1, 1))[gain_valid_idx]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[gain_valid_idx]
            else:
                t = targets[0]
                offsets = 0

            b, a = t[:, [0, 5]].long().T
            gt_xy = t[:, 1:3]
            gt_wh = t[:, 3:5]
            gij = (gt_xy - offsets).long()
            gi, gj = gij.T
            target_indices.append((b, a, gj, gi))
            target_box.append(torch.cat((gt_xy - gij, gt_wh), dim=1))
            target_anchors.append(anchors[a])
        return target_indices, target_box, target_anchors


class PedYOLOv5Loss(object):
    def __init__(self,
                 ratio_thresh=4.,
                 expansion_bias=0.5,
                 layer_balance=None,
                 obj_pw=1.0,
                 iou_type="giou",
                 coord_type="xywh",
                 iou_ratio=1.0,
                 iou_weights=0.05,
                 obj_weights=1.2):
        super(PedYOLOv5Loss, self).__init__()
        if layer_balance is None:
            layer_balance = [4.0, 1.0, 0.4]
        self.layer_balance = layer_balance
        self.iou_ratio = iou_ratio
        self.iou_weights = iou_weights
        self.obj_weights = obj_weights
        self.expansion_bias = expansion_bias
        self.box_similarity = BoxSimilarity(iou_type, coord_type)
        self.target_builder = PedYOLOv5Builder(ratio_thresh, expansion_bias)
        self.obj_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data=[obj_pw]))

    def __call__(self, predicts, targets, wh_anchors):
        device = predicts[0].device
        if self.obj_bce.pos_weight.device != device:
            self.obj_bce.to(device)
        loss_box, loss_obj = torch.zeros(1, device=device), torch.zeros(1, device=device)
        target_indices, target_box, target_anchors = self.target_builder(predicts, targets, wh_anchors)

        num_bs = predicts[0].shape[0]
        match_num = 0
        for i, pi in enumerate(predicts):
            b, a, gj, gi = target_indices[i]
            target_obj = torch.zeros_like(pi[..., 0], device=device)
            n = len(b)
            if n:
                match_num += n
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2. - self.expansion_bias
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * target_anchors[i]
                pbox = torch.cat((pxy, pwh), dim=1).to(device)
                box_sim = self.box_similarity(pbox, target_box[i])
                loss_box += (1.0 - box_sim).mean()
                target_obj[b, a, gj, gi] = \
                    (1.0 - self.iou_ratio) + self.iou_ratio * box_sim.detach().clamp(0).type(target_obj.dtype)
            loss_obj += self.obj_bce(pi[..., 4], target_obj) * self.layer_balance[i]
        loss_box = loss_box * self.iou_weights
        loss_obj = loss_obj * self.obj_weights
        return loss_box * num_bs, loss_obj * num_bs

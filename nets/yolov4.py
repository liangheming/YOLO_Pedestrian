import math
import torch
from torch import nn
from nets.commons import CBR, BottleNeck, BottleNeckCSP, BottleNeckCSP2, SPPCSP, width_grow, depth_grow, model_scale
from nets.activations import MemoryEfficientMish
from nets.commons import fuse_conv_and_bn
from losses.yolov5_loss import PedYOLOv5Loss
from utils.boxs_utils import non_max_suppression, clip_coords


class YOLOv4Backbone(nn.Module):
    def __init__(self, in_channel=3, depth_multiples=0.33, width_multiples=0.50, act=MemoryEfficientMish):
        super(YOLOv4Backbone, self).__init__()
        channel_32 = width_grow(32, width_multiples)
        channel_64 = width_grow(64, width_multiples)
        channel_128 = width_grow(128, width_multiples)
        channel_256 = width_grow(256, width_multiples)
        channel_512 = width_grow(512, width_multiples)
        channel_1024 = width_grow(1024, width_multiples)
        self.out_channels = [channel_128, channel_256, channel_512, channel_1024]
        self.layer1 = nn.Sequential(
            CBR(in_channel, channel_32, 3, 1, act=act),
            CBR(channel_32, channel_64, 3, 2, act=act),
            BottleNeck(channel_64, channel_64, act=act)
        )
        self.layer2 = nn.Sequential(
            CBR(channel_64, channel_128, 3, 2, act=act),
            BottleNeckCSP(channel_128, channel_128, depth_grow(2, depth_multiples), act=act)
        )
        self.layer3 = nn.Sequential(
            CBR(channel_128, channel_256, 3, 2, act=act),
            BottleNeckCSP(channel_256, channel_256, depth_grow(8, depth_multiples), act=act)
        )
        self.layer4 = nn.Sequential(
            CBR(channel_256, channel_512, 3, 2, act=act),
            BottleNeckCSP(channel_512, channel_512, depth_grow(8, depth_multiples), act=act)
        )
        self.layer5 = nn.Sequential(
            CBR(channel_512, channel_1024, 3, 2, act=act),
            BottleNeckCSP(channel_1024, channel_1024, depth_grow(4, depth_multiples), act=act)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return [c3, c4, c5]


class YOLOv4Neck(nn.Module):
    def __init__(self, c2, c3, c4, c5, blocks=1, act=MemoryEfficientMish):
        super(YOLOv4Neck, self).__init__()
        # c2=128,c3=256,c4=512,c5=1024
        self.sppcsp = SPPCSP(c5, c4, act=act)
        self.c5_latent = CBR(c4, c3, 1, 1, act=act)
        self.c4_to_f4 = CBR(c4, c3, 1, 1, act=act)
        self.f4 = BottleNeckCSP2(c3 * 2, c3, blocks=blocks, act=act, expansion=1.0)
        self.f4_latent = CBR(c3, c2, 1, 1, act=act)
        self.c3_to_f3 = CBR(c3, c2, 1, 1, act=act)
        self.f3 = BottleNeckCSP2(c2 * 2, c2, blocks=blocks, act=act, expansion=1.0)
        self.f3_out = CBR(c2, c3, 3, 1, act=act)
        self.f3_to_f4 = CBR(c2, c3, 3, 2, act=act)
        self.f4_2 = BottleNeckCSP2(c3 * 2, c3, blocks=blocks, act=act, expansion=1.0)
        self.f4_out = CBR(c3, c4, 3, 1, act=act)
        self.f4_to_f5 = CBR(c3, c4, 3, 2, act=act)
        self.f5 = BottleNeckCSP2(c4 * 2, c4, blocks=blocks, act=act, expansion=1.0)
        self.f5_out = CBR(c4, c5, 3, 1, act=act)

    def forward(self, xs):
        c3, c4, c5 = xs
        sppcsp = self.sppcsp(c5)
        f4 = self.f4(
            torch.cat([self.c4_to_f4(c4), nn.UpsamplingNearest2d(scale_factor=2)(self.c5_latent(sppcsp))], dim=1))

        f3 = self.f3(
            torch.cat([self.c3_to_f3(c3), nn.UpsamplingNearest2d(scale_factor=2)(self.f4_latent(f4))], dim=1)
        )
        f3_out = self.f3_out(f3)
        f4_2 = self.f4_2(
            torch.cat([self.f3_to_f4(f3), f4], dim=1)
        )
        f4_out = self.f4_out(f4_2)
        f5 = self.f5(
            torch.cat([self.f4_to_f5(f4_2), sppcsp], dim=1)
        )
        f5_out = self.f5_out(f5)
        return [f3_out, f4_out, f5_out]


class YOLOv4Head(nn.Module):
    def __init__(self, c3, c4, c5, num_cls=80, strides=None, anchors=None):
        super(YOLOv4Head, self).__init__()
        self.num_cls = num_cls
        self.output_num = num_cls + 5 if num_cls > 1 else 5
        self.anchors = anchors
        self.strides = strides
        assert len(self.anchors) == len(self.strides)
        self.layer_num = len(self.anchors)
        self.anchor_per_grid = len(self.anchors[0]) // 2
        self.grids = [torch.zeros(1)] * self.layer_num
        a = torch.tensor(self.anchors, requires_grad=False).float().view(self.layer_num, -1, 2)
        normalize_anchors = a / torch.tensor(strides, requires_grad=False).float().view(3, 1, 1)
        self.register_buffer("normalize_anchors", normalize_anchors.clone())
        self.register_buffer("anchor_grid", a.clone().view(self.layer_num, 1, -1, 1, 1, 2))
        self.heads = nn.ModuleList(
            nn.Conv2d(x, self.output_num * self.anchor_per_grid, 1) for x in [c3, c4, c5]
        )
        for mi, s in zip(self.heads, strides):  # from
            b = mi.bias.view(self.anchor_per_grid, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8. / (640. / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.num_cls - 0.99))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xs):
        z = list()
        assert len(xs) == self.layer_num
        for i in range(self.layer_num):
            xs[i] = self.heads[i](xs[i])
            bs, _, ny, nx = xs[i].shape
            xs[i] = xs[i].view(bs, self.anchor_per_grid, self.output_num, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference
                if self.grids[i].shape[2:4] != xs[i].shape[2:4]:
                    self.grids[i] = self._make_grid(nx, ny).to(xs[i].device)
                y = xs[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grids[i]) * self.strides[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.output_num))
        return xs if self.training else torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


default_cfg = {
    # model_param
    "in_channel": 3,
    "num_cls": 1,
    "scale_name": "m",
    "strides": [8., 16., 32.],
    "anchors": [[9, 15, 16, 36, 32, 55],
                [39, 107, 70, 93, 76, 193],
                [139, 176, 161, 330, 311, 405]],

    "iou_type": "ciou",
    "iou_weights": 0.05,
    "obj_weights": 1.2,
    "iou_ratio": 1.0,

    # inference param
    "conf_thresh": 0.01,
    "iou_thresh": 0.6,
    "max_det": 300

}


class YOLOv4(nn.Module):
    def __init__(self, **cfg):
        super(YOLOv4, self).__init__()
        self.cfg = {**default_cfg, **cfg}
        depth_multiples, width_multiples = model_scale(self.cfg['scale_name'])
        act_func = MemoryEfficientMish
        self.backbones = YOLOv4Backbone(self.cfg['in_channel'], depth_multiples, width_multiples, act=act_func)
        c2, c3, c4, c5 = self.backbones.out_channels
        self.neck = YOLOv4Neck(c2, c3, c4, c5, blocks=depth_grow(2, depth_multiples))
        self.head = YOLOv4Head(c3, c4, c5,
                               num_cls=self.cfg['num_cls'],
                               strides=self.cfg['strides'],
                               anchors=self.cfg['anchors'])
        self.loss = PedYOLOv5Loss(
            iou_weights=self.cfg['iou_weights'],
            obj_weights=self.cfg['obj_weights'],
            iou_ratio=self.cfg['iou_ratio'],
            iou_type=self.cfg['iou_type']
        )

    def forward(self, x, targets=None):
        xs = self.head(self.neck(self.backbones(x)))
        if self.training:
            assert targets is not None and len(targets) > 0
            iou_loss, obj_loss = self.loss(predicts=xs, targets=targets, wh_anchors=self.head.normalize_anchors)
            return {
                "iou_loss": iou_loss,
                "obj_loss": obj_loss
            }
        else:
            shape = x.shape[-2:]
            predicts = non_max_suppression(xs, self.cfg['conf_thresh'], self.cfg['iou_thresh'], self.cfg['max_det'])
            for pred in predicts:
                if pred is not None:
                    clip_coords(pred, shape)
        return {
            "predicts": predicts
        }

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is CBR:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        return self
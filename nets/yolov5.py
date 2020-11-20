import math
import torch
from torch import nn
from nets.commons import Focus, CBR, SPP, BottleNeckCSP, width_grow, depth_grow, model_scale
from losses.yolov5_loss import PedYOLOv5Loss
from torchvision.ops import nms
from nets.commons import fuse_conv_and_bn


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def xywh2xyxy(x):
    x1y1 = x[..., [0, 1]] - x[..., [2, 3]] * 0.5
    x2y2 = x1y1 + x[..., [2, 3]]
    return torch.cat([x1y1, x2y2], dim=-1)


default_anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]
default_strides = [8., 16., 32.]


def non_max_suppression(prediction,
                        conf_thresh=0.01,
                        iou_thresh=0.6,
                        max_det=300):
    if prediction.dtype == torch.float16:
        prediction = prediction.float()
    xc = prediction[..., 4] > conf_thresh
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box = xywh2xyxy(x[:, :4])
        x = torch.cat([box, x[:, [4]]], dim=-1)
        boxes, scores = x[:, :4], x[:, 4]
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output


class YOLOv5Backbone(nn.Module):
    def __init__(self, in_channel=3, depth_multiples=0.33, width_multiples=0.50):
        super(YOLOv5Backbone, self).__init__()
        channel_64 = width_grow(64, width_multiples)
        channel_128 = width_grow(128, width_multiples)
        channel_256 = width_grow(256, width_multiples)
        channel_512 = width_grow(512, width_multiples)
        channel_1024 = width_grow(1024, width_multiples)
        self.out_channels = [channel_256, channel_512, channel_1024]
        self.stem = Focus(in_channel, channel_64, 3)
        self.layer1 = nn.Sequential(
            CBR(channel_64, channel_128, 3, 2),
            BottleNeckCSP(channel_128, channel_128, depth_grow(3, depth_multiples))
        )

        self.layer2 = nn.Sequential(
            CBR(channel_128, channel_256, 3, 2),
            BottleNeckCSP(channel_256, channel_256, depth_grow(9, depth_multiples))
        )

        self.layer3 = nn.Sequential(
            CBR(channel_256, channel_512, 3, 2),
            BottleNeckCSP(channel_512, channel_512, depth_grow(9, depth_multiples))
        )

        self.layer4 = nn.Sequential(
            CBR(channel_512, channel_1024, 3, 2),
            SPP(channel_1024, channel_1024, (5, 9, 13)),
            BottleNeckCSP(channel_1024, channel_1024, depth_grow(3, depth_multiples), shortcut=False)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class YOLOv5Neck(nn.Module):
    def __init__(self, c3, c4, c5, blocks=1):
        super(YOLOv5Neck, self).__init__()
        self.latent_c5 = CBR(c5, c4, 1, 1)
        self.c4_fuse = BottleNeckCSP(c4 * 2, c4, blocks=blocks, shortcut=False)
        self.latent_c4 = CBR(c4, c3)
        self.c3_out = BottleNeckCSP(c3 * 2, c3, blocks=blocks, shortcut=False)
        self.c3_c4 = CBR(c3, c3, 3, 2)
        self.c4_out = BottleNeckCSP(c3 * 2, c4, blocks=blocks, shortcut=False)
        self.c4_c5 = CBR(c4, c4, 3, 2)
        self.c5_out = BottleNeckCSP(c4 * 2, c5, blocks=blocks, shortcut=False)

    def forward(self, xs):
        c3, c4, c5 = xs
        latent_c5 = self.latent_c5(c5)
        f4 = torch.cat([nn.UpsamplingNearest2d(scale_factor=2)(latent_c5), c4], dim=1)
        c4_fuse = self.c4_fuse(f4)
        latent_c4 = self.latent_c4(c4_fuse)
        f3 = torch.cat([nn.UpsamplingNearest2d(scale_factor=2)(latent_c4), c3], dim=1)
        c3_out = self.c3_out(f3)
        c3_c4 = self.c3_c4(c3_out)
        c4_out = self.c4_out(torch.cat([c3_c4, latent_c4], dim=1))
        c4_c5 = self.c4_c5(c4_out)
        c5_out = self.c5_out(torch.cat([c4_c5, latent_c5], dim=1))
        return [c3_out, c4_out, c5_out]


class YOLOv5Head(nn.Module):
    def __init__(self, c3, c4, c5, num_cls=80, strides=None, anchors=None):
        super(YOLOv5Head, self).__init__()
        self.num_cls = num_cls
        self.output_num = num_cls + 5
        if self.num_cls == 1:
            self.output_num = 5
        if anchors is None:
            anchors = default_anchors
        self.anchors = anchors
        if strides is None:
            strides = default_strides
        self.strides = strides
        assert len(self.anchors) == len(self.strides)
        self.layer_num = len(self.anchors)
        self.anchor_per_grid = len(self.anchors[0]) // 2
        self.grids = [torch.zeros(1)] * self.layer_num
        a = torch.tensor(self.anchors, requires_grad=False).float().view(self.layer_num, -1, 2)
        normalize_anchors = a / torch.tensor(strides, requires_grad=False).float().view(3, 1, -1)
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
        """
        :param xs:
        :param targets:
        :param shape:h,w
        :return:
        """
        z = list()
        assert len(xs) == self.layer_num
        for i in range(self.layer_num):
            xs[i] = self.heads[i](xs[i])
            bs, _, ny, nx = xs[i].shape
            xs[i] = xs[i].view(bs, self.anchor_per_grid, self.output_num, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference
                if self.grids[i].shape[2:4] != xs[i].shape[2:4]:
                    self.grids[i] = self._make_grid(nx, ny).to(xs[i].device)
                # grid: bs,anchor_per_grid,ny,nx,2
                # xs[i]:bs,anchor_per_grid,ny,nx,output
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
    "num_cls": 1,
    "strides": None,
    "anchors": None,
    "scale_name": "s",
    "in_channel": 3,
    "conf_thresh": 0.001,
    "iou_thresh": 0.6,
    "max_det": 300
}


class YOLOv5(nn.Module):
    def __init__(self,
                 **cfg):
        super(YOLOv5, self).__init__()
        self.cfg = {**default_cfg, **cfg}
        depth_multiples, width_multiples = model_scale(self.cfg['scale_name'])
        self.backbones = YOLOv5Backbone(self.cfg['in_channel'], depth_multiples, width_multiples)
        c3, c4, c5 = self.backbones.out_channels
        self.neck = YOLOv5Neck(c3, c4, c5, blocks=depth_grow(3, depth_multiples))
        self.head = YOLOv5Head(c3, c4, c5, num_cls=self.cfg['num_cls'],
                               strides=self.cfg['strides'],
                               anchors=self.cfg['anchors'])
        self.loss = PedYOLOv5Loss()

    def forward(self, inp, targets=None):
        xs = self.head(self.neck(self.backbones(inp)))
        if self.training:
            assert targets is not None and len(targets) > 0
            iou_loss, obj_loss = self.loss(predicts=xs, targets=targets, wh_anchors=self.head.normalize_anchors)
            return {
                "iou_loss": iou_loss,
                "obj_loss": obj_loss
            }
        else:
            shape = inp.shape[-2:]
            predicts = non_max_suppression(xs, self.cfg['conf_thresh'],
                                           self.cfg['iou_thresh'],
                                           self.cfg['max_det'])

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


@torch.no_grad()
def main():
    input_tensor = torch.rand(size=(2, 3, 416, 416))
    net = YOLOv5(in_channel=3, scale_name='s', num_cls=1).eval()
    ret = net(input_tensor, torch.rand((5, 5)))
    print(ret)


if __name__ == '__main__':
    main()
    # print(norm_anchors)
    # print(norm_anchors.requires_grad)
    # with torch.no_grad():
    #     net.eval()
    #     out = net(input_tensor)
    #     print(out.shape)
    #     print(out.requires_grad)
    # with open("yolov5s_cvt.txt", 'w') as wf:
    #     weights = net.state_dict()
    #     for i, name in enumerate(weights):
    #         item = weights[name]
    #         print(i, name, item.shape)
    #         print(name, file=wf)

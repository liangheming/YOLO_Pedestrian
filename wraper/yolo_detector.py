import torch
import numpy as np
import cv2 as cv
from nets.yolov5 import YOLOv5
from nets.yolov4 import YOLOv4


class ScaleToMax(object):
    def __init__(self,
                 minimum_rectangle=False,
                 scale_up=True,
                 pad_to_square=True,
                 division=64):
        super(ScaleToMax, self).__init__()
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up
        self.pad_to_square = pad_to_square
        self.division = division

    def make_border(self, img: np.ndarray, max_thresh, border_val):
        h, w = img.shape[:2]
        r = min(max_thresh / h, max_thresh / w)
        if not self.scale_up:
            r = min(r, 1.0)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if r != 1.0:
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        if not self.pad_to_square:
            return img, r, (0, 0)
        dw, dh = int(max_thresh - new_w), int(max_thresh - new_h)
        if self.minimum_rectangle:
            dw, dh = np.mod(dw, self.division), np.mod(dh, self.division)
        dw /= 2
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(border_val,
                                                                                          border_val,
                                                                                          border_val))
        return img, r, (left, top)


class YOLOWrapper(object):
    def __init__(self,
                 weight_path,
                 version="v5",
                 device="cuda:0",
                 input_size=640,
                 padding_val=114,
                 **kwargs):
        self.input_size = input_size
        self.padding_val = padding_val
        self.device = torch.device(device)
        self.version = version
        if version == "v5":
            self.model = YOLOv5(**kwargs)
        elif version == "v4":
            self.model = YOLOv4(**kwargs)
        else:
            raise NotImplementedError("version: {:s} is not supported yet!".format(version))
        self.weight_path = weight_path
        self.transform = ScaleToMax(minimum_rectangle=True)

    def model_init(self):
        weights = torch.load(self.weight_path, map_location="cpu")
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device).fuse().eval().half()
        return self

    def convert_weight(self, weight_path):
        weights = torch.load(self.weight_path, map_location="cpu")['ema']
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device).eval().half()
        torch.save(self.model.state_dict(), weight_path)

    @torch.no_grad()
    def predict_one(self, img):
        input_img, ratio, (left, top) = self.transform.make_border(img, self.input_size, self.padding_val)
        img_out = input_img[:, :, [2, 1, 0]].transpose(2, 0, 1)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).div(255.0).to(self.device).half()
        predicts = self.model(img_out)['predicts'][0]
        if predicts is not None:
            predicts[:, [0, 2]] = (predicts[:, [0, 2]] - left) / ratio
            predicts[:, [1, 3]] = (predicts[:, [1, 3]] - top) / ratio
        return predicts

    @torch.no_grad()
    def predict(self, imgs):
        batch_input = list()
        batch_ratios = list()
        batch_borders = list()
        for img in imgs:
            input_img, ratio, (left, top) = self.transform.make_border(img, self.input_size, self.padding_val)
            batch_input.append(input_img[:, :, ::-1])
            batch_ratios.append(ratio)
            batch_borders.append((left, top))
        batch_input = torch.from_numpy(
            np.stack(batch_input, axis=0)).div(255.0).permute(0, 3, 1, 2).contiguous().to(self.device).half()
        predicts = self.model(batch_input)['predicts']
        for predict, ratio, (left, top) in zip(predicts, batch_ratios, batch_borders):
            if predict is not None:
                predict[:, [0, 2]] = (predict[:, [0, 2]] - left) / ratio
                predict[:, [1, 3]] = (predict[:, [1, 3]] - top) / ratio
        return predicts

import torch
import numpy as np
import cv2 as cv
from wraper.simple_pose.utils.decoder import GaussTaylorKeyPointDecoder, BasicKeyPointDecoder
from wraper.simple_pose.utils.kps_nms import oks_nms
from wraper.simple_pose.nets import pose_resnet_duc


def box_to_center_scale(x, y, w, h, aspect_ratio=192. / 256., scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1.0
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32)):
    """
    获得转换仿射变换的矩阵和逆矩阵
    :param center: 中心
    :param scale: 宽高
    :param rot: 旋转角度
    :param output_size: 输出图片尺寸
    :param shift:
    :return:
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans_inv = cv.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv.getAffineTransform(np.float32(src), np.float32(dst))
    return trans, trans_inv


class PersonDetect(object):
    def __init__(self, img, boxes, scores):
        self.img = img
        self.boxes = boxes
        self.scores = scores


class BasicTransform(object):
    def __init__(self,
                 input_shape=(192, 256),
                 output_shape=(48, 64),
                 ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.w_h_ratio = self.input_shape[0] / self.input_shape[1]

    def __call__(self, joint_info: PersonDetect):
        img = joint_info.img
        data_list = list()
        for bbox, score in zip(joint_info.boxes, joint_info.scores):
            x1, y1, x2, y2 = bbox
            center, scale = box_to_center_scale(
                x1, y1, x2 - x1, y2 - y1, self.w_h_ratio)
            img_trans, _ = get_affine_transform(center, scale, 0, self.input_shape)
            _, joint_trans_inv = get_affine_transform(center, scale, 0, self.output_shape)
            input_img = cv.warpAffine(img, img_trans, self.input_shape, flags=cv.INTER_LINEAR)
            data_list.append({
                "input_img": input_img,
                "trans_inv": joint_trans_inv,
                "area": scale[0] * scale[1],
                "score": score,
                "img_trans": img_trans
            })
        return data_list


default_cfg = {
    "input_shape": (192, 256),
    "output_shape": (48, 64),
    "in_vis_thre": 0.2,
    "oks_thresh": 0.9
}

rgb_mean = [0.485, 0.456, 0.406]


class PoseWrapper(object):
    def __init__(self, weight_path, device="cuda:0", name="resnet50", **kwargs):
        self.device = torch.device(device)
        self.model = getattr(pose_resnet_duc, name)(**kwargs)
        weights = torch.load(weight_path, map_location="cpu")['ema']
        self.model.load_state_dict(weights)
        self.model.eval().to(self.device)
        self.decoder = GaussTaylorKeyPointDecoder()

    @torch.no_grad()
    def predict(self, input_tensors, trans_inv):
        input_tensors = input_tensors.to(self.device)
        trans_inv = trans_inv.to(self.device)
        heat_map = self.model(input_tensors)
        predicts, kps_scores, heat_map = self.decoder(heat_map, trans_inv, return_heat_map=True)
        return heat_map, predicts, kps_scores


class MultiPoseEstimator(object):
    def __init__(self, detector, sppe, device="cuda:0", **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        self.detector = detector
        self.device = torch.device(device)
        self.sppe = sppe.to(self.device)
        self.transform = BasicTransform(input_shape=self.cfg['input_shape'],
                                        output_shape=self.cfg['output_shape'])
        self.decoder = GaussTaylorKeyPointDecoder()

    @torch.no_grad()
    def predict_one(self, img):
        predicts = self.detector.predict_one(img)
        if predicts is None:
            return None
        boxes = predicts[:, :4].cpu().numpy()
        box_scores = predicts[:, 4].cpu().numpy()
        detects = PersonDetect(img=img, boxes=boxes, scores=box_scores)
        data_list = self.transform(detects)
        img_list = list()
        trans_inv = list()
        areas = list()
        for item in data_list:
            norm_img = item['input_img'][..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32)
            img_list.append(torch.from_numpy(norm_img).permute(2, 0, 1).contiguous())
            trans_inv.append(torch.from_numpy(item['trans_inv']))
            areas.append(item['area'])
        box_scores = torch.from_numpy(box_scores).to(self.device)
        areas = torch.tensor(areas, device=self.device)
        input_tensors = torch.stack(img_list).to(self.device)
        trans_inv = torch.stack(trans_inv).float().to(self.device)
        heat_map = self.sppe(input_tensors)
        predicts, kps_scores = self.decoder(heat_map, trans_inv)
        kpt_item = torch.cat([predicts, kps_scores], dim=-1)
        kps_scores = ((kps_scores > self.cfg['in_vis_thre']).float() * kps_scores).mean(dim=1).squeeze(-1)
        scores = box_scores * kps_scores
        keep = oks_nms(kpt_item, scores, areas, self.cfg['oks_thresh'])
        return kpt_item[keep], scores[keep]


class DetailMultiPoseEstimator(object):
    def __init__(self, detector, sppe, device="cuda:0", **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        self.detector = detector
        self.device = torch.device(device)
        self.sppe = sppe.to(self.device)
        self.transform = BasicTransform(input_shape=self.cfg['input_shape'],
                                        output_shape=self.cfg['output_shape'])
        self.decoder = GaussTaylorKeyPointDecoder()

    @torch.no_grad()
    def predict_one(self, img):
        predicts = self.detector.predict_one(img).cpu().numpy()
        if predicts is None:
            return None
        boxes = predicts[:, :4]
        box_scores = predicts[:, 4]
        detects = PersonDetect(img=img, boxes=boxes, scores=box_scores)
        data_list = self.transform(detects)
        img_list = list()
        trans_inv = list()
        img_trans = list()
        areas = list()
        for item in data_list:
            norm_img = item['input_img'][..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32)
            img_list.append(norm_img)
            trans_inv.append(item['trans_inv'])
            areas.append(item['area'])
            img_trans.append(item['img_trans'])
        box_scores = torch.from_numpy(box_scores).to(self.device)
        areas = torch.tensor(areas, device=self.device)
        input_tensors = torch.from_numpy(np.stack(img_list)).permute(0, 3, 1, 2).contiguous().to(self.device)
        trans_inv = torch.from_numpy(np.stack(trans_inv)).float().to(self.device)
        img_trans = torch.from_numpy(np.stack(img_trans)).float()
        heat_map = self.sppe(input_tensors)
        predicts, kps_scores, heat_map = self.decoder(heat_map, trans_inv, return_heat_map=True)
        heat_map = torch.nn.functional.interpolate(heat_map, scale_factor=(4, 4), mode="bilinear", align_corners=True)
        kpt_item = torch.cat([predicts, kps_scores], dim=-1)
        kps_scores = ((kps_scores > self.cfg['in_vis_thre']).float() * kps_scores).mean(dim=1).squeeze(-1)
        scores = box_scores * kps_scores
        keep = oks_nms(kpt_item, scores, areas, self.cfg['oks_thresh'])
        return kpt_item[keep], scores[keep], torch.from_numpy(boxes)[keep], heat_map[keep], img_trans[keep]


class MultiPose(object):
    def __init__(self, detector, estimator, **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        self.detector = detector
        self.estimator = estimator
        self.transform = BasicTransform(input_shape=self.cfg['input_shape'],
                                        output_shape=self.cfg['output_shape'])

    def predict_one(self, img):
        predicts = self.detector.predict_one(img)
        if predicts is None:
            return None
        predicts = predicts.cpu().numpy()
        boxes = predicts[:, :4]
        box_scores = predicts[:, 4]
        detects = PersonDetect(img=img, boxes=boxes, scores=box_scores)
        data_list = self.transform(detects)
        input_tensors, trans_inv, img_trans, areas = self.to_tensor(data_list)
        areas = areas.to(self.estimator.device)
        heat_map, predicts, kps_scores = self.estimator.predict(input_tensors, trans_inv)
        box_scores = torch.from_numpy(box_scores).to(self.estimator.device)
        kpt_item = torch.cat([predicts, kps_scores], dim=-1)
        kps_scores = ((kps_scores > self.cfg['in_vis_thre']).float() * kps_scores).mean(dim=1).squeeze(-1)
        scores = box_scores * kps_scores
        keep = oks_nms(kpt_item, scores, areas, self.cfg['oks_thresh'])
        return kpt_item[keep], scores[keep]

    @torch.no_grad()
    def predict(self, imgs):
        predicts = self.detector.predict(imgs)
        batch_count = list()
        data_list = list()
        batch_box_scores = list()
        batch_boxes = list()
        for img, pred in zip(imgs, predicts):
            if pred is None:
                batch_count.append(0)
                continue
            batch_count.append(len(pred))
            pred_cpu = pred.cpu().numpy()
            boxes = pred_cpu[:, :4]
            batch_boxes.append(boxes)
            box_scores = pred_cpu[:, 4]
            batch_box_scores.append(box_scores)
            detects = PersonDetect(img=img, boxes=boxes, scores=box_scores)
            data_list.extend(self.transform(detects))
        input_tensors, trans_inv, img_trans, areas = self.to_tensor(data_list)
        # areas = areas.to(self.estimator.device)
        batch_box_scores = torch.from_numpy(np.concatenate(batch_box_scores))
        batch_boxes = torch.from_numpy(np.concatenate(batch_boxes, axis=0))
        heat_map, predicts, kps_scores = self.estimator.predict(input_tensors, trans_inv)
        kpt_item = torch.cat([predicts, kps_scores], dim=-1)
        heat_map = torch.nn.functional.interpolate(heat_map, scale_factor=(4, 4), mode="bilinear", align_corners=True)
        ret_data = list()
        for hm, kpi, area, box_score, box, img_tran in zip(heat_map.split(batch_count),
                                                           kpt_item.cpu().split(batch_count),
                                                           areas.split(batch_count),
                                                           batch_box_scores.split(batch_count),
                                                           batch_boxes.split(batch_count),
                                                           img_trans.split(batch_count)):
            if len(hm) == 0:
                data_list.append({"kps": []})
                continue
            kps_score = kpi[..., [-1]]
            kps_score = ((kps_score > self.cfg['in_vis_thre']).float() * kps_score).mean(dim=1).squeeze(-1)
            scores = box_score * kps_score
            keep = oks_nms(kpi, scores, area, self.cfg['oks_thresh'])
            ret_data.append({
                "heat_maps": hm[keep],
                "img_trans": img_tran[keep],
                "kps": kpi[keep],
                "scores": scores[keep],
                "boxes": box[keep]
            })
        return ret_data

    @staticmethod
    def to_tensor(data_list):
        img_list = list()
        trans_inv = list()
        img_trans = list()
        areas = list()
        for item in data_list:
            norm_img = item['input_img'][..., ::-1].astype(np.float32) / 255.0 - np.array(rgb_mean, dtype=np.float32)
            img_list.append(norm_img)
            trans_inv.append(item['trans_inv'])
            areas.append(item['area'])
            img_trans.append(item['img_trans'])
        areas = torch.tensor(areas)
        input_tensors = torch.from_numpy(np.stack(img_list)).permute(0, 3, 1, 2).contiguous().float()
        trans_inv = torch.from_numpy(np.stack(trans_inv)).float()
        img_trans = torch.from_numpy(np.stack(img_trans)).float()
        return input_tensors, trans_inv, img_trans, areas

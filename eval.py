import os
import pickle
import torch
import time
import json
import cv2 as cv
import numpy as np

from tqdm import tqdm
from utils.model_utils import AverageLogger
from wraper.yolo_detector import YOLOWrapper
from wraper.estimator import PoseWrapper, MultiPose
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_eavl(gt_path, pred_path):
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(gt_path)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    # print(map, map50)


@torch.no_grad()
def eval_detection_on_coco(img_root="/home/huffman/data/coco/val2017",
                           weight_path="/home/huffman/PycharmProjects/simple_yolov4/weights/v5_l_half.pth",
                           json_path="/home/huffman/data/coco/annotations/instances_val2017.json"
                           ):
    coco = COCO(json_path)
    version, model_type = weight_path.split("/")[-1].split('.')[0].split('_')[:2]
    time_logger = AverageLogger()
    detector = YOLOWrapper(
        weight_path=weight_path,
        version=version,
        device="cuda:0",
        input_size=640,
        conf_thresh=0.001,
        iou_thresh=0.6,
        max_det=300,
        scale_name=model_type
    ).model_init()
    coco_predict_list = list()
    bar = tqdm(coco.imgs.keys())
    for img_id in bar:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(img_root, file_name)
        img = cv.imread(img_path)
        tic = time.time()
        predicts = detector.predict([img])[0]
        duration = time.time() - tic
        fps = 1 / duration
        time_logger.update(fps)
        bar.set_description("fps: {:4.2f}".format(time_logger.avg()))
        if predicts is None:
            continue
        box = predicts.detach().cpu().numpy()
        box_wh = box[:, [2, 3]] - box[:, [0, 1]]
        box_xc = box[:, [0, 1]] + box_wh * 0.5
        pred_box = np.concatenate([box_xc, box_wh], axis=-1)
        pred_box[:, :2] -= pred_box[:, 2:] / 2
        for p, b in zip(box.tolist(), pred_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': 1,
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)
    coco_eavl("/home/huffman/data/coco/annotations/instances_val2017.json", "predicts.json")


def eval_kps(pd_ann_path="test_gt_kpt.json",
             gt_ann_path="/home/huffman/data/coco/annotations/person_keypoints_val2017.json"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_ann_path)
    coco_pd = coco_gt.loadRes(pd_ann_path)
    cocoEval = COCOeval(coco_gt, coco_pd, "keypoints")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]


@torch.no_grad()
def pose_gen(img_root="/home/huffman/data/coco/val2017",
             json_path="/home/huffman/data/coco/annotations/instances_val2017.json"):
    device = "cuda:0"
    coco = COCO(json_path)
    detect_weight_path = "/home/huffman/PycharmProjects/simple_yolov4/weights/v5_l_half.pth"
    estimator_weight_path = "/home/huffman/PycharmProjects/simple_pose/weights/duc.pth"
    version, model_type = detect_weight_path.split("/")[-1].split('.')[0].split('_')[:2]
    detector = YOLOWrapper(
        weight_path=detect_weight_path,
        device=device,
        version=version,
        input_size=640,
        conf_thresh=0.1,
        iou_thresh=0.6,
        max_det=100,
        scale_name=model_type
    ).model_init()
    reduction = estimator_weight_path.find("reduction") >= 0
    estimator = PoseWrapper(estimator_weight_path, device, pretrained=False, num_classes=17, reduction=reduction)
    multi_estimator = MultiPose(detector, estimator)
    filter_list = list()
    time_logger = AverageLogger()
    pbar = tqdm(enumerate(coco.imgs.keys()))
    for idx, img_id in pbar:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(img_root, file_name)
        img = cv.imread(img_path)
        tic = time.time()
        ret = multi_estimator.predict_one(img)
        time_logger.update(1 / (time.time() - tic))
        pbar.set_description("fps: {:4.2f}".format(time_logger.avg()))
        if ret is None:
            continue
        kps_item, kpt_score = ret
        # show_frame = render_kps(img, kps_item, kpt_score)
        # cv.imwrite("{:s}.jpg".format(str(img_id)), show_frame)
        for kp, score in zip(kps_item, kpt_score):
            filter_list.append(
                {
                    "image_id": img_id,
                    "score": float(score.cpu()),
                    "category_id": 1,
                    "keypoints": kp.reshape(-1).cpu().tolist()
                }
            )
    with open("filter_kps_predicts.json", 'w') as wf:
        json.dump(filter_list, wf)
    eval_kps(pd_ann_path="filter_kps_predicts.json")


def convert_weights(weight_dir):
    for filename in os.listdir(weight_dir):
        version, style = filename.split("_")[:2]
        weight_path = os.path.join(weight_dir, filename)
        detector = YOLOWrapper(
            weight_path=weight_path,
            version=version,
            device="cuda:0",
            input_size=640,
            conf_thresh=0.001,
            iou_thresh=0.6,
            max_det=300,
            scale_name=style
        )
        new_weight_path = os.path.join(weight_dir, "{:s}_{:s}_half.pth".format(version, style))
        detector.convert_weight(new_weight_path)


if __name__ == '__main__':
    pose_gen()
    # eval_kps(pd_ann_path="filter_kps_predicts.json")

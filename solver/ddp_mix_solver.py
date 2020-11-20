import os
import yaml
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch import nn
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from datasets.custom import CustomerDataSets
from nets.yolov4 import YOLOv4
from nets.yolov5 import YOLOv5
from torch.utils.data.dataloader import DataLoader
from utils.model_utils import rand_seed, ModelEMA, reduce_sum, AverageLogger
from metrics.map import coco_map
from torch.nn.functional import interpolate
from utils.optims_utils import IterWarmUpCosineDecayMultiStepLRAdjust, split_optimizer

rand_seed(1024)


class DDPMixSolver(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.optim_cfg = self.cfg['optim']
        self.val_cfg = self.cfg['val']
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.val_cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpus']
        self.gpu_num = len(str(self.cfg['gpus']).split(","))
        dist.init_process_group(backend='nccl')
        self.tdata = CustomerDataSets(json_path=self.data_cfg['train_json_path'],
                                      debug=self.data_cfg['debug'],
                                      augment=True,
                                      )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.tdata, shuffle=True))
        self.vdata = CustomerDataSets(json_path=self.data_cfg['val_json_path'],
                                      debug=self.data_cfg['debug'],
                                      augment=False,
                                      )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.vdata, shuffle=False))
        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata))
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))
        if self.cfg['model_name'] == "v4":
            net = YOLOv4
        elif self.cfg['model_name'] == "v5":
            net = YOLOv5
        else:
            raise NotImplementedError("{:s} not supported yet".format(self.cfg['model_name']))
        model = net(num_cls=self.model_cfg['num_cls'],
                    anchors=self.model_cfg['anchors'],
                    strides=self.model_cfg['strides'],
                    scale_name=self.model_cfg['scale_name'],
                    )
        self.best_map = 0.
        optimizer = split_optimizer(model, self.optim_cfg)
        local_rank = dist.get_rank()
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        model.to(self.device)
        self.scaler = amp.GradScaler(enabled=True)
        if self.optim_cfg['sync_bn']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = nn.parallel.distributed.DistributedDataParallel(model,
                                                                     device_ids=[local_rank],
                                                                     output_device=local_rank)
        self.optimizer = optimizer
        self.ema = ModelEMA(self.model)
        self.lr_adjuster = IterWarmUpCosineDecayMultiStepLRAdjust(init_lr=self.optim_cfg['lr'],
                                                                  warm_up_epoch=self.optim_cfg['warm_up_epoch'],
                                                                  iter_per_epoch=len(self.tloader),
                                                                  epochs=self.optim_cfg['epochs'],
                                                                  alpha=self.optim_cfg['alpha'],
                                                                  gamma=self.optim_cfg['gamma'],
                                                                  bias_idx=2,
                                                                  milestones=self.optim_cfg['milestones']
                                                                  )
        self.obj_logger = AverageLogger()
        self.iou_logger = AverageLogger()
        self.loss_logger = AverageLogger()
        self.map_logger = AverageLogger()

    def train(self, epoch):
        self.obj_logger.reset()
        self.iou_logger.reset()
        self.loss_logger.reset()
        self.model.train()
        if self.local_rank == 0:
            pbar = tqdm(self.tloader)
        else:
            pbar = self.tloader
        for i, (img_tensor, targets_tensor) in enumerate(pbar):
            with torch.no_grad():
                if len(self.data_cfg['multi_scale']) > 2:
                    target_size = np.random.choice(self.data_cfg['multi_scale'])
                    img_tensor = interpolate(img_tensor, mode='bilinear', size=target_size, align_corners=False)
                _, _, h, w = img_tensor.shape
                img_tensor = img_tensor.to(self.device)
                targets_tensor = targets_tensor.to(self.device)
            self.optimizer.zero_grad()
            with amp.autocast(enabled=True):
                ret = self.model(img_tensor, targets_tensor)
                obj_loss = ret['obj_loss']
                iou_loss = ret['iou_loss']
                loss = obj_loss + iou_loss
            self.scaler.scale(loss).backward()
            self.lr_adjuster(self.optimizer, i, epoch)
            ulr = self.optimizer.param_groups[0]['lr']
            dlr = self.optimizer.param_groups[2]['lr']
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            self.obj_logger.update(obj_loss.item())
            self.iou_logger.update(iou_loss.item())
            self.loss_logger.update(loss.item())
            if self.local_rank == 0:
                pbar.set_description(
                    "epoch:{:2d}|size:{:3d}|loss:{:6.4f}|obj_loss:{:6.4f}|iou_loss:{:6.4f}|ulr:{:8.6f},dlr:{:8.6f}".format(
                        epoch + 1,
                        h,
                        self.loss_logger.avg(),
                        obj_loss.item(),
                        iou_loss.item(),
                        ulr,
                        dlr
                    ))
        self.ema.update_attr(self.model)
        print(
            "epoch:{:3d}|local:{:3d}|loss:{:6.4f}||obj_loss:{:6.4f}|iou_loss:{:6.4f}".format(epoch + 1,
                                                                                             self.local_rank,
                                                                                             self.loss_logger.avg(),
                                                                                             self.obj_logger.avg(),
                                                                                             self.iou_logger.avg(),
                                                                                             )
        )

    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()
        self.ema.ema.eval()
        predict_list = list()
        target_list = list()
        if self.local_rank == 0:
            pbar = tqdm(self.vloader)
        else:
            pbar = self.vloader
        for img_tensor, targets_tensor in pbar:
            _, _, h, w = img_tensor.shape
            targets_tensor[:, 1:] = targets_tensor[:, 1:] * torch.tensor(data=[w, h, w, h])
            targets_tensor[:, [1, 2]] = targets_tensor[:, [1, 2]] - targets_tensor[:, [3, 4]] * 0.5
            targets_tensor[:, [3, 4]] = targets_tensor[:, [1, 2]] + targets_tensor[:, [3, 4]]
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            predicts = self.ema.ema(img_tensor)['predicts']
            for i, pred in enumerate(predicts):
                if pred is not None:
                    pred = torch.cat([pred, torch.zeros_like(pred[..., [0]])], dim=-1)
                predict_list.append(pred)
                targets_sample = targets_tensor[targets_tensor[:, 0] == i][:, 1:]
                targets_sample = torch.cat([torch.zeros_like(targets_sample[..., [0]]), targets_sample], dim=-1)
                target_list.append(targets_sample)
        mp, mr, map50, map = coco_map(predict_list, target_list)
        mp = reduce_sum(torch.tensor(mp, device=self.device)).item() / self.gpu_num
        mr = reduce_sum(torch.tensor(mr, device=self.device)).item() / self.gpu_num
        map50 = reduce_sum(torch.tensor(map50, device=self.device)).item() / self.gpu_num
        map = reduce_sum(torch.tensor(map, device=self.device)).item() / self.gpu_num
        if self.local_rank == 0:
            print("epoch: {:2d}|gpu_num:{:d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
                  .format(epoch + 1,
                          self.gpu_num,
                          mp * 100,
                          mr * 100,
                          map50 * 100,
                          map * 100))
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_{:s}_last.pth"
                                        .format(self.cfg['model_name'], self.model_cfg['scale_name']))
        best_map_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:s}_{:s}_best_map.pth"
                                            .format(self.cfg['model_name'], self.model_cfg['scale_name']))
        ema_static = self.ema.ema.state_dict()
        cpkt = {
            "ema": ema_static,
            "map": map * 100,
            "epoch": epoch,
        }
        if self.local_rank != 0:
            return
        torch.save(cpkt, last_weight_path)
        if map > self.best_map:
            torch.save(cpkt, best_map_weight_path)
            self.best_map = map

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
        dist.destroy_process_group()
        torch.cuda.empty_cache()

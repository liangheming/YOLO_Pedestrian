import json
import torch
from torch.utils.data.dataset import Dataset
from utils.augmentations import *
from utils.boxs_utils import xyxy2xywh


class CustomerDataSets(Dataset):
    def __init__(self, json_path, augment=True, debug=False, transform=None):
        self.json_path = json_path
        self.augment = augment
        self.data_list = list()
        self.transform = transform
        self.__load_in()
        if debug:
            self.data_list = self.data_list[:debug]
        if self.transform is None:
            self.set_transform(None)

    def __load_in(self):
        with open(self.json_path, 'r') as rf:
            json_data = json.load(rf)
        for item in json_data:
            img_path = item['img_path']
            bndbox = item['bndbox']
            box_info = BoxInfo(img_path=img_path,
                               boxes=np.array(bndbox),
                               shape=(item['width'], item['height']),
                               labels=np.zeros(shape=(len(bndbox),)))
            self.data_list.append(box_info)

    def set_transform(self, transform=None):
        if transform is not None:
            self.transform = transform
            return
        if self.augment:
            color_gitter = OneOf(
                transforms=[
                    Identity(),
                    RandHSV(hgain=0.014,
                            vgain=0.68,
                            sgain=0.36),
                ]
            )
            basic_transform = Compose(
                transforms=[
                    color_gitter,
                    RandCrop(min_thresh=0.6, max_thresh=1.0).reset(p=0.2),
                    RandScaleToMax(max_threshes=[640]),
                    RandPerspective(degree=(-10, 10), scale=(0.6, 1.4), translate=0.1)
                ]
            )
            mosaic = MosaicWrapper(candidate_box_info=self.data_list,
                                   sizes=[640],
                                   color_gitter=color_gitter)
            augment_transform = Compose(
                transforms=[
                    OneOf(transforms=[
                        (0.0, basic_transform),
                        (1.0, mosaic)
                    ]),
                    LRFlip().reset(p=0.5)
                ]
            )
            self.transform = augment_transform
        else:
            self.transform = RandScaleToMax(max_threshes=[640])

    def __getitem__(self, item):
        box_info: BoxInfo = self.data_list[item].clone().load_img()
        box_info = self.transform(box_info)
        # ret_img = box_info.draw_box(colors=[(255, 0, 0)], names=["person"])
        # import uuid
        # name = str(uuid.uuid4()).replace('-', "")
        # cv.imwrite("{:s}.jpg".format(name), ret_img)
        return box_info

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        input_images = list()
        box_list = list()
        for idx, item in enumerate(batch):
            img = torch.from_numpy(item.img[:, :, ::-1] / 255.0).float().permute(2, 0, 1).contiguous()
            h, w = img.shape[1:]
            box = xyxy2xywh(item.boxes) / np.array([w, h, w, h])
            boxes = torch.from_numpy(np.concatenate([np.full_like(box[:, [0]], fill_value=idx), box], axis=1)).float()
            input_images.append(img)
            box_list.append(boxes)
        input_images = torch.stack(input_images, dim=0)
        box_list = torch.cat(box_list, dim=0)
        return input_images, box_list


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    custom_data_sets = CustomerDataSets(json_path="../data/person_train.json", debug=64, augment=True)
    loader = DataLoader(dataset=custom_data_sets, batch_size=16, shuffle=True, collate_fn=custom_data_sets.collate_fn)
    for data in loader:
        print(data[0].shape)

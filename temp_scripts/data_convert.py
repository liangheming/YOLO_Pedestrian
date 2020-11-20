import json
import os


def read_in_data():
    with open("../data/person_val.json", "r") as rf:
        json_data = json.load(rf)
    for item in json_data:
        file_name = item['img_path']
        if "/home/huffman/data/val2017/" in file_name:
            file_name = file_name.replace("/home/huffman/data/val2017/",
                                          "/home/thunisoft-root/liangheming/data/coco/coco2017/images/val2017/")
        elif "/home/huffman/data/train2017/" in file_name:
            file_name = file_name.replace("/home/huffman/data/train2017/",
                                          "/home/thunisoft-root/liangheming/data/coco/coco2017/images/train2017/")
        elif "/home/huffman/data/pascal/VOCdevkit/VOC2012/JPEGImages/" in file_name:
            file_name = file_name.replace("/home/huffman/data/pascal/VOCdevkit/VOC2012/JPEGImages/",
                                          "/home/thunisoft-root/liangheming/data/pascal/VOCdevkit/VOC2012/JPEGImages/")
        elif "/home/huffman/data/pascal/VOCdevkit/VOC2007/JPEGImages/" in file_name:
            file_name = file_name.replace("/home/huffman/data/pascal/VOCdevkit/VOC2007/JPEGImages/",
                                          "/home/thunisoft-root/liangheming/data/pascal/VOCdevkit/VOC2007/JPEGImages/")
        assert os.path.exists(file_name), file_name
        item['img_path'] = file_name
        print(item)
    # with open("person_val.json", 'w') as wf:
    #     json.dump(json_data, wf)


def gen_anchors():
    from utils.boxs_utils import kmean_anchors
    from datasets.custom import CustomerDataSets
    datasets = CustomerDataSets(json_path="../data/person_train.json")
    kmean_anchors(datasets)


def train_temp():
    from nets.yolov4 import YOLOv4
    from datasets.custom import CustomerDataSets
    from torch.utils.data.dataloader import DataLoader
    vdata = CustomerDataSets(
        json_path="../data/person_val.json",
        augment=True
    )
    vloader = DataLoader(dataset=vdata, shuffle=False, batch_size=16, collate_fn=vdata.collate_fn)
    model = YOLOv4()
    model.eval()
    for img_tensor, target_tensor in vloader:
        ret = model(img_tensor, target_tensor)
        print(ret)
        break


if __name__ == '__main__':
    train_temp()

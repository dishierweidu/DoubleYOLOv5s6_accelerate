#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 16:49
# @Author  : wangjianrong
# @File    : 3.协程.py
 
from torchvision.models.resnet import resnet50, resnet101
import random
import cv2
import os
import numpy as np
import torch
from time import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

import argparse
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_suffix, is_ascii, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

class infer(object):
    def __init__(self):
        pass
    @torch.no_grad()
    def load(w1,w2,w3,w4):
        # w1='best.pt'
        # w2='17best.pt'
        car_half = False
        armor_half = False
        car_imgsz = 640
        armor_imgsz = 128
        device=''
        device = select_device(device)

        # stride = 64 #resize步长

        car_model = attempt_load(w1, map_location=device)  # load FP32 model
        armor_model = attempt_load(w2, map_location=device)
        m1 = attempt_load(w3, map_location=device)
        m2 = attempt_load(w4, map_location=device)
        car_stride = int(car_model.stride.max())  # model stride 根据mode值调节步长
        armor_stride = int(armor_model.stride.max())

        if car_half:
            car_model.half()  # to FP16
        if armor_half:
            armor_model.half()  # to FP16

        cudnn.benchmark = True  # set True to speed up constant image size inference

        car_model(torch.zeros(1, 3, car_imgsz, car_imgsz).to(device).type_as(next(car_model.parameters())))  # run once
        armor_model(torch.zeros(1, 3, armor_imgsz, armor_imgsz).to(device).type_as(next(armor_model.parameters())))

        return car_model,armor_model,m1,m2,device

    @torch.no_grad()
    def car(model,armor_model,device,img,imgsz):
        # Initialize
        half=False  # use FP16 half-precision inference
        agnostic_nms=False  # class-agnostic NMS
        classes=None  # filter by class: --class 0, or --class 0 2 3
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=16  # maximum detections per image
        augment=False

        # cudnn.benchmark = True  # set True to speed up constant image size inference

        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        armor_out, out, flag_img = [], [], True

        for i in range(1):
            armor_temp = []
            # img = img1 if flag_img else img2
            # flag_img = False
            # img = img[0]
            # cv2.imshow("winname", img)
            # print(img)
            # img = img1
            im0_sz = img.shape
            im0 = img
            # print(img)
            img = infer.letterbox(im0, imgsz, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            
            pred = model(img, augment=augment)[0]
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t4 = time_sync()
            dt[2] += t4 - t3

            for i, det in enumerate(pred):  # per image
                    seen += 1
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        N = 0
                        # Write results
                        for *xyxy, conf, cls in det:
                            # c = int(cls)  # integer class
                            # label = f'{names[c]} {conf:.2f}'
                            # annotator.box_label(xyxy, label, color=colors(c, True))

                            x1=int(xyxy[0].item())
                            y1=int(xyxy[1].item())
                            x2=int(xyxy[2].item())
                            y2=int(xyxy[3].item())
                            # 由此进入第二层装甲板识别网络
                            infer.armor(armor_model, device, N, im0, 128, x1, y1, x2, y2,armor_temp)
                            # armor_out.append(test)
                            # armor_out = armor_out+test
                            # print("test",test)          
                            N = N+1
                        # print("armor_out",armor_temp)

                        # TODO:尽量保证程序不要在GPU和CPU之间切换
                        # 将det转为np格式以便兼容处理
                        det = det.data.cpu().numpy()
                        car_temp = [['car', float(conf), [float(max(x1,0.)), float(max(y1,0.)), float(min(x2,im0_sz[1])), float(min(y2,im0_sz[0]))]] for x1, y1, x2, y2, conf, cls in det]
                        
                        # 验证car区域内是否存在装甲板
                        if armor_temp != [] :
                            car_with_armor = []
                            armor_temp = np.array(armor_temp).astype(np.float32)
                            temp = [[float(max(x1,0.)), float(max(y1,0.)), float(min(x2,im0_sz[1])), float(min(y2,im0_sz[0]))] for x1, y1, x2, y2, conf, cls in det]
                            # print("temp",temp)
                            temp = np.array(temp).astype(np.float32)
                            car_with_armor.append(armor_temp)
                            car_with_armor.append(temp)
                            armor_out.append(car_with_armor)
                        else:
                            # car = np.array([], shape=(0, 4), dtype=float32)
                            armor_temp = [None, None]
                            armor_out.append(armor_temp)

                        # print("armor_temp",armor_temp)
                        # armor_out.append(armor_temp)
                        out.append(car_temp)
                    else:
                        temp, armor_temp = [], [None,None]
                        out.append(temp)
                        armor_out.append(armor_temp)
        dt[3] += time_sync() - t3
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms armor' % t)
        return out,armor_out

    @torch.no_grad()
    def armor(model,device,NUM,img,imgsz,car_x1,car_y1,car_x2,car_y2,armor_out):
        # Initialize
        half=False  # use FP16 half-precision inference
        agnostic_nms=False  # class-agnostic NMS
        classes=None  # filter by class: --class 0, or --class 0 2 3
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=16  # maximum detections per image
        augment=False

        # cudnn.benchmark = True

        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        out = []
        img = img[car_y1:car_y2,car_x1:car_x2]
        im0 = img
        img = infer.letterbox(im0, imgsz, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        pred = model(img, augment=augment)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # per image
                    seen += 1
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        det = det.data.cpu().numpy()
                        cls = -1
                        for x1, y1, x2, y2, conf, cls in det:
                            temp = [float(car_x1+x1), float(car_y1+y1), float(car_x1+x1), float(car_y1+y2), float(car_x1+x2), float(car_y1+y2), float(car_x1+x2), float(car_y1+y1), float(conf), float(cls), NUM, float(car_x1+x1), float(car_y1+y1), float(x2-x1), float(y2-y1)]
                            # out.append(temp)
                            # print("temp",temp)
                            armor_out.append(temp)
                            # armor_data(out,temp)

                        # temp = np.array(temp).astype(np.float32)
                        # return out

    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
 
 
def inference(a,b,device,img1,imgsz):
    # y = model(x)
    # return name
    img_pred, car_location = infer.car(a,b,device,img1,imgsz)
    return img_pred, car_location
 
 
async def ainference(a,b,device,img1,imgsz):
    # y = model(x)
    img_pred, car_location = infer.car(a,b,device,img1,imgsz)
    return img_pred, car_location
 
 
async def main():
    init_seed(0)
    s = time()
    # fake_input = torch.randn(1, 3, 640, 640)
    cap = cv2.VideoCapture("test.jpg")
    staut,img1 = cap.read()
    img2 = img1
    e = time()
    print("gen data:", e - s)
    # img1 = img1.cuda()
    # img2 = img2.cuda()
    # e = time()
    # print("gen data:", e - s)
    warm_cnt = 100
    repeat = 100
    # model1 = resnet50(True).cuda().eval()
    # model2 = resnet101(True).cuda().eval()
    a,b,c,d,device=infer.load('best.pt','17best.pt','bestCopy.pt','17bestCopy.pt')
    # s = time()
    # for i in range(warm_cnt):
    #     y = model1(img1)
    # e = time()
    # print("warm up res50:", e - s)
    # s = time()
    # for i in range(warm_cnt):
    #     y = model2(img2)
    # e = time()
    # print("warm up re101:", e - s)
 
    loop = asyncio.get_running_loop()
    # 方法1 使用协程函数
    s = time()
    for i in range(repeat):
        tasks = [ainference(a,b,device,img1,640), ainference(c,d,device,img2,640)]
        done, pending = await asyncio.wait(tasks)
    e = time()
    print("直接使用协程函数：",e - s)
 
    # 方法2 将非协程函数转成协程future
    s = time()
    for i in range(repeat):
        f1 = loop.run_in_executor(None, inference, a,b,device,img1,640)
        f2 = loop.run_in_executor(None, inference, c,d,device,img2,640)
        done, pending = await asyncio.wait([f1, f2])
    e = time()
    print("将非协程函数转成协程:",e-s)
 
    # 方法3 通过线程池
    pool = ThreadPoolExecutor(max_workers=2)
    s = time()
    for i in range(repeat):
        f1 = loop.run_in_executor(pool, inference, a,b,device,img1,640)
        f2 = loop.run_in_executor(pool, inference, c,d,device,img2,640)
        done,pending = await asyncio.wait([f1,f2])
    e = time()
    print("通过线程池:",e-s)
 
 
 
if __name__ == '__main__':
    # main()
    asyncio.run(main())
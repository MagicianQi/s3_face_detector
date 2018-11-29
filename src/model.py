# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from PIL import Image
from src import S3FD_net, utils


def init_face_detection_model(model_path, device):
    """
    Initialize the face detection model
    :param model_path: model path
    :param device: which device
    :return: model
    """
    print("Getting detection models...")
    model = S3FD_net.s3fd()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def get_face_detection_result(model, img_path, device, nms_threshold, detector_threshold, img_size=150):
    image = Image.open(img_path)
    image = image.resize((img_size, img_size))
    image = np.array(image)
    # 将通道放在最外层
    image = image.transpose(2, 0, 1)
    # 升为4维
    image = image.reshape((1,) + image.shape)
    # 转换为Tensor
    image = torch.from_numpy(image).float()
    image = image.to(device)
    bboxlist = utils.detect(model, image)
    # 非极大值抑制
    keep = utils.nms(bboxlist, nms_threshold)
    # 剩余结果
    bboxlist = bboxlist[keep, :]

    face_square = 0.0
    for box in bboxlist:
        if box[-1] > detector_threshold:
            face_square += (box[2] - box[0]) * (box[3] - box[1])

    face_iou = face_square / float(img_size ** 2)
    return bboxlist, face_iou

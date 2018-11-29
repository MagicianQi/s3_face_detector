# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from src.model import init_face_detection_model, get_face_detection_result

# --------------------路径参数--------------------

GPU_id = "cuda:3"
face_model_path = './models/s3fd_convert.pth'
# 图像resize大小
image_size = 150
# nms阈值
nms_threshold = 0.3
# 人脸置信度阈值
detector_threshold = 0.9
# 加载位置
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")

# --------------------初始化模型--------------------

model_face = init_face_detection_model(face_model_path, device)
# 测试模式
model_face.eval()

# --------------------人脸检测--------------------

img_path = "./test_imgs/test01.jpg"
bbox, face_square = get_face_detection_result(model_face, img_path, device,
                                              nms_threshold=0.3, detector_threshold=0.9, img_size=150)

# --------------------结果--------------------

print(face_square)
print(len(bbox))

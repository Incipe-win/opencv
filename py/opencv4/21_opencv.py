import cv2
import dlib
from matplotlib import pyplot as plt
import numpy as np


# 读取一张图片
# image = cv2.imread("./Tom.jpeg")
image = cv2.imread("./Tom2.jpeg")

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载预测关键点模型 68个
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# 灰度转化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 人脸检测
faces = detector(gray, 1)

# 循环 遍历每一张人脸 给人脸绘制矩阵框和关键点

for face in faces:
    # 绘制矩形框
    cv2.rectangle(image, (face.left(), face.top()),
                  (face.right(), face.bottom()), (0, 255, 0), 5)
    # 预测关键点
    shape = predictor(image, face)
    # 获取关键点坐标
    for pt in shape.parts():
        # 获取横坐标
        pt_position = (pt.x, pt.y)
        # 绘制关键点坐标
        cv2.circle(image, pt_position, 2, (0, 0, 255), -1)

# 显示整个效果图
plt.imshow(image)
plt.show()

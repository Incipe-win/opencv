import cv2
from matplotlib import pyplot as plt
import numpy as np


def show_image(image, title, pos):
    """
    显示图片
    """
    image_RGB = image[:, :, ::-1]
    plt.subplot(2, 2, pos)
    plt.title(title)
    plt.imshow(image_RGB)
    plt.axis("off")


def plot_rectangle(image, faces):
    """
    绘制图片中检测到的人脸
    """
    # 拿到检测到的人脸数据, 返回4个值:  坐标(x, y), 宽高width, height
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image


def main():
    # 读取图片
    # image = cv2.imread("./family.jpg")
    image = cv2.imread("./girls.jpg")
    # 转化灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 通过OpenCV自带的方法cv2.CascadeClassifier()加载级联分类器
    face_alt2 = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
    # 通过分类器对人脸进行检测
    face_alt2_detect = face_alt2.detectMultiScale(gray)
    # 绘制图片中检测到的人脸
    face_alt2_result = plot_rectangle(image.copy(), face_alt2_detect)
    plt.figure(figsize=(9, 6))
    plt.suptitle("Face detection with Haar Cascade", fontsize=14,
                 fontweight="bold")
    show_image(face_alt2_result, "face_alt2", 1)
    plt.show()


if __name__ == "__main__":
    main()

import cv2
from matplotlib import pyplot as plt
import numpy as np
import dlib


def show_image(image, title):
    """
    显示图片
    """
    image_RGB = image[:, :, ::-1]
    plt.title(title)
    plt.imshow(image_RGB)
    plt.axis("off")


def plot_rectangle(image, faces):
    """
    绘制人脸矩阵
    """
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()),
                      (face.right(), face.bottom()), (255, 0, 0), 4)
    return image


def main():
    # 读取图片
    image = cv2.imread("./family.jpg")
    # image = cv2.imread("./girls.jpg")
    # 灰度装换
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    # 1代表將图片放大一倍
    dets_result = detector(gray, 1)
    img_result = plot_rectangle(image.copy(), dets_result)

    plt.figure(figsize=(9, 6))
    plt.suptitle("face detection with dlib", fontsize=14, fontweight="bold")
    show_image(img_result, "face detection")
    plt.show()


if __name__ == "__main__":
    main()

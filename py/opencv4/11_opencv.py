import cv2
import argparse
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import font_manager

my_font = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def show_image(image, title, pos):
    image_RGB = image[:, :, ::-1]
    plt.title(title, fontproperties=my_font)
    plt.subplot(2, 3, pos)
    plt.imshow(image_RGB)


def show_histogram(hist, title, pos, color):
    plt.title(title, fontproperties=my_font)
    plt.subplot(2, 3, pos)
    plt.xlabel("Bins", fontproperties=my_font)
    plt.ylabel("Pixels", fontproperties=my_font)
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


def main():
    plt.figure(figsize=(15, 6))
    plt.suptitle("灰度直方图", fontsize=14, fontweight="bold",
                 fontproperties=my_font)
    img = cv2.imread("./children.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_img = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    # hist_img2 = cv2.calcHist(img_gray, [0], None, [256], [0, 256])

    img_BGR = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    show_image(img_BGR, "BGR image", 1)

    show_histogram(hist_img, "gray image histogram", 4, "m")
    # show_histogram(hist_img2, "gray image histogram", 5, "m")
    plt.show()


if __name__ == "__main__":
    main()

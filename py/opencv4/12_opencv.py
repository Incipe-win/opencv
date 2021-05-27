import cv2
import argparse
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import font_manager

my_font = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def show_image(image, title, pos):
    image_RGB = image[:, :, ::-1]
    plt.subplot(2, 3, pos)
    plt.imshow(image_RGB)
    plt.title(title, fontproperties=my_font)


def show_histogram(hist, title, pos, color):
    plt.subplot(2, 3, pos)
    plt.xlabel("Bins", fontproperties=my_font)
    plt.ylabel("Pixels", fontproperties=my_font)
    plt.xlim([0, 256])
    plt.plot(hist, color=color)
    plt.title(title, fontproperties=my_font)


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

    M = np.ones(img_gray.shape, np.uint8) * 50
    added_img = cv2.add(img_gray, M)
    add_img_hist = cv2.calcHist([added_img], [0], None, [256], [0, 256])
    added_img_BGR = cv2.cvtColor(added_img, cv2.COLOR_GRAY2BGR)
    show_image(added_img_BGR, "added image", 2)
    show_histogram(add_img_hist, "added image hist", 5, 'm')

    subtract_img = cv2.subtract(img_gray, M)
    subtract_img_hist = cv2.calcHist(
        [subtract_img], [0], None, [256], [0, 256])
    subtract_img_BGR = cv2.cvtColor(subtract_img, cv2.COLOR_GRAY2BGR)
    show_image(subtract_img_BGR, "subtract image", 3)
    show_histogram(subtract_img_hist, "subtract image hist", 6, 'm')
    plt.show()


if __name__ == "__main__":
    main()

import cv2
from matplotlib import pyplot as plt
import numpy as np


def show_image(image, title, pos):
    image_RGB = image[:, :, ::-1]
    plt.subplot(2, 2, pos)
    plt.title(title)
    plt.imshow(image_RGB)


def show_histogram(hist, title, pos, color):
    plt.subplot(2, 2, pos)
    plt.title(title)
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


def main():
    plt.figure(figsize=(12, 7))
    plt.suptitle("Gray Image and Histogram with mask", fontsize=14,
                 fontweight="bold")
    img = cv2.imread("./children.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    show_image(img, "image gray", 1)
    show_histogram(img_gray_hist, "image gray histogram", 2, "m")

    mask = np.zeros(img_gray.shape, np.uint8)
    mask[130:500, 600:1400] = 255
    img_mask_hist = cv2.calcHist([img_gray], [0], mask, [256], [0, 256])

    mask_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    mask_img_BGR = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    show_image(mask_img_BGR, "gray image with mask", 3)
    show_histogram(img_mask_hist, "histogram with masked gray image", 4, 'm')
    plt.show()


if __name__ == "__main__":
    main()

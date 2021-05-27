import cv2
from matplotlib import pyplot as plt
import numpy as np


def show_image(image, title, pos):
    plt.subplot(3, 2, pos)
    plt.title(title)
    image_RGB = image[:, :, ::-1]
    plt.imshow(image_RGB)
    plt.axis("off")


def show_histogram(hist, title, pos, color):
    plt.subplot(3, 2, pos)
    plt.title(title)
    plt.xlim([0, 256])
    for h, c in zip(hist, color):
        plt.plot(h, color=c)


def calc_color_hist(image):
    hist = []
    hist.append(cv2.calcHist([image], [0], None, [256], [0, 256]))
    hist.append(cv2.calcHist([image], [1], None, [256], [0, 256]))
    hist.append(cv2.calcHist([image], [2], None, [256], [0, 256]))
    return hist


def main():
    plt.figure(figsize=(12, 8))
    plt.suptitle("Color Histogram", fontsize=8, fontweight="bold")
    img = cv2.imread("./children.jpg")

    img_hist = calc_color_hist(img)
    show_image(img, "RGB Image", 1)
    show_histogram(img_hist, "RGB Image Hist", 2, ('b', 'g', 'r'))

    M = np.ones(img.shape, dtype="uint8") * 50

    added_image = cv2.add(img, M)
    added_image_hist = calc_color_hist(added_image)
    show_image(added_image, "Added Image", 3)
    show_histogram(added_image_hist, "Added Image Hist", 4, ('b', 'g', 'r'))

    subtract_image = cv2.subtract(img, M)
    subtract_image_hist = calc_color_hist(subtract_image)
    show_image(subtract_image, "Subtract Image Hist", 5)
    show_histogram(subtract_image_hist,
                   "Subtract Image Hist", 6, ('b', 'g', 'r'))
    plt.show()


if __name__ == "__main__":
    main()

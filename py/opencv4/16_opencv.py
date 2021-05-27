import cv2
from matplotlib import pyplot as plt
import numpy as np


colors = {'blue': (255, 0, 0),
          'green': (0, 255, 0),
          'red': (0, 0, 255),
          'yellow': (0, 255, 255),
          'magenta': (255, 0, 255),
          'cyan': (255, 255, 0),
          'white': (255, 255, 255),
          'black': (0, 0, 0),
          'gray': (125, 125, 125),
          'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50),
          'light_gray': (220, 220, 220)
          }


def show_image(image, title):
    image_RGB = image[:, :, ::-1]
    plt.title(title)
    plt.imshow(image_RGB)
    plt.show()


canvas = np.zeros((400, 400, 3), np.uint8)
canvas.fill(255)

cv2.putText(canvas, "Hello Wolrd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            colors['red'], cv2.LINE_4)
cv2.putText(canvas, "Welcome to world", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, colors['blue'], cv2.LINE_8)
show_image(canvas, "Canvas")

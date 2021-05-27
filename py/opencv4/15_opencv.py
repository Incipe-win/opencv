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

rows, cols = 3, 3


def show_image(image, title, pos):
    img_RGB = image[:, :, ::-1]
    plt.subplot(rows, cols, pos)
    plt.title(title)
    plt.imshow(img_RGB)


canvas = np.zeros((400, 400, 3), np.uint8)
canvas[:] = colors['white']
show_image(canvas, "Background", 1)

cv2.line(canvas, (0, 0), (400, 400), colors['green'], 5)
cv2.line(canvas, (400, 0), (0, 400), colors['black'], 5)
show_image(canvas, "cv2.line()", 2)

cv2.rectangle(canvas, (10, 20), (70, 120), colors['green'], 3)
cv2.rectangle(canvas, (150, 50), (200, 300), colors['blue'], -1)
show_image(canvas, "cv2.rectangle()", 3)

cv2.circle(canvas, (200, 200), 150, colors['black'], 3)
cv2.circle(canvas, (200, 200), 50, colors['green'], -1)
show_image(canvas, "cv2.circle()", 4)

pts = np.array([[250, 5], [220, 80], [280, 80]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(canvas, [pts], True, colors['red'], 3)

pts2 = np.array([[150, 200], [90, 130], [280, 180]], np.int32)
pts2 = pts2.reshape((-1, 1, 2))
cv2.polylines(canvas, [pts2], False, colors['black'], 5)
show_image(canvas, "cv2.polylines()", 5)
plt.show()

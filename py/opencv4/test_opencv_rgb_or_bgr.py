from PIL import Image
import cv2

sourceFileName = "./Tom.jpeg"
im = Image.open(sourceFileName)
width = im.size[0]
height = im.size[1]
im = im.convert('RGB')
array = []
for x in range(width):
    for y in range(height):
        r, g, b = im.getpixel((x, y))
        rgb = (r, g, b)
        array.append(rgb)
print(array)
image = cv2.imread(sourceFileName)
print(cv2.split(image))

import cv2
import face_recognition
from matplotlib import pyplot as plt


def show_image(image, title):
    plt.title(title)
    image_RGB = image[:, :, ::-1]
    plt.imshow(image_RGB)
    plt.axis("off")


def show_landmarks(image, landmarks):
    for landmarks_dict in landmarks:
        for landmarks_key in landmarks_dict.keys():
            for point in landmarks_dict[landmarks_key]:
                cv2.circle(image, point, 2, (0, 0, 255), -1)
    return image


def main():
    image = cv2.imread("./Tom.jpeg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_marks = face_recognition.face_landmarks(gray, None, "large")
    img_result = show_landmarks(image.copy(), face_marks)
    plt.figure(figsize=(9, 6))
    plt.suptitle("Face Landmarks with face_recognition", fontsize=14,
                 fontweight="bold")
    show_image(img_result, "landmarks")
    plt.show()


if __name__ == "__main__":
    main()

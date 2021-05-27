import cv2
import dlib


def plot_rectangle(image, faces):
    """
    绘制人脸矩阵
    """
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()),
                      (face.right(), face.bottom()), (255, 0, 0), 4)
    return image


def main():
    # 读取摄像头
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Camera error!")

    while True:
        ret, frame = capture.read()
        if ret:
            # 灰度装换
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detector = dlib.get_frontal_face_detector()
            # 1代表將图片放大一倍
            dets_result = detector(gray, 1)
            img_result = plot_rectangle(frame.copy(), dets_result)

            cv2.imshow("face detction with dlib", img_result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

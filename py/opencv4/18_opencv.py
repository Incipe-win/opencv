import cv2


def plot_rectangle(image, faces):
    """
    绘制图片中检测到的人脸
    """
    # 拿到检测到的人脸数据, 返回4个值:  坐标(x, y), 宽高width, height
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image


def main():
    # 读取摄像头
    capture = cv2.VideoCapture(0)
    # 通过OpenCV自带的方法cv2.CascadeClassifier()加载级联分类器
    face_alt2 = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
    if not capture.isOpened():
        print("Camera error!")

    while True:
        ret, frame = capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 通过分类器对人脸进行检测
            face_alt2_detect = face_alt2.detectMultiScale(gray)
            # 绘制图片中检测到的人脸
            face_alt2_result = plot_rectangle(frame.copy(), face_alt2_detect)
            cv2.imshow("face detection", face_alt2_result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

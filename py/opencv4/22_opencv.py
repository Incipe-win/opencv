import cv2
import dlib

capture = cv2.VideoCapture(2)
if not capture.isOpened():
    print("Camera error!")

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载预测关键点模型 68个
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


while True:
    ret, frame = capture.read()
    if ret:
        # 灰度转化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray, 1)

        # 循环 遍历每一张人脸 给人脸绘制矩阵框和关键点
        for face in faces:
            # 绘制矩形框
            cv2.rectangle(frame, (face.left(), face.top()),
                          (face.right(), face.bottom()), (0, 255, 0), 3)
            # 预测关键点
            shape = predictor(gray, face)
            # 获取关键点坐标
            for pt in shape.parts():
                # 获取圆点
                pt_position = (pt.x, pt.y)
                # 绘制关键点坐标
                cv2.circle(frame, pt_position, 2, (0, 0, 255), -1)

        # 显示整个效果图
        cv2.imshow("detection with dlib", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import face_recognition


class Face:
    def __init__(self, image_file, name_list, video_index):
        self.image_file = image_file
        self.encodings = []
        self.names = name_list
        self.video_index = int(video_index)

    def face_save(self):
        image_list = [cv2.imread(image) for image in self.image_file]
        for image in image_list:
            face = face_recognition.face_locations(image)
            face_encoding = face_recognition.face_encodings(image, face)[0]
            self.encodings.append(face_encoding)

    def face_dectection(self):
        capture = cv2.VideoCapture(self.video_index)
        if not capture.isOpened():
            print("Camera Error !")
            raise IOError("Camera Error !")

        while True:
            ret, frame = capture.read()
            # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # 9 人脸检测
            faces_locations = face_recognition.face_locations(frame)
            # 10 人脸特征编码
            faces_encodings = face_recognition.face_encodings(
                frame, faces_locations)
            # 11 与数据库中的所有人脸进行匹配
            for (top, right, bottom, left), face_encoding in zip(faces_locations, faces_encodings):
                # 12 进行匹配
                matches = face_recognition.compare_faces(
                    self.encodings, face_encoding)
                # 13 计算距离
                distances = face_recognition.face_distance(
                    self.encodings, face_encoding)
                min_distance_index = np.argmin(distances)  # 0, 1, 2
                # 14 判断：如果匹配，获取名字
                name = "Unknown"
                if matches[min_distance_index]:
                    name = names[min_distance_index]
                # 15 绘制人脸矩形框
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 3)
                # 16 绘制、显示对应人脸的名字
                cv2.rectangle(frame, (left, bottom - 30),
                              (right, bottom), (0, 0, 255), 3)
                # 17 显示名字
                cv2.putText(frame, name, (left+10, bottom-10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # 18 显示整个效果
            cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("face recognition", 640, 480)
            cv2.imshow("face recognition", frame)
            # 19 判断 Q , 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

    def run(self):
        self.face_save()
        self.face_dectection()


if __name__ == "__main__":
    image_file = ["./liu.jpeg", "./guo.jpg", "./wu.jpg"]
    video_index = 2
    names = ["liu de hua", "guo fu cheng", "wu hua chao"]
    face = Face(image_file, names, video_index)
    face.run()

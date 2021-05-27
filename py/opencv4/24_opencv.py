import cv2
import dlib


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Camera error!")
        return
    detector = dlib.get_frontal_face_detector()
    tractor = dlib.correlation_tracker()

    tracking_state = False

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter("record.avi", fourcc,
                             frame_fps, (frame_width, frame_height), True)

    while True:
        ret, frame = capture.read()
        if ret:
            if tracking_state is False:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = detector(gray, 1)
                if len(dets) > 0:
                    tractor.start_track(frame, dets[0])
                    tracking_state = True

            if tracking_state:
                tractor.update(frame)
                position = tractor.get_position()
                cv2.rectangle(frame, (int(position.left()), int(position.top())), (int(
                    position.right()), int(position.bottom())), (0, 255, 0), 3)
            cv2.imshow("face tracking", frame)
            output.write(frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

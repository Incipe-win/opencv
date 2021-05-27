import cv2
import dlib


def show_info(frame, track_state):
    pos1 = (20, 40)
    pos2 = (20, 80)
    cv2.putText(frame, "'1' : reset", pos1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0))
    if track_state:
        cv2.putText(frame, "tracking now...", pos2, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0))
    else:
        cv2.putText(frame, "no tracking...", pos2, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0))


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
        show_info(frame, tracking_state)
        if ret:
            if not tracking_state:
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
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('1'):
                tracking_state = False
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

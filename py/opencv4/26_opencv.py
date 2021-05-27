import cv2
import dlib


def show_info(frame, tracking_state):
    pos1 = (10, 20)
    pos2 = (10, 40)
    pos3 = (10, 60)

    info1 = "put left button, select an area, start tracking"
    info2 = "'1' : start tracking, '2' : stop tracking, 'q' : exit"
    cv2.putText(frame, info1, pos1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,
                                                                    255))
    cv2.putText(frame, info2, pos2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,
                                                                    255))
    if tracking_state:
        cv2.putText(frame, "tracking now...", pos3, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0))
    else:
        cv2.putText(frame, "stop tracking...", pos3, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0))


points = []


def mouse_event_handler(event, x, y, flags, parms):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Camera error!")

nameWindow = "Object Tracking"
cv2.namedWindow(nameWindow)
cv2.setMouseCallback(nameWindow, mouse_event_handler)

tracker = dlib.correlation_tracker()

tracking_state = False

while True:
    ret, frame = capture.read()
    show_info(frame, tracking_state)
    if ret:
        if len(points) == 2:
            cv2.rectangle(frame, points[0], points[1], (0, 255, 0), 3)
            dlib_rect = dlib.rectangle(points[0][0], points[0][1], points[1][0],
                                       points[1][1])
        if tracking_state:
            tracker.update(frame)
            pos = tracker.get_position()
            cv2.rectangle(frame, (int(pos.left()), int(pos.top())),
                          (int(pos.right()), int(pos.bottom())), (255, 0, 0), 3)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            if len(points) == 2:
                tracker.start_track(frame, dlib_rect)
                tracking_state = True
                points = []
        elif key == ord('2'):
            points = []
            tracking_state = False
        cv2.imshow(nameWindow, frame)

capture.release()
cv2.destroyAllWindows()

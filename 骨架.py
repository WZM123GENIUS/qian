import cv2
import mediapipe as mp
import numpy as np


def lines_intersection(l1, l2):
    x1, y1 = l1[0], l1[1]
    x2, y2 = l1[2], l1[3]

    a1 = -(y2 - y1)
    b1 = x2 - x1
    c1 = (y2 - y1) * x1 - (x2 - x1) * y1

    x3, y3 = l2[0], l2[1]
    x4, y4 = l2[2], l2[3]

    a2 = -(y4 - y3)
    b2 = x4 - x3
    c2 = (y4 - y3) * x3 - (x4 - x3) * y3

    r = False
    if b1 == 0 and b2 != 0:
        r = True
    elif b1 != 0 and b2 == 0:
        r = True
    elif b1 != 0 and b2 != 0 and a1 / b1 != a2 / b2:
        r = True

    if r:
        x0 = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
        y0 = (a1 * c2 - a2 * c1) / (a2 * b1 - a1 * b2)
        a = np.sqrt((x4 - x2) ** 2 + (y4 - y2) ** 2)
        b = np.sqrt((x4 - x0) ** 2 + (y4 - y0) ** 2)
        c = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)
        angle = np.arccos((b * b + c * c - a * a) / (2 * b * c)) * 180 / np.pi
        return angle


def MyLine(image, points):
    w, h, _ = image.shape
    output = cv2.fitLine(points=np.array(points), distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
    [vx, vy, x, y] = output
    vx = float(vx)
    vy = float(vy)
    x = float(x)
    y = float(y)
    pts = []
    if vx == 0:
        X = lambda yy: x
        pts.append((int(X(0)), 0))
        pts.append((int(X(w)), w))
    elif vy == 0:
        Y = lambda xx: y
        pts.append((0, int(Y(0))))
        pts.append((h, int(Y(h))))
    else:
        Y = lambda xx: vy / vx * (xx - x) + y
        X = lambda yy: vx / vy * (yy - y) + x
        if 0 <= int(X(0)) <= h:
            pts.append((int(X(0)), 0))
        if 0 <= int(X(w)) <= h:
            pts.append((int(X(w)), w))
        if 0 <= int(Y(0)) <= w:
            pts.append((0, int(Y(0))))
        if 0 <= int(Y(h)) <= w:
            pts.append((h, int(Y(h))))
    if len(pts) == 2:
        pt1 = pts[0]
        pt2 = pts[1]
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)
    return image


if __name__ == '__main__':
    # mp.solutions.drawing_utils????????????
    mp_drawing = mp.solutions.drawing_utils

    # ?????????1????????????2??????????????????3???????????????
    # ??????????????????
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
    # ?????????????????????
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)

    # mp.solutions.pose??????????????????
    mp_pose = mp.solutions.pose

    # ?????????1??????????????????????????????2??????????????????????????????3?????????????????????????????????video????????????4??????????????????5???????????????
    pose_mode = mp_pose.Pose(static_image_mode=True)

    file = 'positive.jpg'
    image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    image_hight, image_width, _ = image.shape
    # ???BGR?????????RGB
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ??????RGB??????
    results = pose_mode.process(image1)

    point_list = [[int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width),  # ??????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width),  # ?????????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_hight)],
                  [int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width),  # ?????????
                   int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_hight)]
                  ]
    # ??????????????????????????????????????????
    print(point_list)
    for point in point_list:
        cv2.circle(image, point, 3, (0, 255, 0), 3)

    eye_center = ((point_list[0][0] + point_list[1][0]) // 2, (point_list[0][1] + point_list[1][1]) // 2)
    ankle_center = ((point_list[2][0] + point_list[3][0]) // 2, (point_list[2][1] + point_list[3][1]) // 2)
    # ??????????????????
    cv2.line(image, point_list[0], point_list[1], (0, 0, 255), 2)
    cv2.line(image, point_list[2], point_list[3], (0, 0, 255), 2)
    MyLine(image, (eye_center, ankle_center))
    line1_p = (eye_center[0], eye_center[1], ankle_center[0], ankle_center[1])
    line2_p = (image_width // 2, 0, image_width // 2, image_hight)
    # ??????????????????
    angle = lines_intersection(line1_p, line2_p)
    print('????????????:{:.2f}??'.format(min(angle, 180 - angle)))
    # ???????????????
    cv2.line(image, point_list[4], point_list[5], (0, 0, 255), 2)
    # ?????????????????????
    line1_p = (point_list[4][0], point_list[4][1], point_list[5][0], point_list[5][1])
    line2_p = (0, image_hight // 2, image_width, image_hight // 2)
    angle = lines_intersection(line1_p, line2_p)
    print('?????????????????????:{:.2f}??'.format(min(angle, 180 - angle)))
    # O/X????????????
    cv2.line(image, point_list[2], point_list[6], (0, 0, 255), 2)
    cv2.line(image, point_list[3], point_list[7], (0, 0, 255), 2)
    cv2.line(image, point_list[6], point_list[8], (0, 0, 255), 2)
    cv2.line(image, point_list[7], point_list[9], (0, 0, 255), 2)
    # O/X??????????????????
    line1_p = (point_list[2][0], point_list[2][1], point_list[6][0], point_list[6][1])
    line2_p = (point_list[6][0], point_list[6][1], point_list[8][0],  point_list[8][1])
    angle1 = lines_intersection(line1_p, line2_p)
    angle1 = min(angle1, 180 - angle1)
    line1_p = (point_list[3][0], point_list[3][1], point_list[7][0], point_list[7][1])
    line2_p = (point_list[7][0], point_list[7][1], point_list[9][0], point_list[9][1])
    angle2 = lines_intersection(line1_p, line2_p)
    angle2 = min(angle2, 180 - angle2)
    print('O/X?????????:{:.2f}??'.format(angle1 + angle2))
    # ??????
    cv2.line(image, (image_width // 2, 0), (image_width // 2, image_hight), (0, 255, 0), 2)
    # ?????????
    cv2.line(image, (0, image_hight // 2), (image_width, image_hight // 2), (0, 255, 0), 2)

    cv2.imwrite('image-back.jpg', image)
    pose_mode.close()

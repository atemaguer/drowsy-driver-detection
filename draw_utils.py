import cv2
import numpy as np

from mesh_connection import FACE_CONNECTIONS


def draw_face(image, bboxes, landmarks, scores, confidence=False):
    image_ret = image.copy()
    eye = None
    for bbox, landmark, score in zip(bboxes.astype(int), landmarks.astype(int), scores):
        # image_ret = cv2.rectangle(image_ret, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=2)

        landmark = landmark.reshape(-1, 2)

        right_eye = landmark[0]
        left_eye = landmark[1]

        center_x = (left_eye[0] + right_eye[0]) // 2
        center_y = (left_eye[1] + right_eye[1]) // 2

        width = int(abs(right_eye[0] - center_x) * 0.9)

        right_eye_top_left = (right_eye[0]-width, right_eye[1]-width)
        right_eye_bottom_right = (right_eye[0]+width, right_eye[1]+width)
        
        left_eye_top_left = (left_eye[0]-width, left_eye[1]-width)
        left_eye_bottom_right = (left_eye[0]+width, left_eye[1]+width)

        eye = image_ret[right_eye_top_left[1]:right_eye_bottom_right[1], right_eye_top_left[0]:right_eye_bottom_right[0]]

        image_ret = cv2.rectangle(image_ret, right_eye_top_left, right_eye_bottom_right,  [255, 0, 0], 3)
        image_ret = cv2.rectangle(image_ret, left_eye_top_left, left_eye_bottom_right,  [255, 0, 0], 3)

    return image_ret, eye


def draw_mesh(image, landmarks, offsets=(0, 0), contour=False):
    image_ret = image.copy()
    landmarks = landmarks.reshape(-1, 3)
    landmarks[:, 0] += offsets[0]
    landmarks[:, 1] += offsets[1]
    #for landmark in landmarks.astype(int):
     #   cv2.circle(image_ret, (landmark[0], landmark[1]), 1, color=[255, 0, 0])

    cv2.circle(image_ret, (landmarks[145][0].astype(int), landmarks[159][1].astype(int)), 5, color=[255, 0, 0])
    if False:
        for connection in FACE_CONNECTIONS:
            landmark_1 = landmarks[connection[0]].astype(int)
            landmark_2 = landmarks[connection[1]].astype(int)
            cv2.line(image_ret, tuple(landmark_1[:2]), tuple(landmark_2[:2]), color=[0, 255, 0], thickness=1)

    return image_ret


def draw_pose(image, r_vec, t_vec, camera_matrix, dist_coeffs, color=(255, 0, 255), thickness=2):
    image_ret = image.copy()

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      r_vec,
                                      t_vec,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image_ret, [point_2d], True, color, thickness, cv2.LINE_AA)
    cv2.line(image_ret, tuple(point_2d[1]), tuple(
        point_2d[6]), color, thickness, cv2.LINE_AA)
    cv2.line(image_ret, tuple(point_2d[2]), tuple(
        point_2d[7]), color, thickness, cv2.LINE_AA)
    cv2.line(image_ret, tuple(point_2d[3]), tuple(
        point_2d[8]), color, thickness, cv2.LINE_AA)

    return image_ret


def put_fps(image, fps, color=(0, 255, 0)):
    image_ret = image.copy()
    fps_label = "FPS: {:.2f}".format(fps)
    (label_w, label_h), baseline = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    cv2.putText(image_ret, fps_label, (5, label_h+5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(color), thickness=2)
    return image_ret

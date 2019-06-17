import cv2
import h5py
import numpy as np
import math

anchors = [57, 36, 45, 48, 54, 30]


def draw_landmark(canvas, landmarks, color=(0, 0, 255)):
    for i, l in enumerate(landmarks):
        cv2.circle(canvas, (int(l[0]), int(l[1])), 1, color)


def draw_arrows(canvas, landmarks):
    anchor_landmarks = landmarks[anchors].astype(int)
    normal = find_normal_vector(landmarks) * 100
    up = find_up_vector(landmarks) * 100
    parallel = find_rotation_vector(landmarks) * 100
    cv2.arrowedLine(canvas, tuple(anchor_landmarks[0][:2]), tuple(anchor_landmarks[0][:2] + normal[:2].astype(int)),
                    color=(0, 255, 0))
    cv2.arrowedLine(canvas, tuple(anchor_landmarks[0][:2]),
                    tuple(anchor_landmarks[0][:2] + up[:2].astype(int)),
                    color=(255, 0, 0))
    cv2.arrowedLine(canvas, tuple(anchor_landmarks[0][:2]),
                    tuple(anchor_landmarks[0][:2] + parallel[:2].astype(int)),
                    color=(0, 0, 255))


def find_normal_vector(landmarks):
    anchor_landmarks = landmarks[anchors]
    vector_right = anchor_landmarks[2] - anchor_landmarks[4]
    vector_left = anchor_landmarks[1] - anchor_landmarks[3]
    vector_normal = np.cross(vector_left, vector_right)
    vector_normal = vector_normal / np.linalg.norm(vector_normal)
    return vector_normal


def find_rotation_vector(landmarks):
    anchor_landmarks = landmarks[anchors]
    vector_parallel = anchor_landmarks[2] - anchor_landmarks[1]
    vector_parallel = vector_parallel / np.linalg.norm(vector_parallel)
    return vector_parallel


def find_up_vector(landmarks):
    anchor_landmarks = landmarks[anchors]
    vector_up = (anchor_landmarks[3] + anchor_landmarks[4]) / 2 - (anchor_landmarks[1] + anchor_landmarks[2]) / 2
    vector_up = vector_up / np.linalg.norm(vector_up)
    return vector_up


def transform_lmk(landmarks):
    nose = anchors[5]
    origin = np.array([0, 0, 0])
    canvas_centre = np.array([320, 240, 0])

    translation = origin - landmarks[nose]
    landmarks += translation

    plane_rotation_vector = find_rotation_vector(landmarks).tolist()
    plane_normal = find_normal_vector(landmarks).tolist()
    vec_up = find_up_vector(landmarks).tolist()

    ux, uy, uz = plane_rotation_vector
    x, y, z = plane_normal
    cosx = np.linalg.norm([y, z]) / np.linalg.norm([x, y, z])
    sinx = -y / np.linalg.norm([x, y, z])
    rot_matrix_x = np.array([[cosx + ux**2 * (1 - cosx), ux * uy * (1 - cosx) - uz * sinx, ux * uz * (1 - cosx) + uy * sinx],
                             [uy * ux * (1 - cosx) + uz * sinx, cosx + uy**2 * (1 - cosx), uy * uz * (1 - cosx) - ux * sinx],
                             [uz * ux * (1 - cosx) - uy * sinx, uz * uy * (1 - cosx) + ux * sinx, cosx + uz**2 * (1 - cosx)]])

    ux, uy, uz = vec_up
    cosy = z / np.linalg.norm([x, z])
    siny = -x / np.linalg.norm([x, z])
    rot_matrix_y = np.array(
        [[cosy + ux ** 2 * (1 - cosy), ux * uy * (1 - cosy) - uz * siny, ux * uz * (1 - cosy) + uy * siny],
         [uy * ux * (1 - cosy) + uz * siny, cosy + uy ** 2 * (1 - cosy), uy * uz * (1 - cosy) - ux * siny],
         [uz * ux * (1 - cosy) - uy * siny, uz * uy * (1 - cosy) + ux * siny, cosy + uz ** 2 * (1 - cosy)]])

    x, y, z = vec_up
    cosz = -y / np.linalg.norm([x, y])
    sinz = x / np.linalg.norm([x, y])
    rot_matrix_z = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    landmarkx = np.matmul(landmarks, rot_matrix_x)
    landmarky = np.matmul(landmarkx, rot_matrix_y)
    landmarkz = np.matmul(landmarky, rot_matrix_z)

    translation = canvas_centre - landmarks[nose]
    landmarks += translation
    translation = np.array([120, 340, 0]) - landmarkx[nose]
    landmarkx += translation
    translation = np.array([320, 340, 0]) - landmarky[nose]
    landmarky += translation
    translation = np.array([520, 340, 0]) - landmarkz[nose]
    landmarkz += translation

    return landmarks, landmarkx, landmarky, landmarkz


def project_lmk(landmarks):
    nose = anchors[5]
    origin = np.array([0, 0, 0])
    canvas_centre = np.array([320, 340, 0])

    translation = origin - landmarks[nose]
    landmarks += translation

    x_axis = find_rotation_vector(landmarks)
    z_axis = find_normal_vector(landmarks)
    y_axis = find_up_vector(landmarks)

    r0 = origin + z_axis
    px = np.matmul((lmk - r0), x_axis)
    py = np.matmul((lmk - r0), y_axis)
    landmarks = np.stack([px, py]).T

    translation = canvas_centre[:2] - landmarks[nose]
    landmarks += translation

    return landmarks


lmk_file = 'E:/Projects/pain/artifacts/LM_fa3D_2019-06-13-23-22-32.hdf5'

win_name = 'landmarks'
win = cv2.namedWindow(win_name,  cv2.WINDOW_GUI_EXPANDED)

with h5py.File(lmk_file, 'r') as lmk_h5:
    for grp in lmk_h5:
        if '76' in grp:
            for lmk in lmk_h5[grp]['landmarks']:
                canvas = np.zeros((480, 640, 3), dtype=np.float32)
                trans = np.array([320, 140, 0]) - lmk[30]
                lmk += trans
                draw_landmark(canvas, lmk, (255, 255, 255))

                lmk = project_lmk(lmk)
                draw_landmark(canvas, lmk, (255, 255, 255))

                cv2.imshow(win_name, canvas)
                cv2.waitKey(10)



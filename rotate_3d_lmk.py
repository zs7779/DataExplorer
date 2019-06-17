import cv2
import h5py
import numpy as np
import math

anchors = [57, 36, 45, 48, 54]


def draw_landmark(canvas, landmarks, color=(0, 0, 255)):
    for i, l in enumerate(landmarks):
        cv2.circle(canvas, (int(l[0]), int(l[1])), 1, color)


def draw_arrows(canvas, landmarks):
    anchor_landmarks = landmarks[anchors].astype(int)
    normal = find_normal_vector(landmarks) * 100
    up = find_up_vector(landmarks) * 100
    parallel = find_rotation_vector(landmarks) * 100
    # cv2.arrowedLine(canvas, tuple(anchor_landmarks[0][:2]),
    #                 tuple(anchor_landmarks[1][:2]),
    #                 color=(0, 0, 255))
    # cv2.arrowedLine(canvas, tuple(anchor_landmarks[0][:2]),
    #                 tuple(anchor_landmarks[2][:2]),
    #                 color=(0, 255, 0))
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
    vector_up = (anchor_landmarks[1] + anchor_landmarks[2]) / 2 - (anchor_landmarks[3] + anchor_landmarks[4]) / 2
    vector_up = vector_up / np.linalg.norm(vector_up)
    return vector_up


def transform_lmk(landmarks):
    botmid = anchors[0]
    origin = np.array([0, 0, 0])
    canvas_centre = np.array([320, 240, 0])

    translation = origin - landmarks[botmid]
    landmarks += translation

    plane_rotation_vector = find_rotation_vector(landmarks)
    plane_normal = find_normal_vector(landmarks)
    vec_up = find_up_vector(landmarks)

    ux, uy, uz = plane_rotation_vector.tolist()
    x, y, z = plane_normal.tolist()
    cosx = np.linalg.norm([y, z]) / np.linalg.norm([x, y, z])
    sinx = y / np.linalg.norm([x, y, z])
    rot_matrix_x = np.array([[cosx + ux**2 * (1 - cosx), ux * uy * (1 - cosx) - uz * sinx, ux * uz * (1 - cosx) + uy * sinx],
                             [uy * ux * (1 - cosx) + uz * sinx, cosx + uy**2 * (1 - cosx), uy * uz * (1 - cosx) - ux * sinx],
                             [uz * ux * (1 - cosx) - uy * sinx, uz * uy * (1 - cosx) + ux * sinx, cosx + uz**2 * (1 - cosx)]])

    ux, uy, uz = vec_up.tolist()
    cosy = z / np.linalg.norm([x, z])
    siny = -x / np.linalg.norm([x, z])
    rot_matrix_y = np.array(
        [[cosy + ux ** 2 * (1 - cosy), ux * uy * (1 - cosy) - uz * siny, ux * uz * (1 - cosy) + uy * siny],
         [uy * ux * (1 - cosy) + uz * siny, cosy + uy ** 2 * (1 - cosy), uy * uz * (1 - cosy) - ux * siny],
         [uz * ux * (1 - cosy) - uy * siny, uz * uy * (1 - cosy) + ux * siny, cosy + uz ** 2 * (1 - cosy)]])

    x, y, z = vec_up.tolist()
    cosz = -y / np.linalg.norm([x, y])
    sinz = x / np.linalg.norm([x, y])
    rot_matrix_z = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    landmarkx = np.matmul(landmarks, rot_matrix_x.T)
    landmarky = np.matmul(landmarkx, rot_matrix_y)
    landmarkz = np.matmul(landmarky, rot_matrix_z)

    translation = canvas_centre - landmarks[botmid]
    landmarks += translation
    translation = np.array([120, 340, 0]) - landmarkx[botmid]
    landmarkx += translation
    translation = np.array([320, 340, 0]) - landmarky[botmid]
    landmarky += translation
    translation = np.array([520, 340, 0]) - landmarkz[botmid]
    landmarkz += translation

    return landmarks, landmarkx, landmarky, landmarkz


lmk_file = 'E:/Projects/pain/artifacts/LM_fa3D_2019-06-13-23-22-32.hdf5'

win_name = 'landmarks'
win = cv2.namedWindow(win_name,  cv2.WINDOW_GUI_EXPANDED)

with h5py.File(lmk_file, 'r') as lmk_h5:
    # all_lmks = [lmk_h5[grp]['landmarks'][()] for grp in lmk_h5]
    # all_lmks = np.concatenate(all_lmks)
    # mean_lmks = all_lmks.mean(axis=0)

    for grp in lmk_h5:
        if '76' in grp:
            while 1:
                canvas = np.zeros((480, 640, 3), dtype=np.float32)
                lmk = lmk_h5[grp]['landmarks'][700]
                trans = np.array([220, 140, 0]) - lmk[30]
                lmk += trans
                draw_landmark(canvas, lmk, (255, 255, 255))
                rx, ry, rz = [np.random.normal()*0.3, np.random.normal()*0.3, np.random.normal()*0.3]
                cx = math.cos(rx)
                sx = math.sin(rx)
                cy = math.cos(ry)
                sy = math.sin(ry)
                cz = math.cos(rz)
                sz = math.sin(rz)
                mat_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
                mat_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                mat_z = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
                mat = np.matmul(np.matmul(mat_x, mat_y), mat_z)
                # lmk = np.matmul(lmk, rot_matrix_x)
                # lmk = np.matmul(lmk, rot_matrix_y)
                lmk = np.matmul(lmk, mat)

                trans = np.array([420, 140, 0]) - lmk[30]
                lmk += trans
                draw_landmark(canvas, lmk, (255, 255, 255))
                draw_arrows(canvas, lmk)
                _, lmkx, lmky, lmkz = transform_lmk(lmk)

                draw_landmark(canvas, lmkx, (0, 0, 255))
                draw_landmark(canvas, lmky, (0, 255, 0))
                draw_landmark(canvas, lmkz, (255, 0, 0))
                draw_arrows(canvas, lmkx)
                draw_arrows(canvas, lmky)
                draw_arrows(canvas, lmkz)

                cv2.imshow(win_name, canvas)
                cv2.waitKey()



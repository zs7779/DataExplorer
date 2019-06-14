import cv2
import h5py
import numpy as np


def draw_landmark(canvas, landmark, color=(0, 0, 255)):
    for i, l in enumerate(landmark):
        cv2.circle(canvas, (int(l[0]), int(l[1])), 1, color)


def find_normal_vector(landmark):
    anchors = [57, 36, 45]
    anchor_points = landmark[anchors]
    vec_right = anchor_points[2] - anchor_points[0]
    vec_left = anchor_points[1] - anchor_points[0]
    normal_vec = np.cross(vec_left, vec_right)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    return normal_vec


def transform_lmk(landmark, plane_normal):
    mouth = 57
    anchors = [57, 36, 45]
    origin = np.array([0, 0, 0])
    canvas_centre = np.array([320, 240, 0])

    translation = origin - landmark[mouth]
    landmark += translation
    x, y, z = plane_normal.tolist()
    cosx = z / np.linalg.norm([y, z])
    sinx = y / np.linalg.norm([y, z])
    cosy = z / np.linalg.norm([x, z])
    siny = x / np.linalg.norm([x, z])
    rot_matrix_x = np.array([[1, 0, 0], [0, cosx, sinx], [0, -sinx, cosx]])
    rot_matrix_y = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])

    landmarkx = np.matmul(landmark, rot_matrix_x)
    landmarky = np.matmul(landmarkx, rot_matrix_y)

    anchor_points = landmark[anchors]
    vec_up = anchor_points[0] - (anchor_points[1] + anchor_points[2]) / 2
    x, y, z = vec_up.tolist()
    cosz = y / np.linalg.norm([x, y])
    sinz = x / np.linalg.norm([x, y])
    rot_matrix_z = np.array([[cosz, sinz, 0], [-sinz, cosz, 0], [0, 0, 1]])
    landmarkz = np.matmul(landmarky, rot_matrix_z)

    translation = canvas_centre - landmark[mouth]
    landmark += translation
    translation = np.array([120, 340, 0]) - landmarkx[mouth]
    landmarkx += translation
    translation = np.array([320, 340, 0]) - landmarky[mouth]
    landmarky += translation
    translation = np.array([520, 340, 0]) - landmarkz[mouth]
    landmarkz += translation

    return landmark, landmarkx, landmarky, landmarkz


lmk_file = 'C:\\Users\\zhaosh\\pain\\artifacts\\landmarks\\LM_fa3D_2019-06-02-12-12-06.hdf5'
anchors = [57, 36, 45]

win_name = 'landmarks'
win = cv2.namedWindow(win_name,  cv2.WINDOW_GUI_EXPANDED)

with h5py.File(lmk_file, 'r') as lmk_h5:
    # all_lmks = [lmk_h5[grp]['landmarks'][()] for grp in lmk_h5]
    # all_lmks = np.concatenate(all_lmks)
    # mean_lmks = all_lmks.mean(axis=0)

    for grp in lmk_h5:
        if '76' in grp:
            for fi, lmk in zip(lmk_h5[grp]['frame_num'], lmk_h5[grp]['landmarks']):
                translation = np.array([320, 140, 0]) - lmk[30]
                lmk += translation

                canvas = np.zeros((480, 640, 3), dtype=np.float32)
                draw_landmark(canvas, lmk, (0, 0, 255))

                anchor_points = lmk[anchors].astype(int)
                normal = find_normal_vector(lmk)*100
                cv2.arrowedLine(canvas, tuple(anchor_points[0][:2]), tuple(anchor_points[0][:2] + normal[:2].astype(int)),
                                color=(0, 0, 255))

                lmk, lmkx, lmky, lmkz = transform_lmk(lmk, normal)

                draw_landmark(canvas, lmkx, (0, 0, 255))
                draw_landmark(canvas, lmky, (0, 255, 0))
                draw_landmark(canvas, lmkz, (255, 0, 0))
                # anchor_points = lmk[anchors].astype(int)
                # normal = find_normal_vector(lmk)*100
                # cv2.arrowedLine(canvas, tuple(anchor_points[0][:2]), tuple(anchor_points[0][:2] + normal[:2].astype(int)),
                #                 color=(0, 255, 0))

                cv2.imshow(win_name, canvas)
                cv2.waitKey(30)



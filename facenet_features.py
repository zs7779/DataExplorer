import h5py
import numpy as np


def facenet_h5_to_tsv(feature_file):
    with h5py.File(feature_file, 'r') as feature_h5:
        bottleneck = feature_h5['bottleneck'][::100]
        feature_a = feature_h5['feature_a'][::100]
        feature_b = feature_h5['feature_b'][::100]
        feature_c = feature_h5['feature_c'][::100]
        pspi = feature_h5['pspi'][::100].round()
        au4 = feature_h5['Continuous Sampling-Brow'][::100].round()
        au67 = feature_h5['Continuous Sampling-Orbit'][::100].round()
        au910 = feature_h5['Continuous Sampling-Levator'][::100].round()
        au43 = feature_h5['Continuous Sampling-Eyes Closed'][::100].round()
        subjects = feature_h5['subjects'][::100]
        np.savetxt('pspi.tsv', pspi, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('au4.tsv', au4, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('au67.tsv', au67, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('au910.tsv', au910, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('au43.tsv', au43, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('subjects.tsv', subjects, fmt='%d', delimiter='\t', newline='\n')
        np.savetxt('bottleneck.tsv', bottleneck, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('feature_a.tsv', feature_a, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('feature_b.tsv', feature_b, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('feature_c.tsv', feature_c, fmt='%.4f', delimiter='\t', newline='\n')


# def get_labels(other_file, annot_file):
#     with h5py.File(other_file, 'r') as other_h5, \
#          h5py.File(annot_file, 'r') as pain_h5:
#         vid_annot_pair = {}
#         for vid in other_h5:
#             for annot in pain_h5:
#                 if vid[:4] == annot[:4] and 'exam' in annot.lower():
#                     vid_annot_pair[vid] = annot
#         pains = []
#         for grp in other_h5:
#             if grp in vid_annot_pair:
#                 print(grp, vid_annot_pair[grp])
#                 pains.extend(pain_h5[vid_annot_pair[grp]]['pspi'].value)
#         return np.array(pains)


lmk_file = 'C:/Users/zhaosh/pain/artifacts/landmarks/LM_2019-04-24-18-37-28.hdf5'
annot_file = 'C:/Users/zhaosh/pain/artifacts/annotations/annot_2019-04-17-18-04-39.hdf5'
feature_file = 'C:/Users/zhaosh/pain/artifacts/facenet_features/facenet_features_2019-05-28-17-23-28.hdf5'
# labels = get_labels(lmk_file, annot_file)
facenet_h5_to_tsv(feature_file)
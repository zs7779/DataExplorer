import h5py
import numpy as np


def facenet_h5_to_tsv(feature_file):
    with h5py.File(feature_file, 'r') as feature_h5:
        pspi = feature_h5['pspi'][()].round()
        subjects = feature_h5['subjects'][()]
        inds = np.arange(pspi.shape[0])
        idx = []
        for i in list(range(2))+list(range(7, 17)):
            idx.append(np.random.choice(inds[pspi == i], size=min(len(inds[pspi == i]), 1000), replace=False))
        idx = sorted(np.concatenate(idx))

        bottleneck = feature_h5['bottleneck'][()]
        bottleneck_0 = bottleneck.copy()
        bottleneck_a = bottleneck.copy()
        # feature_a = feature_h5['feature_a'][()]
        # feature_b = feature_h5['feature_b'][()]
        # feature_c = feature_h5['feature_c'][()]
        for s in np.unique(subjects):
            bn_0 = bottleneck[(subjects == s) & (pspi == 0)].mean(axis=0)
            bn_a = bottleneck[(subjects == s)].mean(axis=0)
            # fa_0 = feature_a[(subjects == s)].mean(axis=0)
            # fb_0 = feature_b[(subjects == s)].mean(axis=0)
            # fc_0 = feature_c[(subjects == s)].mean(axis=0)
            bottleneck_0[subjects == s] = bottleneck_0[subjects == s] - bn_0
            bottleneck_a[subjects == s] = bottleneck_a[subjects == s] - bn_a
            # feature_a[subjects == s] = feature_a[subjects == s] - fa_0
            # feature_b[subjects == s] = feature_b[subjects == s] - fb_0
            # feature_c[subjects == s] = feature_c[subjects == s] - fc_0
        bottleneck = bottleneck[idx]
        bottleneck_0 = bottleneck_0[idx]
        bottleneck_a = bottleneck_a[idx]
        # feature_a = feature_a[idx]
        # feature_b = feature_b[idx]
        # feature_c = feature_c[idx]

        subjects = subjects[idx]
        pspi = pspi[idx]
        pspi += np.random.randn(pspi.shape[0]) * 0.001
        au4 = feature_h5['Continuous Sampling-Brow'][idx]
        au4 += np.random.randn(au4.shape[0]) * 0.001
        au67 = feature_h5['Continuous Sampling-Orbit'][idx]
        au67 += np.random.randn(au67.shape[0]) * 0.001
        au910 = feature_h5['Continuous Sampling-Levator'][idx]
        au910 += np.random.randn(au910.shape[0]) * 0.001
        au43 = feature_h5['Continuous Sampling-Eyes Closed'][idx]
        au43 += np.random.randn(au43.shape[0]) * 0.001
        labels = np.stack([subjects, au4, au67, au910, au43, pspi]).transpose()
        np.savetxt('labels.tsv', labels, header='subjects\tau4\tau6|7\tau9|10\tau43\tpspi',
                   fmt=['%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
                   delimiter='\t', newline='\n', comments='')
        np.savetxt('bottleneck.tsv', bottleneck, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('bottleneck_0.tsv', bottleneck_0, fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt('bottleneck_a.tsv', bottleneck_a, fmt='%.4f', delimiter='\t', newline='\n')
        # np.savetxt('feature_a.tsv', feature_a, fmt='%.4f', delimiter='\t', newline='\n')
        # np.savetxt('feature_b.tsv', feature_b, fmt='%.4f', delimiter='\t', newline='\n')
        # np.savetxt('feature_c.tsv', feature_c, fmt='%.4f', delimiter='\t', newline='\n')


lmk_file = 'C:/Users/zhaosh/pain/artifacts/landmarks/LM_2019-04-24-18-37-28.hdf5'
annot_file = 'C:/Users/zhaosh/pain/artifacts/annotations/annot_2019-04-17-18-04-39.hdf5'
feature_file = 'C:/Users/zhaosh/pain/artifacts/facenet_features/facenet_features_2019-06-03-15-14-33.hdf5'
facenet_h5_to_tsv(feature_file)
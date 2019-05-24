import h5py
import numpy as np

def facenet_h5_to_tsv(feature_file, labels):

    with h5py.File(feature_file, 'r') as feature_h5:
        bottleneck = feature_h5['bottleneck'].value
        feature_a = feature_h5'feature_a'].value
        feature_b = feature_h5'feature_b'].value
        feature_c = feature_h5'feature_c'].value
        np.savetxt('bottleneck.tsv', bottleneck, delimiter='\t', newline='\n')
        np.savetxt('feature_a.tsv', feature_a, delimiter='\t', newline='\n')
        np.savetxt('feature_b.tsv', feature_b, delimiter='\t', newline='\n')
        np.savetxt('feature_c.tsv', feature_c, delimiter='\t', newline='\n')

def get_labels(other_file, annot_file):
    with h5py.File(other_file, 'r') as other_h5, h5py.File(annot_file, 'r') as pain_h5:
        vid_annot_pair = {}
        for vid in other_h5:
            for annot in pain_h5:
                if vid[:4] == annot[:4] and 'exam' in annot.lower():
                    vid_annot_pair[vid] = annot
    pains = []
    for grp in other_h5:
        pains.extend(pain_h5[vid_annot_pair[grp]]['pspi'].value)
    return np.array(pains)

lmk_file = ''
annot_file = ''
feature_file = ''
labels = get_labels(lmk_file, annot_file)
facenet_h5_to_tsv(feature_file, labales)
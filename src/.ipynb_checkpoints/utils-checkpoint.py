import numpy as np
import ruptures as rpt

UNLABELLED_IDX = -1


def create_labels(bkps_list, annotation_ratio=1.0):
    labels_list = []
    for bkps in bkps_list:
        bkps = [0] + sorted(bkps)
        labels = np.full((bkps[-1], 1), UNLABELLED_IDX)
        for idx, (start, end) in enumerate(rpt.utils.pairwise(bkps)):
            offset = int((end - start) * (1 - annotation_ratio) // 2)
            labels[start + offset : end - offset] = idx
        labels_list.append(labels)
    return labels_list


def compute_f1(precision, recall):
    return 2 * (recall * precision) / (recall + precision)

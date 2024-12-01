import numpy as np
from scipy.optimize import linear_sum_assignment

def batch_iou(bb1, bb2):
    bb1 = np.expand_dims(bb1, 1)
    bb2 = np.expand_dims(bb2, 0)
    xx1 = np.maximum(bb1[..., 0], bb2[..., 0])
    yy1 = np.maximum(bb1[..., 1], bb2[..., 1])
    xx2 = np.minimum(bb1[..., 2], bb2[..., 2])
    yy2 = np.minimum(bb1[..., 3], bb2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])                                      
        + (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1]) - wh)                                              
    return(o)  

def assignment(cost_matrix):
    unmatched_rows = set([i for i in range(cost_matrix.shape[0])])
    unmatched_cols = set([i for i in range(cost_matrix.shape[1])])
    r, c = linear_sum_assignment(cost_matrix)
    return list(zip(r,c)), list(unmatched_rows.difference(set(r))), list(unmatched_cols.difference(set(c)))

def bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2
    y = bbox[1] + h/2
    s = w * h
    r = float(w / h)
    if s<0:
        return np.array([x, y, 0, 0]).reshape((4, 1))
    else:
        return np.array([x, y, s, r]).reshape((4, 1))

def z_to_bbox(z, with_score=False):
    if z[2] * z[3] < 0:
        w = 0
        h = 0
    else:
        w = np.sqrt(z[2] * z[3])
        h = np.sqrt(z[2] / z[3])
    x1 = z[0] - w / 2
    y1 = z[1] - h / 2
    x2 = z[0] + w / 2
    y2 = z[1] + h / 2
    if with_score:
        return np.array([x1, y1, x2, y2, 0]).reshape((1, 5))
    else:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))

def calculate_metrics(metrics):
    mota = 1 - ((len(metrics['fn']) + len(metrics['fp']) + metrics['id_switch']) / metrics['num_gt_det'])
    motp = metrics['det_sims'] / len(metrics['tp'])
    idf1 = metrics['idtp'] / (metrics['idtp'] + 0.5 * metrics['idfp'] + 0.5 * metrics['idfn'])
    accs = 0
    for tp in metrics['tp']:
        tpa = 0
        fpa = 0
        fna = 0
        for k in metrics['tp']:
            if tp[0] == k[0] and tp[1] == k[1]:
                tpa += 1
            if (tp[0] == k[0] and tp[1] != k[1]):
                fna += 1
            fna += metrics['fn'].count(tp[0])
            if (tp[0] != k[0] and tp[1] == k[1]):
                fpa += 1
            fpa += metrics['fp'].count(tp[1])
        accs += (tpa / (tpa + fpa + fna))
    hota = np.sqrt(accs / (len(metrics['tp']) + len(metrics['fn']) + len(metrics['fp'])))
    association_accuracy = accs / len(metrics['tp'])
    detection_accuracy = len(metrics['tp']) / (len(metrics['tp']) + len(metrics['fp']) + len(metrics['fn']) )
    return {'MOTA':mota, 'MOTP':motp, 'IDF1':idf1, 'HOTA':hota, 'AssA':association_accuracy, 'DetA':detection_accuracy}
    return mota, motp, idf1, hota, association_accuracy, detection_accuracy

def filter_matches(match, unmatched1, unmatched2, cost_matrix, threshold):
    matchs_to_remove = []
    for i,j in match:
        if cost_matrix[i,j] < threshold:
            unmatched1.append(i)
            unmatched2.append(j)
            matchs_to_remove.append((i,j))
    for m in matchs_to_remove:
        match.remove(m)
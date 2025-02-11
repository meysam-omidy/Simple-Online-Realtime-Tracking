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
    motp = float(metrics['det_sims'] / len(metrics['tp']))
    idf1 = metrics['idtp'] / (metrics['idtp'] + 0.5 * metrics['idfp'] + 0.5 * metrics['idfn'])
    tp = np.array(metrics['tp'])
    fp = np.array(metrics['fp'])
    fn = np.array(metrics['fn'])
    tp0 = np.expand_dims(tp, 0)
    tp1 = np.expand_dims(tp, 1)
    tpa = np.count_nonzero(np.all((tp0==tp1) == [True, True], axis=2), axis=1)
    fna = np.count_nonzero(np.all((tp0==tp1) == [True, False], axis=2), axis=1)
    fna += np.count_nonzero((np.expand_dims(fn, 1) == np.expand_dims(tp[:, 0], 0)), axis=0)
    fpa = np.count_nonzero(np.all((tp0==tp1) == [False, True], axis=2), axis=1)
    fpa += np.count_nonzero((np.expand_dims(fp, 1) == np.expand_dims(tp[:, 1], 0)), axis=0)
    A_c = tpa / (tpa + fna + fpa)
    hota = float(np.sqrt(np.sum(A_c) / (len(tp) + len(fp) + len(fn))))
    association_accuracy = float(np.sum(A_c) / len(metrics['tp']))
    detection_accuracy = len(metrics['tp']) / (len(metrics['tp']) + len(metrics['fp']) + len(metrics['fn']) )
    mt = 0
    ml = 0
    for id in metrics['object_tracked'].keys():
        id_tracked, id_existed = metrics['object_tracked'][id]
        if id_tracked / id_existed > 0.8:
            mt += 1
        elif id_tracked / id_existed < 0.2:
            ml += 1
    mt /= len(metrics['object_tracked'].keys())
    ml /= len(metrics['object_tracked'].keys())
    return {'MOTA':mota, 'MOTP':motp, 'IDF1':idf1, 'HOTA':hota, 'AssA':association_accuracy, 'DetA':detection_accuracy, 'MT':mt, 'ML':ml}

def filter_matches(match, unmatched1, unmatched2, cost_matrix, threshold):
    matchs_to_remove = []
    for i,j in match:
        if cost_matrix[i,j] < threshold:
            unmatched1.append(i)
            unmatched2.append(j)
            matchs_to_remove.append((i,j))
    for m in matchs_to_remove:
        match.remove(m)

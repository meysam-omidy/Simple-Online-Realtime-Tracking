from utils import batch_iou, assignment, bbox_to_z, z_to_bbox, calculate_metrics, filter_matches
from filterpy.kalman import KalmanFilter
import numpy as np
import argparse
import time

class KalmanTracker:
    def __init__(self, bbox, id) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10
        self.kf.P[4:,4:] *= 1000
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(bbox)
        self.time_since_last_update = 0
        self.predict_history = []
        self.update_history = []
        self.id = id

    def update(self, bbox):
        self.time_since_last_update = 0
        self.update_history.append(bbox)
        self.kf.update(bbox_to_z(bbox))

    def predict(self):
        self.time_since_last_update += 1
        self.kf.predict()
        self.predict_history.append(z_to_bbox(self.kf.x)[0])


class SORT:
    def __init__(self, detector, max_time_wtih_no_update, iou_threshold_detection, iou_threshold_track) -> None:
        self.trackers = []
        self.metrics = {
            'id_switch': 0,
            'tp': [],
            'fp': [],
            'fn': [],
            'num_gt_det': 0,
            'idtp': 0,
            'idfp': 0,
            'idfn': 0,
            'det_sims': 0,
            'object_tracked': {}
        }
        self.max_time_wtih_no_update = max_time_wtih_no_update
        self.iou_threshold_detection = iou_threshold_detection
        self.iou_threshold_track = iou_threshold_track
        self.detector = detector

    def set_metrics(self, unique_ids, unique_ids_counts):
        for id, id_count in zip(unique_ids, unique_ids_counts):
            self.metrics['object_tracked'][int(id)] = [0, int(id_count)]

    def handle_trackers_removal(self):
        trackers_to_remove = []
        for tracker in self.trackers:
            x1,y1,x2,y2 = tracker.predict_history[-1]
            if x1<0 or x2<0 or y1<0 or y2<0 or x1>x2 or y1>y2 or (x2==x1 and y2==y1) or tracker.time_since_last_update > self.max_time_wtih_no_update or np.any(np.isnan([x1,y1,x2,y2])):
                trackers_to_remove.append(tracker)
        for t in trackers_to_remove:
            self.trackers.remove(t)

    def update(self, gt_dets, image=None, dets=None):
        if image == None:
            dets = dets
        else:
            dets = self.detector.forward(image)
        gt_ids, gt_dets = gt_dets[:, 0], gt_dets[:, 1:]
        dets, confidences = dets[:, :-1], dets[:, -1]
        cost_matrix = batch_iou(gt_dets, dets)
        matched_dets, unmatched_gt_dets, unmatched_dets = assignment(-cost_matrix)
        filter_matches(matched_dets, unmatched_gt_dets, unmatched_dets, cost_matrix, self.iou_threshold_detection)
        detections = []
        detection_gt_ids = []
        for i,j in matched_dets:
            detections.append(list(dets[j]))
            detection_gt_ids.append(gt_ids[i])
            self.metrics['det_sims'] += cost_matrix[i,j]
        self.metrics['num_gt_det'] += len(gt_dets)
        self.metrics['fn'].extend(gt_ids[unmatched_gt_dets])
        self.metrics['fp'].extend([cost_matrix[:,id].argmax() for id in unmatched_dets])
        tracks = []
        for tracker in self.trackers:
            tracks.append(list(tracker.predict_history[-1]))
        if len(tracks) == 0:
            matched_tracks, unmatched_tracks, unmatched_detections = [], [], [i for i in range(len(detections))]
        elif len(detections) == 0:
            matched_tracks, unmatched_tracks, unmatched_detections = [], [i for i in range(len(tracks))], []
        else:
            cost_matrix = batch_iou(np.array(tracks), np.array(detections))
            matched_tracks, unmatched_tracks, unmatched_detections = assignment(-cost_matrix)
        filter_matches(matched_tracks, unmatched_tracks, unmatched_detections, cost_matrix, self.iou_threshold_track)
        pred_ids_list = [self.trackers[m[0]].id for m in matched_tracks]
        self.metrics['idtp'] += len(set(gt_ids).intersection(pred_ids_list))
        self.metrics['idfp'] += len(set(pred_ids_list).difference(gt_ids))
        self.metrics['idfn'] += len(set(gt_ids).difference(pred_ids_list))
        detection_pred_ids = [-1 for _ in range(len(detection_gt_ids))]
        for i,j in matched_tracks:
            self.trackers[i].update(detections[j])
            detection_pred_ids[j] = self.trackers[i].id
            if self.trackers[i].id != detection_gt_ids[j]:
                self.metrics['id_switch'] += 1
                self.trackers[i].id = detection_gt_ids[j]
        for i in unmatched_detections:
            self.trackers.append(KalmanTracker(detections[i], detection_gt_ids[i]))
            detection_pred_ids[i] = detection_gt_ids[i]
        for gt_id, pred_id in zip(detection_gt_ids, detection_pred_ids):
            self.metrics['tp'].append((gt_id, pred_id))
            self.metrics['object_tracked'][int(gt_id)][0] += 1
        for tracker in self.trackers:
            tracker.predict()
        self.handle_trackers_removal()

def parse_arguments():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--iou_threshold_detection", 
                        help="Threshold used for filtering matches between ground truth detections and detections.", 
                        type=float, default=0.5)
    parser.add_argument("--iou_threshold_track", 
                        help="Threshold used for filtering matches between tracklets and detections.", 
                        type=float, default=0.5)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    gt_dets_file = np.loadtxt('gt.txt', delimiter=',')
    dets_file = np.loadtxt('det.txt', delimiter=',')
    sort = SORT(None, args.max_age, args.iou_threshold_detection, args.iou_threshold_track)
    unique_ids, unique_ids_counts = np.unique(gt_dets_file[:, 1], return_counts=True)
    sort.set_metrics(unique_ids, unique_ids_counts)
    start_time = time.time()
    for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
        gt_dets, dets = gt_dets_file[gt_dets_file[:,0] == frame_number][:, 1:6], dets_file[dets_file[:,0] == frame_number][:, 2:7]
        gt_dets[:, 4] += gt_dets[:, 2]
        gt_dets[:, 3] += gt_dets[:, 1]
        dets[:, 3] += dets[:, 1]
        dets[:, 2] += dets[:, 0]
        sort.update(gt_dets, dets=dets)
    end_time = time.time()
    metrics = calculate_metrics(sort.metrics)
    print(f'Total time: {round(end_time-start_time, 3)}s')
    print(f'FPS: {round(len(np.unique(gt_dets_file[:,0])) / (end_time-start_time), 3)}')
    print('Metrics:')
    for metric in list(metrics.keys()):
        print(f'    {metric}: {round(metrics[metric], 3)}')

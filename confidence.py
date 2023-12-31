
class ConfidenceIniter:
    def __init__(self, comparator):
        self.comparator = comparator

    def __call__(self, data, llrs, detections, new_tracks, new_detections):
        box_sizes = detections[self.comparator.bbox_id][0][new_detections, 2]
        scores = detections[self.comparator.conf_id][new_detections]

        llrs[new_tracks] += self.comparator.eval(scores, box_sizes)


class ConfidenceComparator:
    def __init__(self, bbox_id, conf_id, detection_lr_hist):
        self.bbox_id = bbox_id
        self.conf_id = conf_id
        self.detection_lr_hist = detection_lr_hist

    def __str__(self):
        return "ConfidenceComparator"

    def __call__(self, data, detections, valid_tracks):
        z = detections[self.bbox_id][0]
        return self.eval(detections[self.conf_id], z[:, 2].contiguous())[None]

    def eval(self, scores, box_sizes):
        return self.detection_lr_hist(scores, box_sizes).log()


class ConfidenceEstimator:
    def __init__(self, bbox_id, conf_id, detection_lr_hist):
        self.comparator = ConfidenceComparator(bbox_id, conf_id, detection_lr_hist)
        self.initer = ConfidenceIniter(self.comparator)

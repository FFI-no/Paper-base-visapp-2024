import json
import numpy as np
import sys, os

import torchvision

from utils import estimate_W
from sht_tracker import SHTTracker
from bbox import BBoxEstimator
from confidence import ConfidenceEstimator
from collections import namedtuple, deque

sys.path.insert(0, os.path.abspath('../build'))
sys.path.insert(0, os.path.abspath('./build'))
import torch

TrackData = namedtuple('TrackData',
                       'all_valid_tracks all_ids ids llrs new_tracks tlwhs detected confidences label')


class Detection(object):
    def __init__(self, label, score, llr, x1, y1, x2, y2, img_size, id=None, sigma_center=None, visibility=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.llr = llr
        self.label = label
        self.id = id
        self.sigma_center = sigma_center
        self.img_size = img_size
        self.visibility = visibility

    def __str__(self):
        box = self.to_ccwh()
        return f"box: senter ({box[0]:.1f} {box[1]:.1f}) wh ({box[2]:.1f} {box[3]:.1f}) score: {self.score:.2f}"

    def to_cpp(self, img_size=None):
        from tracktor_mot import DNNDetection
        scale_h, scale_w = self.get_scale_for_img_size(img_size)
        return DNNDetection(self.label, self.score, self.x1 * scale_w, self.y1 * scale_h, self.x2 * scale_w,
                            self.y2 * scale_h)

    def to_cv_bb(self, img_size=None):
        scale_h, scale_w = self.get_scale_for_img_size(img_size)
        return (int(round(self.x1 * scale_w)), int(round(self.y1 * scale_h))), (
            int(round(self.x2 * scale_w)), int(round(self.y2 * scale_h)))

    def get_scale_for_img_size(self, img_size):
        if img_size is None:
            scale_h, scale_w = 1.0, 1.0
        else:
            scale_h, scale_w = img_size[0] / self.img_size[0], img_size[1] / self.img_size[1]
        return scale_h, scale_w

    def to_tlwh(self, img_size=None):
        scale_h, scale_w = self.get_scale_for_img_size(img_size)
        return self.x1 * scale_w, self.y1 * scale_h, (self.x2 - self.x1) * scale_w, (self.y2 - self.y1) * scale_h

    def to_ccwh(self, img_size=None):
        scale_h, scale_w = self.get_scale_for_img_size(img_size)
        return (self.x1 + self.x2) / 2 * scale_w, (self.y1 + self.y2) / 2 * scale_h, (self.x2 - self.x1) * scale_w, (
                self.y2 - self.y1) * scale_h


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5: 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


class CameraDetectorState:
    def __init__(self, pd, compares, cd):
        self.pd = pd
        self.compares = compares
        self.cd = cd

    def l_adjust(self, l, llr):
        lr = llr.exp()
        p = lr / (lr + 1)

        return torch.log(p + (1 - l)) - torch.log(1 - p)

    def __call__(self, data, valid_tracks):
        if not valid_tracks.any() or self.compares.shape[1] == 0:
            return torch.zeros(self.compares.shape[0], dtype=self.compares.dtype, device=self.compares.device)

        p = self.compares.exp()
        ptilde = p / (self.cd + p.sum(dim=0) + 1e-14)

        np_res = (1 - ptilde).log().sum(dim=1)
        llr = (1 - np_res.exp()).log() - np_res
        llr = self.l_adjust(self.pd, llr)

        assert not llr.isnan().any(), "got nan in llr"

        return llr


class BufferedLookaheadTracker:
    def __init__(self, tracker, params):
        self.im_file_buf = deque()
        self.result_buf = deque()
        self.params = params
        self.tracker = tracker
        self.clip_boxes = params.get("clip_boxes", False)
        self.buffer_size = params.get("lookahead_len", 5)

    def track(self, frame_id, detections, detection_size, update_s, in_size_hw, out_size_hw=None):
        if out_size_hw is None:
            out_size_hw = in_size_hw
        track_data: TrackData = self.tracker.track(frame_id, detections, detection_size, in_size_hw, update_s)
        self.result_buf.append((frame_id, track_data, detections))

        if len(self.result_buf) > self.buffer_size:
            return self.track_from_buffer(out_size_hw)
        return False, *[None] * 5

    def track_from_buffer(self, in_size_hw):
        if len(self.result_buf) == 0:
            return False
        h, w = in_size_hw
        frame_id, track_data, detections = self.result_buf.popleft()
        all_valid_tracks = track_data.all_valid_tracks
        all_ids = track_data.all_ids
        killed = ~all_valid_tracks
        llrs = track_data.llrs
        max_llrs = torch.full(all_valid_tracks.shape, -10, dtype=llrs.dtype, device=llrs.device)
        max_llrs[all_valid_tracks] = llrs

        has_more_detections = track_data.all_valid_tracks.clone()
        has_more_detections[:] = False

        for _, next_tracks, detections_ in self.result_buf:
            all_ids_ = next_tracks.all_ids
            all_valid_tracks_ = next_tracks.all_valid_tracks
            killed |= ~(all_valid_tracks_ & (all_ids_ == all_ids))
            valid_tracks = ~killed

            has_more_detections[valid_tracks] |= next_tracks.detected[valid_tracks]
            max_llrs[valid_tracks] = torch.maximum(max_llrs[valid_tracks],
                                                   next_tracks.llrs[(~killed)[all_valid_tracks_]])

        valid_tracks = (max_llrs > self.params['min_report_llr'])[all_valid_tracks]
        if not self.params['report_killed_and_undetected']:
            valid_tracks &= ~(killed & (~track_data.detected) & (~has_more_detections))[all_valid_tracks]
        if not self.params['report_undetected']:
            valid_tracks &= track_data.detected[all_valid_tracks] | track_data.new_tracks[all_valid_tracks]
        tlwh_scale = torch.tensor((w, h), dtype=llrs.dtype, device=llrs.device).repeat(2)
        track_tlwhs = track_data.tlwhs[valid_tracks] * tlwh_scale
        intersect_area, track_area = intersection_over_track(track_tlwhs,
                                                             torch.tensor([0, 0, w, h], dtype=track_tlwhs.dtype,
                                                                          device=track_tlwhs.device)[None])
        inside_frame = intersect_area[:, 0] > 0
        valid_tracks[valid_tracks.clone()] &= inside_frame

        track_tlwhs = track_data.tlwhs[valid_tracks] * tlwh_scale
        track_ids = (track_data.ids[valid_tracks] + 1).cpu().numpy()
        track_confidences = track_data.confidences[valid_tracks].cpu().numpy()
        track_labels = np.full(track_confidences.shape, track_data.label, dtype=np.int32)
        if self.clip_boxes:
            img_size = np.array([in_size_hw])[:, ::-1]
            subtract_wh = -np.minimum(track_tlwhs[:, :2], 0)
            track_tlwhs[:, :2] = np.maximum(track_tlwhs[:, :2], 0)
            subtract_wh += np.maximum((track_tlwhs[:, :2] + track_tlwhs[:, 2:]) - img_size - 1, 0)
            track_tlwhs[:, 2:] = track_tlwhs[:, 2:] - subtract_wh
            track_tlwhs[:, :2] = np.minimum(track_tlwhs[:, :2], img_size)
        return frame_id, track_confidences, track_data.llrs[
            valid_tracks], track_ids, track_labels, track_tlwhs.cpu().numpy()

    def reset(self, **kwargs):
        self.tracker.reset(**kwargs)


class DynamicSigmaAndClutterDensityTracker:
    def __init__(self, tracker, params):
        self.tracker = tracker
        self.seq_vo = None
        self.params = params
        self.first = True
        self.tracker.reset()

    def track(self, frame_id, detections, detection_size, in_size_hw, update_s) -> TrackData:
        assert self.seq_vo is not None, "You need to set seq_vo before calling track"
        params = self.params
        h, w = in_size_hw
        R = self.params['R']
        tracker = self.tracker
        meas_scale = np.array((detection_size[1] / w, detection_size[0] / h))
        J = torch.tensor(np.diag(np.sqrt(meas_scale).repeat(2)))
        seq_R = J @ R @ J
        box_sizes = np.array([det.to_tlwh(detection_size)[-2] for det in detections])
        if self.first:
            cd_fac = params['cd_fac0']
            self.first = False
        else:
            cd_fac = params['cd_fac']
        clutter_density = self.estimate_clutter_density(torch.tensor(box_sizes), cd_fac, self.params['boxsize_hist'],
                                                        self.params['scale_power'], detection_size, h, w)
        W, sigma_W = estimate_W(frame_id, self.seq_vo, detection_size, in_size_hw)
        return tracker.track(detections, seq_R + np.diag(np.array((sigma_W, sigma_W, 0, 0)) ** 2),
                             clutter_density, detection_size, W=W, update_s=update_s)

    @staticmethod
    def estimate_clutter_density(box_sizes, cd_fac0, cdfac_hist, scale_power, detection_size, h, w):
        clutter_density = np.prod(np.array((h, w)) / detection_size) ** scale_power * cd_fac0 * cdfac_hist(
            box_sizes) / np.prod(detection_size)
        return clutter_density

    def reset(self, seq_vo=None, **kwargs):
        if seq_vo is not None:
            self.seq_vo = seq_vo
        self.tracker.reset(**kwargs)


def intersection_over_track(track_tlwh, det_tlwh):
    track_tl = track_tlwh[:, :2]
    track_br = track_tlwh[:, :2] + track_tlwh[:, 2:4]
    det_tl = det_tlwh[:, :2]
    det_br = det_tlwh[:, :2] + det_tlwh[:, 2:4]

    intersect_tl = torch.maximum(track_tl[:, None], det_tl[None])
    intersect_br = torch.minimum(track_br[:, None], det_br[None])

    track_area = track_tlwh[:, 2:4].prod(dim=-1)
    intersect_area = (intersect_br - intersect_tl).clamp(min=0).prod(dim=-1)

    return intersect_area, track_area


class MOTTracker:

    def __init__(self, parameters=None, detection_size=(1080, 1920), num_classes=1,
                 **extra_parameters):
        super()
        self.num_classes = num_classes

        if parameters is None:
            parameters = {}
        parameters.update(extra_parameters)

        self.device = torch.device('cpu')
        self.dtype = torch.float32

        self.parameters = parameters
        self.scorings = {}
        self.detection_size = detection_size
        self.mot_accumulators = {}
        self.bbox_estimator = BBoxEstimator(
            'bbox_motion',
            'bbox',
            parameters['sigma_ca'],
            parameters['sigma_sr'],
            parameters['mvs_coeff'],
            parameters['Pcr'].to(dtype=self.dtype, device=self.device),
            parameters['cms_coeff'],
            self.dtype,
            self.device
        )

        self.confidence_estimator = ConfidenceEstimator(
            'bbox',
            'score',
            parameters['inlier_odds_hist']
        )
        self.tracker = SHTTracker(
            params=self.parameters,
            allocators=[
                self.bbox_estimator.allocator,
            ],
            initers=[
                self.bbox_estimator.initer,
                self.confidence_estimator.initer
            ],
            predictors=[
                self.bbox_estimator.predictor,
            ],
            comparators=[
                self.bbox_estimator.comparator,
                self.confidence_estimator.comparator
            ],
            updaters=[
                self.bbox_estimator.updater,
            ],
            validators=[
                self.bbox_estimator.validator
            ],
            max_tracks=self.parameters['max_tracks'],
            dtype=self.dtype,
            device=self.device
        )
        self.W_buf = deque()
        self.full_W = torch.eye(3, dtype=self.dtype, device=self.device)

    def track(self, detections, R, cd, img_shape=(480, 640), update_s=0.0333333333, W=None):
        R, cd, classification, score, z = self.preprocess_detections(R, cd, detections)

        self.full_W = torch.tensor(W, dtype=self.dtype, device=self.device) @ self.full_W
        self.W_buf.append(torch.tensor(W, dtype=self.dtype, device=self.device))

        if len(self.W_buf) > 10:
            self.full_W = self.full_W @ self.W_buf.popleft().inverse()

        self.tracker.predict(
            {
                'dt': update_s,
                'W': torch.tensor(W, dtype=self.dtype, device=self.device)
            }
        )
        self.tracker.validate()

        detections_sht = {
            'num': len(detections),
            'cd': cd,
            'bbox': (z, R),
            'class': classification,
            'score': score
        }

        self.tracker.observe(
            CameraDetectorState(
                self.parameters['pd'],
                self.tracker._compare(detections_sht, self.tracker.all_valid_tracks),
                cd
            )
        )

        associated_tracks, compares, new_tracks = self.tracker.update(
            detections_sht
        )

        x, P = self.tracker.data['bbox_motion']

        unassociated_tracks = self.tracker.valid_tracks.clone()
        unassociated_tracks[associated_tracks] = False
        unassociated_tlwh = x[unassociated_tracks]
        unassociated_tlwh[:, :2] -= unassociated_tlwh[:, 2:4] / 2
        det_tlwh = z  # OVERWRITTEN!
        det_tlwh[:, :2] -= det_tlwh[:, 2:] / 2

        self.tracker.llrs[self.tracker.all_valid_tracks] = self.tracker.llrs[self.tracker.all_valid_tracks].clamp(
            max=self.parameters['max_llr'], min=self.parameters['min_llr'])

        self.associated_tracks = associated_tracks
        return self.extract_current_track_data(tuple(img_shape), new_tracks)

    def preprocess_detections(self, R, cd, detections):
        z = torch.empty((len(detections), 4), dtype=self.dtype, device=self.device)
        if len(R.shape) == 2:
            R = R.to(dtype=self.dtype, device=self.device).repeat(len(detections), 1, 1)
        classification = torch.empty((len(detections), self.num_classes), dtype=self.dtype, device=self.device)
        score_t = torch.empty((len(detections),), dtype=self.dtype, device=self.device)
        for i, d in enumerate(detections):
            z[i] = torch.tensor(d.to_ccwh(self.detection_size))

            score = d.score
            score_t[i] = score
            classification[i, d.label - 1] = score
        return R, cd, classification, score_t, z

    def extract_current_track_data(self, img_shape, new_tracks):
        valid_tracks = self.tracker.all_valid_tracks
        if not valid_tracks.any():
            return []
        label = 1
        cc = self.tracker.data['bbox_motion'][0][valid_tracks, :2]
        wh = self.tracker.data['bbox_motion'][0][valid_tracks, 2:4]
        tlwh_scale = 1 / torch.tensor(img_shape[::-1], dtype=cc.dtype, device=cc.device).repeat(2)
        detected = torch.zeros(self.tracker.all_valid_tracks.shape, dtype=torch.bool, device=cc.device)
        detected[self.associated_tracks] = True
        return TrackData(
            all_valid_tracks=self.tracker.all_valid_tracks.detach().clone(),
            all_ids=self.tracker.track_ids.detach().clone(),
            ids=self.tracker.track_ids[valid_tracks].detach().clone(),
            llrs=self.tracker.llrs[valid_tracks].detach().clone(),
            new_tracks=new_tracks,
            tlwhs=(torch.cat((cc - wh / 2, wh), 1) * tlwh_scale[None]).detach().clone(),
            detected=detected,
            confidences=1 / (1 + (-self.tracker.llrs[valid_tracks]).exp()),
            label=label
        )

    def __str__(self):
        return 'SHTTracker_yolov5x'

    def reset(self, **kwargs):
        self.tracker.reset()


def create_tracker(best_dict, detection_size, run_buffer=True):
    tracker = MOTTracker(
        parameters=best_dict,
        detection_size=detection_size
    )
    tracker = DynamicSigmaAndClutterDensityTracker(
        tracker=tracker,
        params=best_dict
    )
    if run_buffer:
        tracker = BufferedLookaheadTracker(tracker, best_dict)
    return tracker


def get_parameters(prefix=''):
    estimated_params = torch.load(f'./data/{prefix}estimated_params.pt')
    manual_params = json.load(open(f'./data/{prefix}manual_params.json', 'r'))

    params = dict(
        boxsize_hist=torch.load(f'./data/{prefix}boxsize_hist.pt'),
        inlier_odds_hist=torch.load(f'./data/{prefix}inlier_odds_hist.pt'),
    )
    params.update(estimated_params)
    params.update(manual_params)

    return params

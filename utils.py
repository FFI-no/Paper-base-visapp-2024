import json

import numpy as np
import torch


def numpy_tlwh_to_tlbr(tlwhs):
    return np.concatenate((tlwhs[:, :2], tlwhs[:, :2] + tlwhs[:, 2:]), 1)


def to_numpy(detections, hw=None):
    return np.array([[d.score, d.id, d.label, d.to_tlwh(hw)] for d in detections], dtype=object)


def to_numpy_ltwh(detections, hw=None):
    return np.array([d.to_tlwh(hw) for d in detections])


def to_numpy_ids(tracks):
    return np.array([t.id for t in tracks], dtype=int)


def to_numpy_confidences(tracks):
    return np.array([t.score for t in tracks])


def quiet_divide(a, b):
    """Quiet divide function that does not warn about (0 / 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.true_divide(a, b)


def rect_min_max(r):
    min_pt = r[..., :2]
    size = r[..., 2:]
    max_pt = min_pt + size
    return min_pt, max_pt


def boxiou(a, b):
    """Computes IOU of two rectangles."""
    a = a[:, None].copy()
    b = b[None].copy()
    a_min, a_max = rect_min_max(a)
    b_min, b_max = rect_min_max(b)
    # Compute intersection.
    i_min = np.maximum(a_min, b_min)
    i_max = np.minimum(a_max, b_max)
    i_size = np.maximum(i_max - i_min, 0)
    i_vol = np.prod(i_size, axis=-1)
    # Get volume of union.
    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = np.prod(a_size, axis=-1)
    b_vol = np.prod(b_size, axis=-1)
    u_vol = a_vol + b_vol - i_vol
    return np.where(i_vol == 0, np.zeros_like(i_vol, dtype=np.float64),
                    quiet_divide(i_vol, u_vol))


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def write_results(filename, sequence_result):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        iter = zip(
            sequence_result.frame_ids,
            sequence_result.ids,
            sequence_result.dets,
            sequence_result.confidences
        )

        for frame_id, track_ids, tlwhs, scores in iter:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue

                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                          w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))


def get_hist_bin(lo_edges, lookup):
    return (torch.searchsorted(lo_edges, lookup) - 1).clamp(0)


class Hist1D:
    def __init__(self, hist, edges):
        self.hist = hist
        self.edges = edges[:-1]

    def __call__(self, lookup):
        return self.hist[get_hist_bin(self.edges, lookup)]


class Hist2D:
    def __init__(self, hist, edges0, edges1):
        self.hist = hist
        self.edges0 = edges0[:-1]
        self.edges1 = edges1[:-1]

    def __call__(self, lookup0, lookup1):
        return self.hist[get_hist_bin(self.edges0, lookup0), get_hist_bin(self.edges1, lookup1)]


def read_vo(path):
    vo_file = open(path, 'r')
    vo = json.load(vo_file)

    return vo


def estimate_W(frame_id, seq_ecc, detection_size_hw, in_size_hw):
    h, w = in_size_hw
    dh, dw = detection_size_hw
    ecc_scale = 2
    J = np.diag([ecc_scale * dw / w, ecc_scale * dh / h, 1.])
    J_inv = np.diag([w / (dw * ecc_scale), h / (dh * ecc_scale), 1.])
    W = np.eye(3)
    warp_id = str(frame_id)

    if warp_id in seq_ecc:
        W_str = seq_ecc[warp_id]  # , "1"
        if int(W_str[1]) != 1:

            cnt_back = 0
            found_back = False
            while (frame_id - cnt_back) >= 0:
                cnt_back += 1
                W_str_back = seq_ecc[str(frame_id - cnt_back)]
                if W_str_back[1] == "1":
                    found_back = True
                    break

            cnt_front = 0
            found_front = False
            while True:
                cnt_front += 1
                W_str_front = seq_ecc[str(frame_id + cnt_front)]
                if W_str_front[1] == "1":
                    found_front = True
                    break

            if found_front and found_back:
                W = (cnt_front / (cnt_front + cnt_back)) * np.array(W_str_front[0], dtype=np.float64) + (
                        cnt_back / (cnt_front + cnt_back)) * np.array(W_str_back[0], dtype=np.float64)
            elif found_front:
                W = np.array(W_str_front[0], dtype=np.float64)
            elif found_back:
                W = np.array(W_str_back[0], dtype=np.float64)
        else:
            W = np.array(W_str[0]).astype(np.float64)  # @ J_inv
            last_correct = W
        # print(W_str[1])
        W = J @ W @ J_inv
        # sigma_W = np.trace(np.array(seq_ecc[warp_id]).astype(dtype=np.float64) - np.identity(3)) * 100
    return W, 0

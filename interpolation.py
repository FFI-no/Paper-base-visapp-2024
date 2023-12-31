import numpy as np
import os
import motmetrics as mm

from mot_accumulator import MOTSequenceResult
from utils import estimate_W


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(
                frame=frame_id,
                id=track_id,
                x1=round(x1, 1),
                y1=round(y1, 1),
                w=round(w, 1),
                h=round(h, 1),
                s=round(score, 2)
            )
            f.write(line)


def transform_bbox(W, bbox):
    z = (W[None, 2, :2]@bbox[:2, None] + 1).squeeze(-1)
    bbox[:2] = ((W[:2, :2]@bbox[:2, None]).squeeze(-1) + W[:2, 2])/z

    return bbox


def get_seq_data(sequence_result):
    seq_data = []

    iter = zip(
        sequence_result.frame_ids,
        sequence_result.ids,
        sequence_result.dets,
        sequence_result.confidences
    )

    for frame_id, track_ids, tlwhs, confidences in iter:
        for track_id, tlwh, confidence in zip(track_ids, tlwhs, confidences):
            seq_data.append(np.array((
                frame_id,
                track_id,
                *tlwh.astype(float).round(1),
                round(confidence, 2),
                -1,
                -1,
                -1
            )))

    return np.vstack(seq_data)


def get_sequence_result(seq_data, name, label):
    sequence_result = MOTSequenceResult(name)
    for frame_id in range(int(round(seq_data[:, 0].min())), int(round(seq_data[:, 0].max())) + 1):
        mask = seq_data[:, 0] == frame_id

        sequence_result.add_frame_data(
            frame_id=frame_id,
            np_track_ids=seq_data[mask, 1].astype(int),
            np_tlwhs=seq_data[mask, 2:6],
            np_confidences=seq_data[mask, 6],
            np_labels=np.full(mask.sum(), label)
        )

    return sequence_result


def dti_sequence_result(sequence_result, seq_vo, hw, n_min=5):
    seq_data = get_seq_data(sequence_result)
    interpolated_seq_data = dti_seq(seq_data, seq_vo, hw, n_min)
    return get_sequence_result(interpolated_seq_data, sequence_result.seq_name, sequence_result.classes[-1][0])


def dti_seq(seq_data, seq_vo, hw, n_min):
    min_id = int(seq_data[:, 1].min())
    max_id = int(seq_data[:, 1].max())
    h, w = hw

    seq_results = []

    for track_id in range(min_id, max_id + 1):
        index = (seq_data[:, 1] == track_id)
        tracklet = seq_data[index]
        tracklet_dti = tracklet
        if tracklet.shape[0] == 0:
            continue

        n_frame = tracklet.shape[0]

        if n_frame > n_min:
            frames = tracklet[:, 0]
            frames_dti = {}
            for i in range(0, n_frame):
                right_frame = frames[i]
                if i > 0:
                    left_frame = frames[i - 1]
                else:
                    left_frame = frames[i]
                # disconnected track interpolation
                if 1 < right_frame - left_frame:
                    num_bi = int(right_frame - left_frame - 1)

                    W = [np.eye(3)]
                    for j in range(int(left_frame) + 1, int(right_frame) + 1):
                        W.append(estimate_W(j, seq_vo, (h, w), (h, w))[0]@W[-1])

                    Wi = [np.eye(3)]
                    for j in range(int(right_frame), int(left_frame), -1):
                        Wi.append(np.linalg.inv(estimate_W(j, seq_vo, (h, w), (h, w))[0])@Wi[-1])
                    Wi.reverse()

                    right_bbox = tracklet[i, 2:6]
                    left_bbox = tracklet[i - 1, 2:6]
                    for j in range(1, num_bi + 1):
                        curr_frame = j + left_frame
                        lbox = transform_bbox(W[j], np.copy(left_bbox))
                        rbox = transform_bbox(Wi[j], np.copy(right_bbox))

                        curr_bbox = (curr_frame - left_frame) * (rbox - lbox) / \
                                    (right_frame - left_frame) + lbox
                        frames_dti[curr_frame] = curr_bbox
            num_dti = len(frames_dti.keys())
            if num_dti > 0:
                data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                for n in range(num_dti):
                    data_dti[n, 0] = list(frames_dti.keys())[n]
                    data_dti[n, 1] = track_id
                    data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                    data_dti[n, 6:] = [1, -1, -1, -1]
                tracklet_dti = np.vstack((tracklet, data_dti))
        seq_results.append(tracklet_dti)

    seq_results = np.vstack(seq_results)
    return seq_results[seq_results[:, 0].argsort()]


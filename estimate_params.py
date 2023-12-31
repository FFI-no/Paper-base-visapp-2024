import matplotlib.pyplot as plt

import json
import motmetrics as mm
import numpy as np
from scipy import optimize
import skimage

import torch

from utils import to_numpy_ltwh, to_numpy_confidences, estimate_W, Hist2D, Hist1D
from sequence_reader import SequenceReader

class_names = {1: 'Pedestrian',
               2: 'POV',
               3: 'Car',
               4: 'Bicycle',
               5: 'Motorbike',
               6: 'NMV',
               7: 'SP',
               8: 'Dist',
               9: 'Ocl',
               10: 'OclOG',
               11: 'OclFull',
               12: 'Refl'}


def get_track_ids(seq):
    track_ids = set()
    for detections, gt in seq:
        for t in gt:
            track_ids.add(t.id)

    return list(track_ids)


def get_sequence_data(hw, img_size, seq, vo, iou_threshold, dtype, device):
    id_map = {id: i for i, id in enumerate(get_track_ids(seq))}
    num_frames = len(seq)
    num_tracks = len(id_map)
    h, w = img_size

    clutter_scores = []
    det_scores = []
    clutter_sizes = []
    det_sizes = []

    meas_data = torch.zeros((num_frames, num_tracks, 4), dtype=dtype, device=device)
    gt_data = torch.zeros((num_frames, num_tracks, 4), dtype=dtype, device=device)
    is_valid = torch.zeros((num_frames, num_tracks), dtype=torch.bool, device=device)
    is_detected = torch.zeros((num_frames, num_tracks), dtype=torch.bool, device=device)
    W = torch.empty((num_frames, 3, 3), dtype=dtype, device=device)
    sigma_W = torch.zeros((num_frames,), dtype=dtype, device=device)

    valid_indices = seq.valid_indices
    total_dets = 0

    for frame_id, (frame_idx, (detections, gt)) in zip(valid_indices, enumerate(seq)) \
            if valid_indices is not None else enumerate(enumerate(seq), start=1):

        detections = [det for det in detections if det.score > 0.1]

        total_dets += len(detections)
        tlwh_dets = to_numpy_ltwh(detections, hw=hw)
        scores = to_numpy_confidences(detections)
        tlwh_gt = to_numpy_ltwh(gt, hw=hw)
        iou_matrix = mm.distances.iou_matrix(tlwh_gt, tlwh_dets)

        best_dets = iou_matrix.argmin(axis=1) if len(iou_matrix) > 0 else []
        best_tracks = iou_matrix.argmin(axis=0) if len(iou_matrix) > 0 else []

        clutter_dets = np.ones((len(detections)), dtype=bool)
        for gt_idx, track in enumerate(gt):
            if not (track.label in [1, -1]):
                continue

            track_idx = id_map[track.id]
            is_valid[frame_idx, track_idx] = True
            gt_data[frame_idx, track_idx, :] = torch.tensor(tlwh_gt[gt_idx])
            gt_data[frame_idx, track_idx, :2] += gt_data[frame_idx, track_idx, 2:] / 2

            det_idx = best_dets[gt_idx]

            if gt_idx == best_tracks[det_idx] and iou_matrix[gt_idx, det_idx] < iou_threshold:
                clutter_dets[det_idx] = False
                is_detected[frame_idx, track_idx] = True
                meas_data[frame_idx, track_idx, :] = torch.tensor(tlwh_dets[det_idx])
                meas_data[frame_idx, track_idx, :2] += meas_data[frame_idx, track_idx, 2:] / 2

        clutter_scores += np.atleast_1d(scores[clutter_dets]).tolist()
        det_scores += np.atleast_1d(scores).tolist()
        clutter_sizes += np.atleast_1d(tlwh_dets[clutter_dets, 2]).tolist()
        det_sizes += np.atleast_1d(tlwh_dets[:, 2]).tolist()

        W[frame_idx] = torch.tensor(estimate_W(frame_id, vo, hw, in_size_hw=(h, w))[0])

    return gt_data, meas_data, W, sigma_W, is_valid, is_detected, clutter_scores, det_scores, clutter_sizes, det_sizes


def kf_predict(x, P, F, Q):
    return torch.matmul(F, x[:, :, None]).squeeze(-1), torch.matmul(torch.matmul(F, P), F.T) + Q


def kf_update(x, P, z, R, H):
    hx = torch.matmul(H, x[:, :, None]).squeeze(-1)
    y = z - hx
    S = H @ P @ H.T + R

    L = torch.linalg.cholesky_ex(S.transpose(1, 2))[0]
    HtSi = torch.cholesky_solve(H, L).transpose(1, 2)
    K = torch.bmm(P, HtSi)
    Kt = K.transpose(-2, -1)
    C = torch.eye(6, 6, dtype=P.dtype, device=P.device) - K @ H
    Ct = C.transpose(-2, -1)

    return x + torch.matmul(K, y[:, :, None]).squeeze(-1), C @ P @ Ct + K @ R @ Kt


def getF(dt, dtype, device):
    F = torch.eye(6, 6, dtype=dtype, device=device)
    F[:2, -2:].diagonal().fill_(dt)

    return F


def getQ(dt, sigma_center_acc, sigma_size_rate, dtype, device):
    batch_size = sigma_center_acc.shape[0]
    Q = torch.empty((batch_size, 6, 6), dtype=dtype, device=device)
    I2 = torch.eye(2, 2, dtype=dtype, device=device)[None, :, :]

    # center motion
    var_center_acc = (sigma_center_acc ** 2)[:, None, None]
    Q[:, :2, :2] = var_center_acc * dt ** 3 / 3 * I2
    Q[:, -2:, -2:] = var_center_acc * dt * I2
    Q[:, :2, -2:] = var_center_acc * dt ** 2 / 2 * I2
    Q[:, -2:, :2] = Q[:, :2, -2:]

    # size motion
    var_size_rate = (sigma_size_rate ** 2)[:, None, None]
    Q[:, :2, 2:-2] = 0
    Q[:, 2:-2, :2] = 0
    Q[:, -2:, 2:-2] = 0
    Q[:, 2:-2, -2:] = 0
    Q[:, 2:-2, 2:-2] = var_size_rate * dt ** 3 / 3 * I2

    return Q


def getH(dtype, device):
    return torch.eye(4, 6, dtype=dtype, device=device)


def getR(sc, ss, dtype, device):
    return torch.diag(torch.tensor((sc, sc, ss, ss), dtype=dtype, device=device) ** 2)


def squared_mah_dist(P, y):
    L = torch.linalg.cholesky_ex(P)[0]
    e = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False)

    return e.transpose(-1, -2) @ e


def neg_log_norm_pdf(P, y):
    L = torch.linalg.cholesky_ex(P)[0]
    e = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False)

    mah_dist2 = (e.transpose(-1, -2) @ e).squeeze(-1).squeeze(-1)

    res = mah_dist2 + 2 * torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(dim=1)
    return res


def compute_cov_sum(meas_scale, gt_data, meas_data, is_detected, dtype, device):
    num_frames, num_tracks = gt_data.shape[:2]

    cov_sum = torch.zeros(4, 4, dtype=dtype, device=device)

    for frame_idx in range(num_frames):
        hx = gt_data[frame_idx, is_detected[frame_idx]]
        z = meas_data[frame_idx, is_detected[frame_idx]]

        y = hx - z
        y = y.unsqueeze(-1)

        cov_sum += (y @ y.transpose(-2, -1)).sum(dim=0)

    J = torch.diag(1 / np.sqrt(meas_scale).repeat(2))

    return J @ cov_sum @ J, is_detected.sum()


def compute_R(sequences, dtype, device):
    cov_sum = torch.zeros(4, 4, dtype=dtype, device=device)
    n = 0

    for name in sequences:
        seq = sequences[name]
        gt = seq['gt']
        meas = seq['meas']
        is_detected = seq['is_detected']
        meas_scale = seq['meas_scale']

        cov_sumi, ni = compute_cov_sum(meas_scale, gt, meas, is_detected, dtype, device)
        cov_sum += cov_sumi
        n += ni

    return cov_sum / (n - 1)


def compute_seq_init_Pcr_coeff(dt, gt, is_valid, W, dtype, device):
    num_frames, num_tracks = gt.shape[:2]
    has_first = torch.zeros(num_tracks, dtype=torch.bool, device=device)
    first_pos = torch.empty((num_tracks, 2), dtype=dtype, device=device)
    first_t = torch.empty(num_tracks, dtype=dtype, device=device)
    size = torch.empty(num_tracks, dtype=dtype, device=device)
    second_pos = torch.empty((num_tracks, 2), dtype=dtype, device=device)
    dts = torch.empty(num_tracks, dtype=dtype, device=device)
    first_idx = -1 * torch.ones(num_tracks, dtype=dtype, device=device)
    second_idx = -1 * torch.ones(num_tracks, dtype=dtype, device=device)

    has_second = torch.zeros(num_tracks, dtype=torch.bool, device=device)

    t = 0

    for frame_idx in range(num_frames):
        warp_idxs = has_first & ~has_second
        z = (W[None, None, frame_idx, 2, :2] @ first_pos[warp_idxs, :2, None] + 1).squeeze(-1)
        first_pos[warp_idxs, :2] = ((W[None, frame_idx, :2, :2] @ first_pos[warp_idxs, :2, None]).squeeze(-1) + W[
                                                                                                                frame_idx,
                                                                                                                :2,
                                                                                                                2]) / z

        second_idxs = is_valid[frame_idx] & has_first & ~has_second

        if second_idxs.any():
            second_pos[second_idxs] = gt[frame_idx, second_idxs, :2]
            dts[second_idxs] = (t - first_t[second_idxs])
            second_idx[second_idxs] = frame_idx
            has_second[second_idxs] = True

        if has_second.all():
            break

        first_idxs = is_valid[frame_idx] & ~has_first

        if first_idxs.any():
            first_pos[first_idxs] = gt[frame_idx, first_idxs, :2]
            first_t[first_idxs] = t
            size[first_idxs] = gt[frame_idx, first_idxs, 2]
            first_idx[first_idxs] = frame_idx
            has_first[first_idxs] = True

        t += dt

    vel = (second_pos[has_second] - first_pos[has_second]) / dts[has_second].unsqueeze(-1)

    size = size[has_second].unsqueeze(-1).repeat((1, 2))
    vel = (vel / size).unsqueeze(-1)
    res = (vel @ vel.transpose(-2, -1)).sum(dim=0)

    return res, num_tracks


def compute_init_Pcr_coeff(sequences, dtype, device):
    cov_sum = torch.zeros(2, 2, dtype=dtype, device=device)
    n = 0

    for name in sequences:
        seq = sequences[name]
        gt = seq['gt']
        is_valid = seq['is_valid']
        dt = seq['dt']
        W = seq['W']

        cov_sumi, ni = compute_seq_init_Pcr_coeff(dt, gt, is_valid, W, dtype, device)

        cov_sum += cov_sumi
        n += ni

    return cov_sum / (n - 1)


def seq_score_sigma_acc(data, is_detected, W, sigma_W, dt, sca_coeff, ssr_coeff, init_Pcr_coeff, R,
                        dtype, device):
    F = getF(dt, dtype, device)
    H = getH(dtype, device)
    dim = F.shape[0]

    num_frames, num_tracks = data.shape[:2]

    x = torch.zeros((num_tracks, dim), dtype=dtype, device=device)
    P = torch.zeros((num_tracks, dim, dim), dtype=dtype, device=device)
    is_inited = torch.zeros((num_tracks,), dtype=torch.bool, device=device)

    neg_log_lik = 0

    for i in range(num_frames):
        if ((P - P.transpose(-2, -1)).sum(dim=1).sum(dim=1) > 0).any():
            print('non sym')

        # predict inited tracks
        size = x[is_inited, 2]
        Q = getQ(dt, sca_coeff * size, ssr_coeff * size, dtype, device)
        x[is_inited], P[is_inited] = kf_predict(x[is_inited], P[is_inited], F, Q)

        if ((P - P.transpose(-2, -1)).sum(dim=1).sum(dim=1) > 0).any():
            print('non sym')

        if x[is_inited].shape[0] > 0:
            z = (W[None, None, i, 2, :2] @ x[is_inited, :2, None] + 1).squeeze(-1)
            x[is_inited, :2] = ((W[None, i, :2, :2] @ x[is_inited, :2, None]).squeeze(-1) + W[i, :2, 2]) / z

        # compare to gt
        comp_idxs = is_inited & is_detected[i]
        y = x[comp_idxs, :4] - data[i, comp_idxs]

        R_W = torch.diag(torch.tensor((sigma_W[i], sigma_W[i], 0, 0), dtype=R.dtype, device=R.device) ** 2)

        err = neg_log_norm_pdf(P[comp_idxs, :4, :4] + R + R_W, y).sum()
        neg_log_lik += err

        if err.isnan().any():
            print('nan')

        # update inited tracks
        upd_idxs = is_inited & is_detected[i]
        x[upd_idxs], P[upd_idxs] = kf_update(x[upd_idxs], P[upd_idxs], data[i, upd_idxs, :], R + R_W, H)

        if ((P - P.transpose(-2, -1)).sum(dim=1).sum(dim=1) > 0).any():
            print('non sym')

        # init new tracks
        init_idxs = is_detected[i] & ~is_inited
        x[init_idxs, :4] = data[i, init_idxs]
        P[init_idxs, :4, :4] = R + R_W
        P[init_idxs, -2:, -2:] = data[i, init_idxs, 2].unsqueeze(-1).unsqueeze(-1) ** 2 * init_Pcr_coeff.unsqueeze(0)
        is_inited[init_idxs] = True

        if ((P - P.transpose(-2, -1)).sum(dim=1).sum(dim=1) > 0).any():
            print('non sym')

    err = neg_log_lik.cpu().detach().item()

    return err


def score_sigma_acc(sequences, hw, init_Pcr_coeff, R, xi):
    sca_coeff, ssr_coeff = np.exp(xi)
    score = 0

    for name in sequences:
        seq = sequences[name]

        gt = seq['gt']
        meas = seq['meas']
        W = seq['W']
        sigma_W = seq['sigma_W']
        is_valid = seq['is_valid']
        is_detected = seq['is_detected']
        dt = seq['dt']
        meas_scale = seq['meas_scale']

        dtype = gt.dtype
        device = gt.device

        J = torch.diag(np.sqrt(meas_scale).repeat(2))

        score += seq_score_sigma_acc(
            meas,
            is_detected,
            W,
            sigma_W,
            dt,
            sca_coeff,
            ssr_coeff,
            init_Pcr_coeff,
            J @ R @ J,
            dtype,
            device
        )

    return score


def read_sequence_data(reader, vo, target_hw, iou_threshold, dtype, device):
    all_clutter_scores = []
    all_det_scores = []
    all_clutter_sizes = []
    all_det_sizes = []

    sequences = {}

    for seq, info in reader:
        name = info['name']
        dt = info['update_s']

        w, h = info['imWidth'], info['imHeight']
        gt_data, meas_data, W, sigma_W, is_valid, is_detected, clutter_scores, det_scores, clutter_sizes, det_sizes = get_sequence_data(target_hw,
                                                                                                             [h, w],
                                                                                                             seq,
                                                                                                             vo[name],
                                                                                                             iou_threshold,
                                                                                                             dtype,
                                                                                                             device)
        all_clutter_scores += clutter_scores
        all_det_scores += det_scores
        all_clutter_sizes += clutter_sizes
        all_det_sizes += det_sizes

        meas_scale = torch.tensor((target_hw[1] / info['imWidth'], target_hw[0] / info['imHeight']), dtype=dtype,
                                  device=device)

        sequences[name] = {
            'gt': gt_data,
            'meas': meas_data,
            'W': W,
            'sigma_W': sigma_W,
            'is_valid': is_valid,
            'is_detected': is_detected,
            'dt': dt,
            'meas_scale': meas_scale,
        }

    return sequences, (all_clutter_scores, all_det_scores, all_clutter_sizes, all_det_sizes)


def generate_boxsize_hist(reader, hw, prefix=''):
    boxsizes = []

    for seq, _ in reader:
        for detections, gt in seq:
            tlwh_dets = to_numpy_ltwh(detections, hw=hw)
            boxsizes += [x[2] for x in tlwh_dets]

    num_bins = 200
    h, bins, _ = plt.hist(boxsizes, density=True, log=True, bins=num_bins, range=(0, 1000))
    hist = Hist1D(torch.tensor(h), torch.tensor(bins))
    torch.save(hist, f'./data/{prefix}boxsize_hist.pt')

def generate_inlier_odds_hist(clutter_scores, det_scores, clutter_sizes, det_sizes, prefix=''):
    num_bins = 30
    score_edges = np.linspace(0, 1, num_bins + 1)
    size_edges = np.exp(np.linspace(np.log(min(det_sizes)), np.log(max(det_sizes)), num_bins + 1))
    bins = [score_edges, size_edges]

    clutter_bins = plt.hist2d(clutter_scores, clutter_sizes, bins=bins)[0]
    det_bins = plt.hist2d(det_scores, det_sizes, bins=bins)[0]

    detection_lr = (skimage.filters.gaussian(det_bins - clutter_bins, 2, mode='reflect')) / (
        skimage.filters.gaussian(clutter_bins, 2, mode='reflect'))
    detection_lr_hist = Hist2D(torch.tensor(detection_lr), torch.tensor(score_edges), torch.tensor(size_edges))
    torch.save(detection_lr_hist, f"./data/{prefix}inlier_odds_hist.pt")


def estimate_process_noise(sequences, hw, Pcr, R):
    initial_sigma_ca = 1
    initial_sigma_sr = 5

    res = optimize.minimize(
        lambda xi: score_sigma_acc(sequences, hw, Pcr, R, xi),
        np.log((initial_sigma_ca, initial_sigma_sr)),
        tol=1e-5,
        method='L-BFGS-B'
    )
    sigma_ca = np.exp(res.x[0])
    sigma_sr = np.exp(res.x[1])

    return sigma_ca, sigma_sr


def read_vo(path):
    vo_file = open(path, 'r')
    vo = json.load(vo_file)

    return vo


def estimate_parameters_mot17():
    print('--------------------------')
    print('Began estimation for MOT17')
    dtype = torch.float64
    device = torch.device('cpu')

    hw = (1080, 1920)
    vo = read_vo('./data/MOT17/vo.json')
    prefix = 'mot17_'

    detections_filename = 'base_det'
    print('Loading data...')
    reader = SequenceReader(
        [
            './data/MOT17/train/MOT17-02-FRCNN',
            './data/MOT17/train/MOT17-04-FRCNN',
            './data/MOT17/train/MOT17-10-FRCNN',
            './data/MOT17/train/MOT17-11-FRCNN',
            './data/MOT17/train/MOT17-13-FRCNN',
        ],
        detections_filename=detections_filename
    )
    sequences = read_sequence_data(reader, vo, hw, 0.2, dtype, device)[0]

    print('Estimating R...')
    R = compute_R(sequences, dtype, device)
    print(f'R:\n{R}')

    print('Loading some more data...')
    reader = SequenceReader(
        './data/MOT17/train/MOT17-*-FRCNN',
        detections_filename=detections_filename
    )
    sequences, clutter_counts = read_sequence_data(reader, vo, hw, 0.3, dtype, device)

    print('Estimating Pcr...')
    Pcr = compute_init_Pcr_coeff(sequences, dtype, device)
    print(f'Pcr:\n{Pcr}')

    print('Generating p_w histogram... ', end='')
    generate_boxsize_hist(reader, hw, prefix)
    print('done')

    print('Generating P_C histgram... ', end='')
    generate_inlier_odds_hist(*clutter_counts, prefix)
    print('done')

    print('Performing MLE for process noise parameters...')
    sigma_ca, sigma_sr = estimate_process_noise(sequences, hw, Pcr, R)
    print(f'sigma_ca: {sigma_ca:.2f}')
    print(f'sigma_sr: {sigma_sr:.2f}')

    params_path = f"./data/{prefix}estimated_params.pt"
    torch.save(
        dict(
            Pcr=Pcr,
            R=R,
            sigma_ca=round(sigma_ca, 2),
            sigma_sr=round(sigma_sr, 2)
        ),
        params_path
    )
    print(f'wrote estimated params to {params_path}')


def estimate_parameters_mot20():
    print('--------------------------')
    print('Began estimation for MOT20')
    dtype = torch.float64
    device = torch.device('cpu')

    hw = (1080, 1920)
    vo = read_vo('./data/MOT20/vo.json')
    prefix = 'mot20_'

    print('Loading data...')
    reader = SequenceReader(
        './data/MOT20/train/MOT20-*',
        detections_filename='base_det'
    )
    sequences, clutter_counts = read_sequence_data(reader, vo, hw, 0.3, dtype, device)

    print('Estimating R...')
    R = compute_R(sequences, dtype, device)
    print(f'R:\n{R}')

    print('Estimating Pcr...')
    Pcr = compute_init_Pcr_coeff(sequences, dtype, device)
    print(f'Pcr:\n{Pcr}')

    print('Generating p_w histogram... ', end='')
    generate_boxsize_hist(reader, hw, prefix)
    print('done')

    print('Generating P_C histgram... ', end='')
    generate_inlier_odds_hist(*clutter_counts, prefix)
    print('done')

    print('Performing MLE for process noise parameters...')
    sigma_ca, sigma_sr = estimate_process_noise(sequences, hw, Pcr, R)
    print(f'sigma_ca: {sigma_ca:.2f}')
    print(f'sigma_sr: {sigma_sr:.2f}')

    params_path = f"./data/{prefix}estimated_params.pt"
    torch.save(
        dict(
            Pcr=Pcr,
            R=R,
            sigma_ca=round(sigma_ca, 2),
            sigma_sr=round(sigma_sr, 2)
        ),
        params_path
    )
    print(f'wrote estimated params to {params_path}')


if __name__ == '__main__':
    estimate_parameters_mot17()
    estimate_parameters_mot20()

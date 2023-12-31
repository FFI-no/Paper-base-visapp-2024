import kalman
import math
import torch


class BBoxAllocator:
    def __init__(self, state_id, dtype, device):
        self.state_id = state_id
        self.dtype = dtype
        self.device = device

    def __call__(self, num_tracks):
        x = torch.empty((num_tracks, 6), dtype=self.dtype, device=self.device)
        P = torch.empty((num_tracks, 6, 6), dtype=self.dtype, device=self.device)

        return self.state_id, (x, P)


class BBoxIniter:
    def __init__(self, state_id, meas_id, init_Pcr_coeff):
        self.state_id = state_id
        self.meas_id = meas_id
        self.init_Pcr_coeff = init_Pcr_coeff

    def __call__(self, data, llrs, detections, new_tracks, new_detections):
        x, P = data[self.state_id]
        z, R = detections[self.meas_id]

        x[new_tracks, :4] = z[new_detections]
        x[new_tracks, 4:] = 0

        P[new_tracks] = 0
        P[new_tracks, :4, :4] = R[new_detections]
        P[new_tracks, -2:, -2:] = x[new_tracks, 2, None, None] ** 2 * self.init_Pcr_coeff[None]


class BBoxPredictor:
    def __init__(self, state_id, sigma_ca, sigma_sr, mvs_coeff):
        self.state_id = state_id
        self.sigma_ca = sigma_ca
        self.sigma_sr = sigma_sr
        self.mvs_coeff = mvs_coeff

    def __call__(self, data, valid_tracks, prediction_set):
        x, P = data[self.state_id]

        if not valid_tracks.any():
            return

        dtype = x.dtype
        device = x.device

        dt = prediction_set['dt']
        F = self.getF(dt, dtype, device)

        size = x[valid_tracks, 2:4].max(dim=-1).values
        sigma_center_acc = self.sigma_ca * size
        sigma_size_rate = self.sigma_sr * size

        Q = self.getQ(dt, sigma_center_acc, sigma_size_rate, dtype, device)
        x[valid_tracks], P[valid_tracks] = kalman.predict(x[valid_tracks], P[valid_tracks], F, Q)

        # constrain velocity uncertainty
        vz = x[valid_tracks, -2:]
        vR = x[valid_tracks, 2, None, None] ** 2 * (self.mvs_coeff ** 2 * torch.eye(2, dtype=x.dtype, device=x.device))[
            None]
        vH = torch.zeros((2, 6), dtype=x.dtype, device=x.device)
        vH[-2, -2] = 1
        vH[-1, -1] = 1
        x[valid_tracks], P[valid_tracks] = kalman.update(x[valid_tracks], P[valid_tracks], vz, vR, vH)

        if 'W' in prediction_set and prediction_set['W'] is not None:
            W = prediction_set['W']
            z = (W[None, None, 2, :2] @ x[valid_tracks, :2, None] + 1).squeeze(-1)
            x[valid_tracks, :2] = ((W[None, :2, :2] @ x[valid_tracks, :2, None]).squeeze(-1) + W[:2, 2]) / z

    def getF(self, dt, dtype, device):
        F = torch.eye(6, 6, dtype=dtype, device=device)
        F[:2, -2:].diagonal().fill_(dt)

        return F

    def getlF(self, dt, dtype, device):
        lF = torch.eye(4, 4, dtype=dtype, device=device)
        lF[:2, -2:].diagonal().fill_(dt)

        return lF

    def getQ(self, dt, sigma_center_acc, sigma_size_rate, dtype, device):
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

    def getMotionQ(self, dt, sigma_center_acc, dtype, device):
        Q = torch.empty((4, 4), dtype=dtype, device=device)
        I2 = torch.eye(2, 2, dtype=dtype, device=device)[None, :, :]

        # center motion
        var_center_acc = sigma_center_acc ** 2
        Q[:2, :2] = var_center_acc * dt ** 3 / 3 * I2
        Q[-2:, -2:] = var_center_acc * dt * I2
        Q[:2, -2:] = var_center_acc * dt ** 2 / 2 * I2
        Q[-2:, :2] = Q[:2, -2:]

        return Q[None]


class BBoxComparator:
    def __init__(self, state_id, meas_id):
        self.state_id = state_id
        self.meas_id = meas_id

    def __str__(self):
        return "BBoxComparator"

    def __call__(self, data, detections, valid_tracks):
        x = data[self.state_id][0][valid_tracks]
        P = data[self.state_id][1][valid_tracks]

        if len(x) == 0:
            return 0

        z, R = detections[self.meas_id]

        y = x[:, None, :4] - z[None]
        S = P[:, None, :4, :4] + R[None]

        L = torch.linalg.cholesky_ex(S)[0]
        e = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False)

        mah_dist2 = (e.transpose(-1, -2) @ e).squeeze(-1).squeeze(-1)
        log_det_S = 2 * torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)

        llr = -0.5 * 4 * math.log(2 * math.pi) - 0.5 * mah_dist2 - 0.5 * log_det_S
        return llr


class BBoxUpdater:
    def __init__(self, state_id, meas_id, dtype, device):
        self.state_id = state_id
        self.meas_id = meas_id
        self.H = torch.eye(4, 6, dtype=dtype, device=device)
        self.lH = torch.eye(4, 6, dtype=dtype, device=device)

    def __call__(self, data, detections, associated_tracks, associated_measurements):
        if len(associated_tracks) == 0:
            return

        x = data[self.state_id][0]
        P = data[self.state_id][1]
        z, R = detections[self.meas_id]

        x[associated_tracks], P[associated_tracks] = kalman.update(x[associated_tracks], P[associated_tracks],
                                                                   z[associated_measurements],
                                                                   R[associated_measurements], self.H)


class BBoxValidator:
    def __init__(self, state_id, cms_coeff):
        self.state_id = state_id
        self.cms_coeff = cms_coeff

    def __call__(self, data, valid_tracks):
        sigmas = data[self.state_id][1][valid_tracks, :2, :2].diagonal(dim1=-2, dim2=-1).sqrt()
        center_max_sigmas = data[self.state_id][0][valid_tracks, 2:4] * self.cms_coeff

        return (sigmas < center_max_sigmas).all(dim=-1)


class BBoxEstimator:
    def __init__(self, state_id, meas_id, sigma_ca, sigma_sr, mvs_coeff, init_Pcr_coeff, cms_coeff, dtype, device):
        self.allocator = BBoxAllocator(state_id, dtype, device)
        self.initer = BBoxIniter(state_id, meas_id, init_Pcr_coeff)
        self.predictor = BBoxPredictor(state_id, sigma_ca, sigma_sr, mvs_coeff)
        self.comparator = BBoxComparator(state_id, meas_id)
        self.updater = BBoxUpdater(state_id, meas_id, dtype, device)
        self.validator = BBoxValidator(state_id, cms_coeff)

import torch


def cholesky_inverse(M):
    L = torch.linalg.cholesky_ex(M)[0]
    I = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)

    return torch.cholesky_solve(I, L)


def predict(x, P, F, Q, Bu=0):
    return torch.matmul(F, x[:, :, None]).squeeze(-1) + Bu, torch.matmul(torch.matmul(F, P), F.T) + Q


def if_predict(y, Y, Fi, Qi, Bu=0):
    '''
    source: Brown (Random Signals and Applied Kalman Filtering) kap 5.1
    + rett frem utledning for å inkludere pådrag
    '''
    M = Fi.T@Y@Fi
    C = M@cholesky_inverse(M + Qi)
    Ct = C.transpose(-2, -1)
    L = torch.eye(Y.shape[-1], dtype=Y.dtype, device=Y.device) - C
    Lt = L.transpose(-2, -1)

    Y_bar = L@M@Lt + C@Qi@Ct
    return (L@Fi.T@y[:, :, None] + Y_bar@Bu.unsqueeze(-1)).squeeze(-1), Y_bar


def update(x, P, z, R, H):
    hx = torch.matmul(H, x[:, :, None]).squeeze(-1)
    y = z - hx
    S = H@P@H.T + R

    L = torch.linalg.cholesky_ex(S.transpose(1, 2))[0]
    HtSi = torch.cholesky_solve(H, L).transpose(1, 2)
    K = torch.bmm(P, HtSi)
    Kt = K.transpose(-2, -1)
    C = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device) - K@H
    Ct = C.transpose(-2, -1)

    return x + torch.matmul(K, y[:, :, None]).squeeze(-1), C@P@Ct + K@R@Kt


class Allocator:
    def __init__(self, state_id, dim, dtype, device):
        self.state_id = state_id
        self.dim = dim
        self.dtype = dtype
        self.device = device

    def __call__(self, num_tracks):
        x = torch.empty((num_tracks, self.dim), dtype=self.dtype, device=self.device)
        P = torch.empty((num_tracks, self.dim, self.dim), dtype=self.dtype, device=self.device)
        return self.state_id, (x, P)

class LinearUpdater:
    def __init__(self, state_id, meas_id, H):
        self.state_id = state_id
        self.meas_id = meas_id
        self.H = H

    def __call__(self, data, detections, associated_tracks, associated_measurements):
        if len(associated_tracks) == 0:
            return

        x = data[self.state_id][0]
        P = data[self.state_id][1]
        z, R, _ = detections[self.meas_id]

        x[associated_tracks], P[associated_tracks] = update(x[associated_tracks], P[associated_tracks], z[associated_measurements], R[associated_measurements], self.H)

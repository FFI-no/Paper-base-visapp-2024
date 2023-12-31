from lapjv import lapjv
import math
import torch


class SHTTracker:
    '''
    allocators: Iterable of callables which take the number of tracks to allocate storage for, and returns a tuple of a unique ID for the type, and the storage
    '''

    def __init__(self, params, allocators, initers, predictors, comparators, updaters, validators, max_tracks, dtype, device):
        self.params = params
        self.allocators = allocators
        self.initers = initers
        self.predictors = predictors
        self.comparators = comparators
        self.updaters = updaters
        self.validators = validators
        self.dtype = dtype
        self.device = device

        self.track_ids = torch.empty(max_tracks, dtype=torch.int32, device=device)
        self.next_track_id = 0
        self.all_valid_tracks = torch.empty(max_tracks, dtype=torch.bool, device=device)
        self.valid_tracks = torch.empty(max_tracks, dtype=torch.bool, device=device)
        self.valid_candidate_tracks = torch.empty(max_tracks, dtype=torch.bool, device=device)
        self.reset()

        self.data = {id: data for id, data in
                     (allocator(max_tracks) for allocator in self.allocators)}
        self.llrs = torch.empty(max_tracks, dtype=self.dtype, device=device)

    def reset(self):
        self.all_valid_tracks[:] = False
        self.valid_tracks[:] = False
        self.valid_candidate_tracks[:] = False
        self.next_track_id = 0
        self.unused_tracks = set(range(len(self.all_valid_tracks)))

    def predict(self, prediction_set):
        for predictor in self.predictors:
            predictor(self.data, self.all_valid_tracks, prediction_set)

    def validate(self):
        invalidated_tracks = self.all_valid_tracks.clone()

        for validator in self.validators:
            tracks_to_check = invalidated_tracks.clone()
            invalidated_tracks[tracks_to_check] = ~validator(self.data, tracks_to_check)

        self.all_valid_tracks[invalidated_tracks] = False
        self.valid_tracks[invalidated_tracks] = False
        self.valid_candidate_tracks[invalidated_tracks] = False

        for track_idx in invalidated_tracks.nonzero().cpu():
            self.unused_tracks.add(track_idx.item())

    def update(self, detections):
        num_detections = detections['num']

        compares = self._compare(detections, self.all_valid_tracks)
        cd = detections['cd']

        unused_detections = torch.ones(num_detections, dtype=torch.bool, device=self.device)
        valid_compares = compares[self.valid_tracks[self.all_valid_tracks]]
        associated_tracks, associated_detections = self._associate(valid_compares, cd, self.valid_tracks, unused_detections)
        self._update(detections, associated_tracks, associated_detections)

        if len(associated_detections) > 0:
            unused_detections[associated_detections] = False

        if len(unused_detections) > 0 and self.valid_candidate_tracks.any():
            candidate_compares = compares[self.valid_candidate_tracks[self.all_valid_tracks]]
            candidate_compares = candidate_compares[:, unused_detections]
            associated_candidates, cand_associated_detections = self._associate(candidate_compares, cd[unused_detections],
                                                                           self.valid_candidate_tracks,
                                                                           unused_detections)
            self._update(detections, associated_candidates, cand_associated_detections)

            if len(cand_associated_detections) > 0:
                associated_tracks = torch.cat((associated_tracks, associated_candidates), dim=0)
                unused_detections[cand_associated_detections] = False

        new_tracks = torch.zeros(self.valid_tracks.shape, dtype=torch.bool, device=self.device)
        if len(unused_detections) > 0:
            new_tracks = self._init(detections, unused_detections)

        compares = self._compare(detections, self.all_valid_tracks)
        return associated_tracks, compares, new_tracks

    def observe(self, detector_state):
        self.llrs[self.all_valid_tracks] += detector_state(self.data, self.all_valid_tracks)

        self._kill_bad_tracks()
        self._kill_bad_candidate_tracks()
        self._promote_good_candidate_tracks()

    def _compare(self, detections, valid_tracks):
        num_detections = detections['num']

        num_tracks = valid_tracks.sum()
        llrs = torch.zeros((num_tracks, num_detections), dtype=self.dtype, device=self.device)

        for comparator in self.comparators:
            comp_llrs = comparator(self.data, detections, valid_tracks)
            llrs += comp_llrs

        return llrs

    def _associate(self, llrs, cd, valid_tracks=None, valid_detections=None):
        if len(llrs) == 0 or not valid_tracks.any() or not valid_detections.any():
            return torch.zeros(0, dtype=torch.long, device=llrs.device), torch.zeros(0, dtype=torch.long,
                                                                                     device=llrs.device)

        num_tracks, num_detections = llrs.shape
        n = max(num_tracks, num_detections)
        costs = torch.empty((n, n), dtype=llrs.dtype, device=llrs.device)
        max_cost = 1e6

        if num_tracks < n:
            costs[num_tracks:, :] = max_cost

        if num_detections < n:
            costs[:, num_detections:] = max_cost

        p = llrs.exp()
        ptilde = p / (cd + p.sum(dim=0) + 1e-14)

        llrs = ptilde.log() - (1 - ptilde).log()
        costs[:num_tracks, :num_detections] = -llrs
        llr_gate = math.log(self.params['gate']) - math.log(1 - self.params['gate'])

        costs[:num_tracks, :num_detections][llrs < llr_gate] = max_cost
        row_ind, _, _ = lapjv(costs.cpu().numpy())
        row_ind = torch.tensor(row_ind[:num_tracks], dtype=torch.long, device=llrs.device)
        matched_tracks = (costs[torch.arange(num_tracks, device=llrs.device), row_ind] != max_cost)

        matched_track_idxs = matched_tracks \
            if valid_tracks is None else \
            torch.arange(len(valid_tracks), device=llrs.device)[valid_tracks][matched_tracks]

        matched_detection_idxs = row_ind[matched_tracks] \
            if valid_detections is None else \
            torch.arange(len(valid_detections), device=llrs.device)[valid_detections][row_ind[matched_tracks]]

        return matched_track_idxs, matched_detection_idxs

    def _update(self, detections, associated_tracks, associated_measurements):
        for updater in self.updaters:
            updater(self.data, detections, associated_tracks, associated_measurements)

    def _init(self, detections, init_detections=None):
        num_detections = detections['num']
        new_tracks = torch.zeros(self.valid_tracks.shape, dtype=torch.bool, device=self.device)

        assert len(self.unused_tracks) > 0, "no available space to init track"
        for det_idx in init_detections.nonzero().cpu() if init_detections is not None else range(num_detections):
            track_idx = self.unused_tracks.pop()
            self.all_valid_tracks[track_idx] = True
            self.valid_candidate_tracks[track_idx] = True
            new_tracks[track_idx] = True

        self.llrs[new_tracks] = 0

        for initer in self.initers:
            initer(self.data, self.llrs, detections, new_tracks, init_detections)

        self.all_valid_tracks[new_tracks] = True

        id_begin = self.next_track_id
        id_end = id_begin + new_tracks.sum()
        self.track_ids[new_tracks] = torch.arange(id_begin, id_end, dtype=self.track_ids.dtype,
                                                  device=self.track_ids.device)
        self.next_track_id = id_end

        self.valid_candidate_tracks[new_tracks] = True
        return new_tracks

    def _kill_bad_tracks(self):
        killed = self.valid_tracks & (self.llrs < self.params['min_track_llr'])

        if not killed.any():
            return

        self.valid_tracks[killed] = False
        self.all_valid_tracks[killed] = False

        for track_idx in killed.nonzero().cpu():
            self.unused_tracks.add(track_idx.item())

    def _kill_bad_candidate_tracks(self):
        killed = self.valid_candidate_tracks & (self.llrs < self.params['min_candidate_llr'])

        if not killed.any():
            return

        self.valid_candidate_tracks[killed] = False
        self.all_valid_tracks[killed] = False

        for track_idx in killed.nonzero().cpu():
            self.unused_tracks.add(track_idx.item())

    def _promote_good_candidate_tracks(self):
        promoted_tracks = self.valid_candidate_tracks & (self.llrs > self.params['candidate_llr_thresh'])

        if not promoted_tracks.any():
            return

        self.valid_candidate_tracks[promoted_tracks] = False
        self.valid_tracks[promoted_tracks] = True

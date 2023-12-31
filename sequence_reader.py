import time

import glob
import os

import numpy as np
import pandas as pd

from sequence_helper import load_motchallenge
from mot_tracker import Detection

det_filename_to_size = dict(
    det_yolox=(800, 1440),
    det_yolox2=(800, 1440),
    det_yolox_train=(800, 1440),
    det_yolox_val=(800, 1440)
)


class FrameIterator:
    def __init__(self, path, detections_filename='det', gt_path='gt/gt.txt', seq_split=None):
        self._groundtruth_frame = None
        self._detections = {}
        self._detections_frame = None
        self.path = path
        self._info = None
        self.iter_index = 0
        self.detections_filename = detections_filename
        self.valid_indices = self.get_valid_indices(seq_split)
        self.seq_split = seq_split
        self.have_gt = True
        self._groundtruth = {}
        self.gt_path = gt_path

    @property
    def gt_path(self):
        return self._gt_path

    @gt_path.setter
    def gt_path(self, value):
        if not os.path.exists(value):
            value = os.path.join(self.path, value)

        self._gt_path = value
        self.have_gt = os.path.exists(self._gt_path)

    def get_valid_indices(self, seq_split):
        if seq_split is not None:
            with open(os.path.join(self.path, seq_split + '.txt'), 'r') as f:
                return np.array(sorted([int(l.strip()) for l in f.readlines()]))
        return None

    def gt_frame_to_tracks(self, gt_frame, index):
        w, h = self.info['imWidth'], self.info['imHeight']
        tracks = []
        if gt_frame is None:
            return tracks
        if index not in gt_frame.index:
            return tracks
        for j, track in gt_frame.loc[index].iterrows():
            # if int(track['ClassId']) != 1:
            #     continue
            tracks.append(Detection(int(track['ClassId']), track['Confidence'], 0, track['X'], track['Y'],
                                    track['X'] + track['Width'],
                                    track['Y'] + track['Height'], img_size=(h, w), id=j,
                                    visibility=track['Visibility']))
        return tracks

    @property
    def groundtruth_frame(self):
        if self._groundtruth_frame is not None:
            return self._groundtruth_frame
        gt_path = os.path.join(self.gt_path)
        if not os.path.exists(gt_path):
            self.have_gt = False
            return None
        self._groundtruth_frame = load_motchallenge(gt_path, min_confidence=-1.2).sort_index(level=['FrameId', 'Id'])
        if self.valid_indices is not None:
            self._groundtruth_frame = self._groundtruth_frame.loc[self.valid_indices]
        return self._groundtruth_frame

    def groundtruth(self, index):
        if index in self._groundtruth:
            return self._groundtruth[index]
        self._groundtruth[index] = self.gt_frame_to_tracks(self.groundtruth_frame, index)
        return self._groundtruth[index]

    def max_track_id(self):
        return np.stack(self._groundtruth_frame.index.values)[:, 1].max()

    def load_detections_frame(self):
        detection_file = os.path.join(self.path, 'det', self.detections_filename + '.txt')
        return load_motchallenge(detection_file, min_confidence=-1.2)

    @property
    def detections_frame(self):
        if self._detections_frame is not None:
            return self._detections_frame
        self._detections_frame = self.load_detections_frame()
        return self._detections_frame

    def detections(self, index):
        if index in self._detections:
            return self._detections[index]
        frame_detections = []
        seq_info = self.info
        if self.detections_filename in det_filename_to_size:
            h, w = det_filename_to_size[self.detections_filename]
        else:
            h, w = seq_info['imHeight'], seq_info['imWidth']
        frame_detections_raw = self.detections_frame.loc[index] if index in self.detections_frame.index else \
            pd.DataFrame(columns=['Confidence', 'X', 'Y', 'Width', 'Height'])
        for i, detection in frame_detections_raw.iterrows():
            frame_detections.append(Detection(1, detection['Confidence'], 0, detection['X'], detection['Y'],
                                              detection['X'] + detection['Width'],
                                              detection['Y'] + detection['Height'], img_size=(h, w)))
        self._detections[index] = frame_detections
        return frame_detections

    def __iter__(self):
        self.iter_index = 1
        return self

    def __next__(self):
        if self.iter_index > self.__len__():
            raise StopIteration
        idx = self.iter_index
        self.iter_index += 1
        return self.__getitem__(idx)

    def __len__(self):
        return self.info["seqLength"] if self.valid_indices is None else len(self.valid_indices)

    def __getitem__(self, index):
        if self.valid_indices is None:
            i = index
        else:
            i = self.valid_indices[index - 1]
        return self.detections(i), self.groundtruth(i) if self.have_gt else None

    @property
    def info(self):
        if self._info is not None:
            return self._info
        info_filename = os.path.join(self.path, "seqinfo.ini")
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)
        info_dict['update_s'] = 1 / int(info_dict["frameRate"])
        info_dict['imHeight'] = int(info_dict['imHeight'])
        info_dict['imWidth'] = int(info_dict['imWidth'])
        info_dict['seqLength'] = int(info_dict['seqLength'])
        self._info = info_dict
        return info_dict


class SequenceReader:
    def __init__(self, sequence_searches, gt_path=None, **kwargs):
        if isinstance(sequence_searches, str):
            sequence_searches = [sequence_searches]
        self.sequences = []
        self.iter_index = 0
        for s in sequence_searches:
            sequence_paths = sorted(glob.glob(s))

            if len(sequence_paths) == 0:
                raise ValueError('Found no sequences')
            if gt_path is None:
                self.sequences += [FrameIterator(f, **kwargs) for f in sequence_paths]
            else:
                sequence_names = [os.path.basename(path) for path in sequence_paths]
                gt_path = [os.path.abspath(os.path.join(gt_path, sequence_name + '.txt')) for sequence_name in
                           sequence_names]
                self.sequences += [FrameIterator(f, gt_path=gt, **kwargs) for f, gt in zip(sequence_paths, gt_path)]

    def __iter__(self):
        self.iter_index = 0
        return self

    def __len__(self):
        return len(self.sequences)

    def __next__(self):
        if self.iter_index >= self.__len__():
            raise StopIteration
        ind = self.iter_index
        self.iter_index += 1
        return self[ind]

    def __getitem__(self, item):
        return self.sequences[item], self.sequences[item].info
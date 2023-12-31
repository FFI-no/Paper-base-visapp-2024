import numpy as np
import motmetrics as mm

import trackeval
from sequence_helper import load_motchallenge
from utils import to_numpy
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval.datasets.mot_challenge_2d_box import get_preprocessed_seq_data

metric_config = {'PRINT_CONFIG': False}


class MOTSequenceResult:
    def __init__(self, seq_name):
        self.confidences = []
        self.ids = []
        self.classes = []
        self.dets = []
        self.frame_ids = []
        self.visibility = []
        self.num_timesteps = 0
        self.seq = seq_name
        self.seq_name = seq_name
        self._similarity_scores = None

    def from_pd_dataframe(self, df, valid_frame_ids=None):
        if isinstance(df, str):
            df = load_motchallenge(df, min_confidence=-1.2)
        self.frame_ids = df.index.get_level_values("FrameId").unique().sort_values()

        if valid_frame_ids is not None:
            self.frame_ids = self.frame_ids[valid_frame_ids - 1]

        for i, frame_id in enumerate(self.frame_ids):
            frame_df = df.loc[frame_id]
            self.confidences.append(frame_df['Confidence'].values)
            labels = frame_df['ClassId'].values
            labels[labels == -1] = 1
            self.classes.append(labels)
            self.dets.append(frame_df[['X', 'Y', 'Width', 'Height']].values)
            self.visibility.append(frame_df['Visibility'].values)
            self.ids.append(frame_df.index.values)
            self.num_timesteps += 1
        return self

    def add_frame_data(self, frame_id, np_confidences, np_track_ids, np_labels, np_tlwhs):
        self.confidences.append(np_confidences.astype(float))
        self.classes.append(np_labels.astype(int))
        self.ids.append(np_track_ids.astype(int))

        if np_tlwhs.shape[0] > 0:
            self.dets.append(np.stack(np_tlwhs))
        else:
            self.dets.append(np.array([]).reshape((-1, 4)))

        self.frame_ids.append(frame_id)
        self.num_timesteps += 1
        self._similarity_scores = None

    def add_frame(self, frame_id, detections, hw):
        np_detections = to_numpy(detections, hw).reshape((-1, 4))
        self.confidences.append(np_detections[:, 0].astype(float))
        self.classes.append(np_detections[:, 2].astype(int))
        self.ids.append(np_detections[:, 1].astype(int))
        if np_detections[:, 3].shape[0] > 0:
            self.dets.append(np.stack(np_detections[:, 3]))
        else:
            self.dets.append(np.array([]).reshape((-1, 4)))
        self.frame_ids.append(frame_id)
        self.num_timesteps += 1
        self._similarity_scores = None

    def get_dict(self, prepend='tracker'):
        return {
            prepend + "_" + "confidences": self.confidences,
            prepend + "_" + "dets": self.dets,
            prepend + "_" + "ids": self.ids,
            prepend + "_" + "classes": self.classes,
            prepend + "_" + "visibility": self.visibility,
            prepend + "_frame_ids": self.frame_ids,
            "seq": self.seq_name,
            "num_timesteps": self.num_timesteps
        }


class SequenceEvaluator:
    all_metrics = (trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE)
    name_to_metric = {m.get_name(): m(metric_config) for m in all_metrics}

    def __init__(self, name, metric_names=('CLEAR',)):
        self.name = name
        self.metrics = [SequenceEvaluator.name_to_metric[n] for n in metric_names]

        self.raw_data = {}
        self._similarity_scores = []
        self.metric_names = metric_names
        self.need_calculation = True
        self._seq_res = {}

    def add_gt(self, sequence_results):
        self.raw_data.update(sequence_results.get_dict("gt"))
        self.need_calculation = True

    def add_results(self, sequence_results):
        self.raw_data.update(sequence_results.get_dict("tracker"))
        self.need_calculation = True

    def add_missing_rows(self):
        gt_length = len(self.raw_data['gt_frame_ids'])
        gt_frame_ids = self.raw_data['gt_frame_ids'].to_list()
        for i, t_frame_id in enumerate(self.raw_data['tracker_frame_ids']):
            frame_i = gt_frame_ids[i] if i < gt_length else gt_length
            if frame_i > t_frame_id:
                for j in range(frame_i - t_frame_id):
                    self.raw_data['gt_dets'].insert(i, np.array([]).reshape((0, 4)))
                    self.raw_data['gt_ids'].insert(i, np.array([], dtype=int))
                    self.raw_data['gt_classes'].insert(i, np.array([], dtype=int))
                    self.raw_data['gt_confidences'].insert(i, np.array([], dtype=int))
                    gt_frame_ids.insert(i + j, t_frame_id + j)
                gt_length = len(gt_frame_ids)
            gt_dets = self.raw_data['gt_dets'][i]
            track_dets = self.raw_data['tracker_dets'][i]
            ious = _BaseDataset._calculate_box_ious(gt_dets, track_dets)
            self._similarity_scores.append(ious)
        self.gt_frame_ids = np.array(gt_frame_ids)

    @property
    def similarity_scores(self):
        if len(self._similarity_scores) == self.raw_data['num_timesteps']:
            return self._similarity_scores
        assert len(self._similarity_scores) < self.raw_data['num_timesteps']
        for i in range(len(self._similarity_scores), self.raw_data['num_timesteps']):
            frame_i = self.raw_data['gt_frame_ids'][i] - 1
            gt_dets = self.raw_data['gt_dets'][i]
            track_dets = self.raw_data['tracker_dets'][frame_i]
            ious = _BaseDataset._calculate_box_ious(gt_dets, track_dets)
            self._similarity_scores.append(ious)
        return self._similarity_scores

    @property
    def sequence_results(self):
        if self.need_calculation:
            self.run_evaluation()
        return self._seq_res

    def run_evaluation(self):
        self.add_missing_rows()
        self.raw_data['similarity_scores'] = self.similarity_scores
        data = get_preprocessed_seq_data(self.raw_data, 'pedestrian')
        seq_res = {}
        for metric in self.metrics:
            seq_res.update(metric.eval_sequence(data))
        self._seq_res = seq_res
        self.need_calculation = False
        return seq_res


def get_combined_results(sequence_evaluators: [SequenceEvaluator]):
    res_combined = {'COMBINED_SEQ': {}}
    for metric in sequence_evaluators[0].metrics:
        curr_res = {sequence_evaluator.name: sequence_evaluator.sequence_results for sequence_evaluator in
                    sequence_evaluators}
        res_combined.update(curr_res)
        res_combined['COMBINED_SEQ'].update(metric.combine_sequences(curr_res))
    return res_combined


def print_sequence_results(sequence_evaluators):
    sequence_results = get_combined_results(sequence_evaluators)
    for metric in sequence_evaluators[0].metrics:
        metric.print_table(sequence_results, "Tracker",
                           "pedestrian")

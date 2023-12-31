import os
import time

import numpy as np

import torch

from sequence_reader import SequenceReader
from utils import write_results, read_vo
from interpolation import dti_sequence_result
from mot_accumulator import MOTSequenceResult, SequenceEvaluator, print_sequence_results
from mot_tracker import create_tracker, get_parameters
from pathlib import Path


def run_stuff():
    get_metrics = True
    vo = read_vo('./data/MOT20/vo.json')
    write_to_file = True

    results_path = './data/results/mot20_base'

    if write_to_file:
        Path(results_path).mkdir(parents=True, exist_ok=True)

    reader = SequenceReader(
        [
            #'./data/MOT20/train/MOT20-*',
            './data/MOT20/test/MOT20-*',
        ],
        detections_filename='base_det',
    )
    sequence_results = []
    detection_size = np.array((1080, 1920))  # tracker.detection_size
    parameters = get_parameters('mot20_')
    tracker = create_tracker(parameters, detection_size)

    names = []
    for (seq, info) in reader:
        valid_indices = seq.valid_indices
        print(f"Processing sequence {info['name']}")
        name = info['name']
        seq_vo = vo[name]
        names.append(name)
        sequence_result = MOTSequenceResult(name)
        sequence_evaluator = SequenceEvaluator(name, metric_names=('CLEAR', 'Identity'))

        if seq.groundtruth_frame is None:
            get_metrics = False
            print('Found no ground truth, running without metrics')

        if get_metrics:
            sequence_evaluator.add_gt(MOTSequenceResult(name).from_pd_dataframe(seq.gt_path, valid_indices))

        tracker.reset(seq_vo=seq_vo)

        w, h = info['imWidth'], info['imHeight']
        update_s = info['update_s']

        begin = time.time()

        for seq_frame_id, (detections_unfiltered, gt) in zip(valid_indices,
                                                             seq) if valid_indices is not None else enumerate(
            seq,
            start=1):
            detections = []
            for detection in detections_unfiltered:
                if detection.score >= parameters['filter_detection_threshold']:
                    detections.append(detection)

            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = tracker.track(
                frame_id=seq_frame_id,
                detections=detections,
                detection_size=detection_size,
                in_size_hw=(h, w),
                update_s=update_s
            )

            if not frame_id:
                continue
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        while data := tracker.track_from_buffer((h, w)):
            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = data
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        end = time.time()
        print(f'took {(end - begin) * 1e3:.2f}ms')

        if parameters['interpolate']:
            sequence_result = dti_sequence_result(sequence_result, seq_vo, (h, w))

        if get_metrics:
            sequence_evaluator.add_results(sequence_result)
            sequence_results.append(sequence_evaluator)
            print(
                f'MOTA: {sequence_evaluator.sequence_results["MOTA"]}'
                f' IDF1: {sequence_evaluator.sequence_results["IDF1"]}'
            )
        if write_to_file:
            result_filename = os.path.join(results_path, '{}.txt'.format(name))
            write_results(result_filename, sequence_result)
    if get_metrics:
        print_sequence_results(sequence_results)
    print('done')


if __name__ == '__main__':
    with torch.no_grad():
        run_stuff()

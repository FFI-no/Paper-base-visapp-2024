import pandas as pd


def load_motchallenge(fname, **kwargs):
    sep = kwargs.pop('sep', r'\s+|\t+|,')
    min_confidence = kwargs.pop('min_confidence', -1)
    df = pd.read_csv(
        fname,
        sep=sep,
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )

    del df['unused']
    return df[df['Confidence'] >= min_confidence]


class SequenceIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 1
        self.current_sequence_name = None

    def __iter__(self):
        self.index = 1
        img, _, info_imgs, ids = self.dataset[self.index]
        self.current_sequence_name = info_imgs[-1].split('/')[0]
        return self

    def sequence_iterator(self):
        for i in range(self.index, len(self.dataset)):
            img, _, info_imgs, ids = self.dataset[self.index]
            sequence_name = info_imgs[-1].split('/')[0]
            if sequence_name != self.current_sequence_name:
                self.current_sequence_name = sequence_name
                break
            yield self.dataset[self.index]
            self.index += 1

    def __next__(self):
        if self.index > len(self.dataset):
            raise StopIteration
        return self.sequence_iterator

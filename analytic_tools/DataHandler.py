from datetime import datetime

import pandas as pd
# noinspection PyUnresolvedReferences
import plotly.express as px
# noinspection PyUnresolvedReferences
import plotly.graph_objects as go

from configuration.Configuration import Configuration
from configuration.Enums import DatasetPart as DPE
from stgcn.Dataset import Dataset


class DataHandler:

    def __init__(self, config: Configuration, default_range=None):
        self.config = config
        self.dataset = Dataset(config)
        self.dataset.load()
        self.default_range = range(0, len(self.dataset.feature_names) // 5) if default_range is None else default_range

        self.colors = ['blue', 'orange', 'purple', 'cyan', 'red', 'green', 'black', ]
        self.stream_labels = self.dataset.get_feature_names()
        self.freq_str = str(config.resample_frequency) + 'ms'

    def prepare_example(self, part: DPE, requested_label, example_index, sampling_factor):
        indices_with_label = self.dataset.get_part(part).get_indices_with_label(requested_label)

        # TODO Better handling
        if example_index >= len(indices_with_label):
            selected_index = indices_with_label[0]
            out_of_range = True
        else:
            selected_index = indices_with_label[example_index]
            out_of_range = False

        example = self.dataset.get_part(part).get_x()[selected_index]
        label = self.dataset.get_part(part).get_y_strings()[selected_index]
        example_start_time = self.dataset.get_part(part).get_start_time(selected_index)
        assert label == requested_label, 'Error in example selection!'

        # Temp fix for old window time format
        if not '.' in example_start_time:
            example_start_time += '.0000'

        example_start_time = datetime.strptime(example_start_time, self.config.dt_format)

        # Construct a dataframe for this example
        example_time_range = pd.date_range(start=example_start_time, freq=self.freq_str, periods=example.shape[0])
        example_df = pd.DataFrame(example, columns=self.stream_labels, index=example_time_range)

        # reduce to each sampling_factor th row
        example_df = example_df.iloc[::sampling_factor, :]

        return example_df, selected_index, out_of_range

    def get_traces(self, df, displayed_stream_indices: [int]):

        traces = []

        displayed_stream_indices = sorted(displayed_stream_indices)

        if len(displayed_stream_indices) == 0:
            streams = self.stream_labels[self.default_range]
            indices = [i for i in self.default_range]
            print('No indices passed.')
        elif any(len(self.stream_labels) <= index or index < 0 for index in displayed_stream_indices):
            streams = self.stream_labels[self.default_range]
            indices = [i for i in self.default_range]
            print('Requested indices are out of range.')
        else:
            streams = self.stream_labels[displayed_stream_indices]
            indices = displayed_stream_indices

        for index, (label_index, label) in enumerate(zip(indices, streams)):
            color = self.colors[index % len(self.colors)]
            label_with_index = str(label_index) + ' ' + label

            trace = go.Scatter(x=df.index, y=df[label], name=label_with_index, text=label_with_index,
                               marker={'color': color},
                               hoverinfo="y+text")
            traces.append(trace)

        return traces

    @staticmethod
    def ranges_string_to_indices(ranges):
        stream_ranges = []

        try:
            parts = ranges.split(',')
            for part_str in parts:
                range_parts = []
                for x in part_str.strip().split(':'):
                    if x == '':
                        range_parts.append(0)
                    else:
                        range_parts.append(int(x))
                stream_ranges.append(range(*range_parts))
        except Exception:
            stream_ranges = []

        indices = []
        for r in stream_ranges:
            indices.extend([x for x in r])

        indices = list(set(indices))

        return indices

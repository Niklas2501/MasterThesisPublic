import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, classification_report, fbeta_score

from configuration.Enums import DatasetPart
from stgcn.Dataset import Dataset


class Evaluator:

    def __init__(self, dataset, part_tested: DatasetPart):
        self.dataset: Dataset = dataset
        self.part_tested = self.dataset.get_part(part_tested)

        self.tested_indices = []
        self.y_true = []
        self.y_pred = []

        self.results = pd.DataFrame()

    def get_nbr_examples_tested(self):
        return len(self.y_pred)

    def get_f2_scores(self):
        args = {'y_true': self.y_true, 'y_pred': self.y_pred, 'labels': self.dataset.unique_labels_overall, 'beta': 2,
                'average': None, 'zero_division': 0}

        f2_scores = pd.Series(data=list(fbeta_score(**args)), index=self.dataset.unique_labels_overall)

        for average in ['micro', 'macro', 'weighted']:
            args['average'] = average
            f2_avg = fbeta_score(**args)
            f2_scores[average + ' avg'] = f2_avg

        return f2_scores

    def calculate_results(self):
        all_labels = self.dataset.unique_labels_overall
        col_mapping = {'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1', 'support': 'Count',
                       'index': 'Label'}

        mcm = multilabel_confusion_matrix(y_true=self.y_true, y_pred=self.y_pred, labels=all_labels)
        tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]

        mcm = pd.DataFrame({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}, index=all_labels)

        report = classification_report(y_true=self.y_true, y_pred=self.y_pred, labels=all_labels,
                                       output_dict=True, zero_division=0)
        report = pd.DataFrame.from_dict(report).T

        results = mcm
        results = results.join(report, how='outer')
        results.loc[:, 'F2'] = self.get_f2_scores()
        results = results.loc[list(report.index)].reset_index()
        results = results.rename(columns=col_mapping)
        results = results.set_index('Label')
        results = results[['Count', 'TP', 'TN', 'FP', 'FN', 'F1', 'F2', 'Recall', 'Precision']]

        self.results = results

    def get_results(self):
        self.calculate_results()
        return self.results.copy(deep=True)

    def add_prediction_batch(self, tested_indices, predictions):
        assert len(tested_indices) == len(predictions), 'Number of examples tested does not match predictions.'

        self.tested_indices.extend(tested_indices)
        self.y_true.extend(self.part_tested.get_y_strings()[tested_indices])
        self.y_pred.extend(predictions)

    def add_single_prediction(self, tested_example_index, prediction):
        self.tested_indices.append(tested_example_index)

        self.y_true.append(self.part_tested.get_y_strings()[tested_example_index])
        self.y_pred.append(prediction)

    def print_results(self, elapsed_time):
        nbr_examples_tested = len(self.y_true)
        self.calculate_results()
        spacer = '------------------------------------------------------------------' * 2

        print(spacer)
        print('Final Result:')
        print(spacer)
        print('General information:')
        print('Elapsed time:', round(elapsed_time, 6), 'Seconds')
        print('Average time per example:', round(elapsed_time / nbr_examples_tested, 6), 'Seconds')
        print(spacer)
        print(self.results.round(5).to_string())
        print(spacer, '\n')

    def store_prediction_matrix(self, model_path):

        classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        data = np.zeros(shape=(classes.size, classes.size))

        df = pd.DataFrame(index=classes, columns=classes, data=data)

        for true, pred in zip(self.y_true, self.y_pred):
            df.loc[true, pred] += 1

        model_path = model_path if model_path.endswith('/') else model_path + '/'
        df.to_excel(f"{model_path}prediction_matrix.xlsx")

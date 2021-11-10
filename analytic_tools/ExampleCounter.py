import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from stgcn.Dataset import Dataset
from configuration.Configuration import Configuration
from configuration.Enums import DatasetPart


def main():
    """
    Displays the example distribution of the individual dataset parts.
    """

    # Ensure everything is printed.
    pd.set_option('display.max_colwidth', None)
    config = Configuration()
    dataset = Dataset(config)
    dataset.load()

    y_train = dataset.get_train().get_y_strings()
    y_test = dataset.get_test().get_y_strings()

    # Get the unique classes and the number of examples in each for the dataset parts.
    y_train_single, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_single, y_test_counts = np.unique(y_test, return_counts=True)

    # Create dataframes for each part and merge them together.
    x = np.stack((y_train_single, y_train_counts)).transpose()
    x = pd.DataFrame.from_records(x)

    y = np.stack((y_test_single, y_test_counts)).transpose()
    y = pd.DataFrame.from_records(y)

    x = x.merge(y, how='outer', on=0)
    x = x.rename(index=str, columns={0: 'Failure mode', '1_x': 'Train', '1_y': 'Test'})

    # Convert column types in order to be able to sum the values.
    x['Train'] = pd.to_numeric(x['Train']).fillna(value=0).astype(int)
    x['Test'] = pd.to_numeric(x['Test']).fillna(value=0).astype(int)
    x['Total'] = x[['Test', 'Train']].sum(axis=1)

    # Repeat the same steps for the test validation part if one exists.
    if dataset.has_test_val():
        y_val = dataset.get_part(DatasetPart.TEST_VAL).get_y_strings()
        y_val_single, y_val_counts = np.unique(y_val, return_counts=True)
        v = np.stack((y_val_single, y_val_counts)).transpose()
        v = pd.DataFrame.from_records(v, columns=['Failure mode', 'Test Val'])

        x = x.merge(v, how='outer', on='Failure mode')
        x['Test Val'] = pd.to_numeric(x['Test Val']).fillna(value=0).astype(int)
    else:
        x['Test Val'] = 0

    # Repeat the same steps for the train validation part if one exists.
    if dataset.has_train_val():
        y_val = dataset.get_part(DatasetPart.TRAIN_VAL).get_y_strings()
        y_val_single, y_val_counts = np.unique(y_val, return_counts=True)
        v = np.stack((y_val_single, y_val_counts)).transpose()
        v = pd.DataFrame.from_records(v, columns=['Failure mode', 'Train Val'])

        x = x.merge(v, how='outer', on='Failure mode')
        x['Train Val'] = pd.to_numeric(x['Train Val']).fillna(value=0).astype(int)
    else:
        x['Train Val'] = 0

    x['Total'] = x[['Train', 'Test', 'Train Val', 'Test Val']].sum(axis=1)
    x = x[['Failure mode', 'Train', 'Train Val', 'Test', 'Test Val', 'Total']]
    x = x.set_index('Failure mode')

    # Sort rows by class label
    x = x.sort_index(key=lambda s: s.str.lower())

    # Print the information.
    spacer = '----------------------------------------------' * 2
    print(spacer)
    print('Train and test data sets:')
    print(spacer)
    print(x.to_string())
    print(spacer)
    print('Total sum in train:', x['Train'].sum(axis=0))
    print('Total sum in train validation:', x['Train Val'].sum(axis=0))
    print('Total sum in test:', x['Test'].sum(axis=0))
    print('Total sum in test validation:', x['Test Val'].sum(axis=0))
    print('Total sum examples:', x['Total'].sum(axis=0))
    print(spacer)

    # Print as latex table source code.
    # print(x.to_latex(label ='tab:dataset'))

    # Export a list of features and labels in the dataset in a json compatible formatting.
    # import json
    # print(json.dumps({'labels': list(x.index.values),
    #                   'features':list(dataset.feature_names)},
    #                  sort_keys=False, indent=4))


if __name__ == '__main__':
    main()

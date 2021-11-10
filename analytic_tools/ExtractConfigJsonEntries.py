import json

import pandas as pd

from configuration.Configuration import Configuration


def main():
    """
    Can be used to extract the entries for the config.json regarding the feature types and which features are used
    from an excel sheet.
    """

    config = Configuration()
    file = 'feature_selection_process.xlsx'
    primary_feature_col = 'F1-Final-Feature-Selection'
    print_only_primary = True

    feature_selection = pd.read_excel(config.get_additional_data_path(file))

    # Check that a data type is defined for each feature.
    if feature_selection['type'].isnull().sum() > 0:
        raise Exception('Missing value in type column')

    # Check that for each primary feature a selection was made whether it should be included or not.
    if feature_selection[primary_feature_col].isnull().sum() > 0:
        raise Exception('Missing value in primary features')

    # Check that for each secondary feature a selection was made whether it should be included or not.
    # if feature_selection['secondary_features'].isnull().sum() > 0:
    #     raise Exception('Missing value in secondary features')

    # There is no differentiation between boolean and categorical values in the preprocessing process.
    feature_selection['type'] = feature_selection['type'].replace({'booleanValues': 'categoricalValues'})

    # Get the list of features that matches the condition, e.g. that are of type integer.
    primary_streams = list(feature_selection.loc[feature_selection[primary_feature_col] == 1.0, 'streams'])
    secondary_streams = list(feature_selection.loc[feature_selection['secondary_features'] == 1.0, 'streams'])
    integer_streams = list(feature_selection.loc[feature_selection['type'] == 'integerValue', 'streams'])
    float_streams = list(feature_selection.loc[feature_selection['type'] == 'realValues', 'streams'])
    cat_streams = list(feature_selection.loc[feature_selection['type'] == 'categoricalValues', 'streams'])

    # Add the parts that should be output to a dictionary.
    config_json_entries = {}
    config_json_entries['primary_features'] = primary_streams
    if not print_only_primary:
        config_json_entries['integer_features'] = integer_streams
        config_json_entries['float_features'] = float_streams
        config_json_entries['categorical_features'] = cat_streams
        config_json_entries['secondary_features'] = secondary_streams

    # Before printing the output convert it to a json object such that it can be directly copied to the config.json.
    print(json.dumps(config_json_entries, sort_keys=False, indent=2))

    print()
    # Print which files was used in order to avoid wrong configurations.
    print('For file:', file)


if __name__ == '__main__':
    main()

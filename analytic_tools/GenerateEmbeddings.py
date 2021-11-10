import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def generate_embeddings(config: Configuration, feature_names, daemon=True, temp_id=None):
    """
    Generates a numpy file that includes the node embeddings matching the indices of features in feature_names.

    :param config:
    :param feature_names:
    :param daemon: Whether the methode is invoked manually (False) or automatically (True), primarily by the
        subsequent feature reduction method of the dataset class.
    :param temp_id: If daemon is true, the generated file will be stored in a temporary directory using this id.
    """

    if daemon and temp_id is None:
        raise ValueError('If running this as a daemon service a temporary id must be passed.')

    # Load the overall embedding file from disk and convert to a processable data frame.
    # TODO error_bad_lines=False, warn_bad_lines=False ->  on_bad_lines='skip' when pandas version was updated
    embedding_df = pd.read_csv(config.get_additional_data_path('embeddings.tsv'), sep=' ', skiprows=1, header=None,
                               error_bad_lines=False, warn_bad_lines=False, index_col=0)

    # Reduce rows for fast search and avoidance of errors cause by space being used as separator ... why? ...
    embedding_df = embedding_df[embedding_df.index.str.startswith('http://iot.uni-trier.de/FTOnto#')]

    for col in embedding_df:
        embedding_df[col] = pd.to_numeric(embedding_df[col])

    # Import the mapping created using Onto2Matrix.py as a secondary output when generating the predefined matrix.
    with open(config.get_additional_data_path('feature_2_iri.json'), 'r') as f:
        feature_2_iri: dict = json.load(f)

        for feature, iri in feature_2_iri.items():

            # Check if entries in mapping are empty.
            if iri is None:
                raise ValueError(f'No IRI defined for feature {feature}')
            else:
                # Replace base_uri with the one used in the embedding file
                feature_2_iri[feature] = iri.replace('FTOnto:', 'http://iot.uni-trier.de/FTOnto#')

    # Check if the mapping contains an entry for each feature, different from the check above.
    for feature in feature_names:
        if feature not in feature_2_iri.keys():
            raise ValueError(f'No IRI mapping found for feature {feature}')

    # Extract the embedding vectors for the required features and store in a dictionary.
    feature_2_emb_vec = {}
    for feature, iri in feature_2_iri.items():
        emb_vec = embedding_df.loc[iri].values
        feature_2_emb_vec[feature] = emb_vec

    # Generate a array with the embedding vectors being on located on the index matching its feature.
    emb_array = np.zeros(shape=(len(feature_names), len(feature_2_emb_vec.get(feature_names[0]))), dtype='float32')
    for index, feature in enumerate(feature_names):
        emb_array[index, :] = feature_2_emb_vec.get(feature)

    if daemon:
        config.embeddings_file = f'/temp/embeddings_{temp_id}.npy'

        if not os.path.exists(config.get_additional_data_path('temp/')):
            os.makedirs(config.get_additional_data_path('temp/'))

    np.save(config.get_additional_data_path(config.embeddings_file), emb_array)


if __name__ == '__main__':
    config = Configuration()
    feature_names = np.load(config.get_training_data_path() + 'feature_names.npy')
    generate_embeddings(config, feature_names, daemon=False)
    print('Generation of embedding file finished.')

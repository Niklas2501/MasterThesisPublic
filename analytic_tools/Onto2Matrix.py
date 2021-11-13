import itertools
import json
import os
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import owlready2 as owl
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from stgcn.Dataset import Dataset
from configuration.Configuration import Configuration


def discrete_cmap(N):
    color_list = ['#808080', '#FF0000', '#FFFF00', '#00FF00', '#008000', '#00FFFF', '#000080', '#FF00FF', '#800000',
                  '#008080', '#0000FF', '#800080', '#DFFF00', '#FFBF00', '#FF7F50', '#DE3163', '#40E0D0', '#CCCCFF',
                  '#8e44ad']

    # Reduce the list to the number of colors needed.
    if N < len(color_list):
        color_list = color_list[0:N]
    # Convert to a hex color object required by the ListedColormap.
    color_list = [mlp.colors.hex2color(c) for c in color_list]
    return ListedColormap(color_list, name='OrangeBlue')


def plot(feature_names, linked_features, responsible_relations, force_self_loops, display_labels):
    # Recreate the ADJ but with different values = colors based on the relationship of the connection
    n = feature_names.size
    a_plot = pd.DataFrame(index=feature_names, columns=feature_names, data=np.zeros(shape=(n, n)))

    color_values = {
        'no_relation': 0,
        'self_loops': 1,
        'component': 2,
        'same_iri': 3,
        'connection': 4,
        'actuation': 5,
        'calibration': 6,
        'precondition': 7,
        'postcondition': 8,
    }

    if force_self_loops:

        for f_j in feature_names:
            a_plot.loc[f_j, f_j] = color_values['self_loops']

    for (f_j, f_i), r in zip(linked_features, responsible_relations):
        c_val = color_values[r]

        if c_val > a_plot.loc[f_i, f_j]:
            a_plot.loc[f_i, f_j] = c_val

    size = 22 if display_labels else 15
    dpi = 200 if display_labels else 200

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=dpi)
    im = ax.imshow(a_plot.values, cmap=discrete_cmap(len(list(color_values.keys()))), )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    im.set_clim(vmin=-0.5, vmax=len(list(color_values.keys())) - 0.5)
    ax.set_title(color_values)

    ax.set_ylabel('i (target)')
    ax.set_xlabel('j (source)')

    # x.set_ylabel('i (Zielknoten)')
    # ax.set_xlabel('j (Ausgangsknoten)')

    ax.tick_params(which='minor', width=0, color='white')
    ax.set_xticks(np.arange(-.5, n, 10), minor=True)
    ax.set_yticks(np.arange(-.5, n, 10), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

    if display_labels:
        # Minor ticks with width = 0 so they are not really visible
        ax.set_xticks(np.arange(0, n, 1), minor=False)
        ax.set_yticks(np.arange(0, n, 1), minor=False)

        features = [f[0:20] if len(f) > 20 else f for f in a_plot.columns]

        ax.set_xticklabels(features, minor=False)
        ax.set_yticklabels(features, minor=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(f"../logs/{config.a_pre_file.split('.')[0]}.pdf", dpi=dpi, bbox_inches='tight')

    plt.show()


# Split string at pos th occurrence of sep
def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def name_2_iri(feature_names: np.ndarray):
    """
    Creates a mapping from feature names to the iris in the ontology.
    :param feature_names: A numpy array with features names that matches the order of features in the dataset.
    :return: A dictionary that matches the feature names to their iri.
    """

    base_iri = 'FTOnto:'
    feature_2_iri = {}
    manual_corrections = {
        base_iri + "OV_1_Compressor_8": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "WT_1_Compressor_8": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "OV_2_Compressor_8": base_iri + "OV_2_WT_2_Compressor_8",
        base_iri + "WT_2_Compressor_8": base_iri + "OV_2_WT_2_Compressor_8",
        base_iri + "MM_1_Pneumatic_System_Pressure": base_iri + "MM_1_Compressor_8",
        base_iri + "OV_1_WT_1_Pneumatic_System_Pressure": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "SM_1_Pneumatic_System_Pressure": base_iri + "SM_1_Compressor_8",
        base_iri + "VGR_1_Pneumatic_System_Pressure": base_iri + "VGR_1_Compressor_7",
    }

    for feature in feature_names:

        # Remember whether a matching iri could be found
        matched_defined_type = True
        iri_comps = []

        # Derive iri from feature name
        main_component, specific_part = split(feature, '_', 2)
        iri_comps.append(main_component.upper())
        identifier, type = split(specific_part, '_', 1)

        if main_component.startswith('shop'):
            substring = split(feature, '_', 4)[1].title()
            a, b = split(substring, '_', -3)
            iri_comps = [a.upper(), b]
        elif identifier.startswith('i'):
            # Light barrier and position switches
            nbr = identifier[1]
            type = split(type, '_', 2)[0].title()
            iri_comps.append(type)
            iri_comps.append(nbr)
        elif identifier.startswith('m'):
            # Motor speeds
            nbr = identifier[1]
            iri_comps.append('Motor')
            iri_comps.append(nbr)
        elif identifier.startswith('o'):
            # Valves and compressors
            nbr = identifier[1]
            iri_comps.append(split(type, '_', 1)[0].title())
            iri_comps.append(nbr)
        elif identifier in ['current', 'target']:
            iri_comps.append('Crane_Jib')
        elif identifier == 'temper':
            # Untested because not present in dataset
            iri_comps.append('Temperature')
        elif main_component.startswith('bmx'):
            main_component = split(main_component, '_', 1)[1].upper()
            iri_comps = [main_component, identifier, 'Crane_Jib']
        elif main_component.startswith('acc'):
            # Acceleration sensors are associated with the component they observe e.g. a motor
            main_component = split(main_component, '_', 1)[1].upper()

            if type.startswith('m'):
                iri_comps = [main_component, identifier, 'Motor', type.split('_')[1]]
            elif type.startswith('comp'):
                iri_comps = [main_component, identifier, 'Compressor_8']
        elif main_component == 'sm_1' and identifier == 'detected':
            iri_comps = [main_component.upper(), 'Color', 'Sensor', '2']
        else:
            # No matching iri was found
            matched_defined_type = False

        if matched_defined_type:
            iri = base_iri + '_'.join(iri_comps)

            if iri in manual_corrections.keys():
                iri = manual_corrections.get(iri)

            feature_2_iri[feature] = iri
        else:
            # Mark the no valid iri was found for this feature
            feature_2_iri[feature] = None

    return feature_2_iri


def invert_dict(d: dict):
    """
    Creates an inverted dictionary of d: All values of a key k in d will become a key with value k in the inverted dict.
    :param d: The dictionary that should be inverted.
    :return: The resulting inverted dict.
    """

    inverted_dict = {}

    for key, value in d.items():
        if value in inverted_dict.keys():
            inverted_dict[value].append(key)
        else:
            inverted_dict[value] = [key]

    return inverted_dict


def tuple_corrections(feature_tuples, iri=None):
    # Remove self loops
    feature_tuples = [(a, b) for (a, b) in feature_tuples if a != b]

    # Add inverse, necessary because not all relations are present in the ontology
    feature_tuples.extend([(b, a) for (a, b) in feature_tuples])

    # Remove duplicates
    feature_tuples = list(set(feature_tuples))

    return feature_tuples


def store_mapping(config: Configuration, feature_2_iri: dict):
    """
    Stores the dictionary mapping features to their iri such that it can be used by other programs,
        mainly GenerateFeatureEmbeddings.py
    :param config: The configuration object.
    :param feature_2_iri: The dictionary mapping features to their iri.
    """

    with open(config.get_additional_data_path('feature_2_iri.json'), 'w') as outfile:
        json.dump(feature_2_iri, outfile, sort_keys=True, indent=2)


def check_mapping(feature_2_iri: dict):
    """
    Ensures a iri could be determined for all features.
    :param feature_2_iri: The dictionary mapping features to their iri.
    """

    for feature, iri in feature_2_iri.items():
        if iri is None:
            raise ValueError(f'No IRI could be generated for feature {feature}.'
                             ' This would cause problems when trying to assign an embedding or finding relations.')


def onto_2_matrix(config, feature_names, daemon=True, temp_id=None):
    ##############################

    # Settings which relations to include in the generated adjacency matrix.
    component_of_relation = True
    iri_relation = True
    connected_to_relation = True
    calibration_relation = True
    actuates_relation = True
    both_precondition_same_service = True
    both_postcondition_same_service = True

    # Should not be used. Used the corresponding gsl mod instead.
    force_self_loops = False

    # Settings regarding the adj plot.
    plot_labels = True
    print_linked_features = False
    ##############################

    # Consistency check to ensure the intended configuration is used.
    if not all([component_of_relation, iri_relation, connected_to_relation, calibration_relation, actuates_relation,
                both_postcondition_same_service, both_precondition_same_service, not force_self_loops]):
        if not daemon:
            reply = input('Configuration deviates from the set standard. Continue?\n')
            if reply.lower() != 'y':
                sys.exit(-1)
        else:
            raise ValueError('Configuration deviates from the set standard.')

    if daemon and temp_id is None:
        raise ValueError('If running this as a daemon service a temporary id must be passed.')

    # Create dictionary that matches feature names to the matched iri
    feature_2_iri = name_2_iri(feature_names)

    check_mapping(feature_2_iri)
    store_mapping(config, feature_2_iri)

    # Invert such that iri is matched to list of associated features
    iri_2_features = invert_dict(feature_2_iri)

    onto_file = config.get_additional_data_path('FTOnto.owl')
    ontology = owl.get_ontology("file://" + onto_file).load()

    linked_features = []
    responsible_relations = []

    # Link features which are matched to the same iri
    if iri_relation:
        matched_iri_dict = {}

        for name, iri in feature_2_iri.items():
            if iri is None:
                continue

            if iri in matched_iri_dict:
                matched_iri_dict[iri].append(name)
            else:
                matched_iri_dict[iri] = [name]

        for iri, feature_lists in matched_iri_dict.items():
            feature_tuples = list(itertools.product(feature_lists, repeat=2))
            feature_tuples = tuple_corrections(feature_tuples, iri)
            linked_features.extend(feature_tuples)

            # Assign same relation for plotting
            responsible_relations.extend(['same_iri' for _ in range(len(feature_tuples))])

    if component_of_relation:
        r1 = 'FTOnto:isComponentOf'
        r2 = 'FTOnto:hasComponent'

        feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                           direct_relation=False, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['component' for _ in range(len(feature_tuples))])

    if connected_to_relation:
        r = 'FTOnto:isConnectedTo'
        feature_tuples = infer_connections(feature_2_iri, iri_2_features, r,
                                           direct_relation=True, symmetric_relation=True)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['connection' for _ in range(len(feature_tuples))])

    if calibration_relation:
        r1 = 'FTOnto:calibrates'
        r2 = 'FTOnto:isCalibratedBy'

        feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['calibration' for _ in range(len(feature_tuples))])

    if actuates_relation:
        # Superclasses FTOnto:actuates and FTOnto:isActuatedBy not present
        actuation_relations = [
            ('FTOnto:actuatesHorizontallyForwardBackward', 'FTOnto:isActuatedHorizontallyForwardBackwardBy'),
            ('FTOnto:actuatesHorizontallyLeftRight', 'FTOnto:isActuatedHorizontallyLeftRightBy'),
            ('FTOnto:actuatesRotationallyAroundVerticalAxis', 'FTOnto:isActuatedRotationallyAroundVerticalAxisBy'),
            ('FTOnto:actuatesVertically', 'FTOnto:isActuatedVerticallyBy')
        ]

        for r1, r2 in actuation_relations:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)
            feature_tuples = tuple_corrections(feature_tuples)
            linked_features.extend(feature_tuples)

            # Assign same relation for plotting
            responsible_relations.extend(['actuation' for _ in range(len(feature_tuples))])

    # Load the service pre- and postcondtions from a file.
    with open(config.get_additional_data_path('service_condition_pairs.json'), 'r') as f:
        service_condition_pairs = json.load(f)
        precondition_pairs = service_condition_pairs['precondition_pairs']
        postcondition_pairs = service_condition_pairs['postcondition_pairs']

    if both_precondition_same_service:
        iri_tuples = []

        for key_iri, values in precondition_pairs.items():
            iri_tuples.extend([(key_iri, value_iri) for value_iri in values])

        feature_tuples = feature_tuples_from_iri_tuples(iri_tuples, iri_2_features)
        linked_features.extend(feature_tuples)

        # Assign same relation for plotting
        responsible_relations.extend(['precondition' for _ in range(len(feature_tuples))])

    if both_postcondition_same_service:
        iri_tuples = []

        for key_iri, values in postcondition_pairs.items():
            iri_tuples.extend([(key_iri, value_iri) for value_iri in values])

        feature_tuples = feature_tuples_from_iri_tuples(iri_tuples, iri_2_features)
        linked_features.extend(feature_tuples)

        # Assign same relation for plotting
        responsible_relations.extend(['postcondition' for _ in range(len(feature_tuples))])

    if not daemon and print_linked_features:
        rows = []
        for a, (b, c) in zip(responsible_relations, linked_features):
            rows.append([a, b, feature_2_iri.get(b), c, feature_2_iri.get(c)])

        df = pd.DataFrame(data=rows, columns=['Relation', 'Feature 1', 'IRI Feature 1', 'Feature 2', 'IRI Feature 2'])
        print(df.to_string())

    n = feature_names.size
    a_df = pd.DataFrame(index=feature_names, columns=feature_names, data=np.zeros(shape=(n, n)))

    for f_j, f_i in linked_features:
        if f_i != f_j:
            a_df.loc[f_i, f_j] = 1

    if force_self_loops:
        for f_i in feature_names:
            a_df.loc[f_i, f_i] = 1

    a_df.index.name = 'Features'

    if daemon:
        config.a_pre_file = f'temp/predefined_a_{temp_id}.xlsx'

        if not os.path.exists(config.get_additional_data_path('temp/')):
            os.makedirs(config.get_additional_data_path('temp/'))

        a_df.to_excel(config.get_additional_data_path(config.a_pre_file))
    else:
        a_df.to_excel(config.get_additional_data_path(config.a_pre_file))
        a_analysis(a_df)
        plot(feature_names, linked_features, responsible_relations, force_self_loops, display_labels=plot_labels)
        # plot_for_thesis(feature_names, linked_features, responsible_relations)
        # thesis_output(feature_2_iri, responsible_relations, linked_features)


def thesis_output(feature_2_iri, responsible_relations, linked_features):
    def shorten_feature(feature):
        limit = 35
        return feature[0:limit - 2] + '...' if len(feature) > limit else feature

    def combine_relations(relations):
        rel_2_int = {
            'no_relation': 0, 'self_loops': 1, 'component': 2, 'same_iri': 3, 'connection': 4,
            'actuation': 5, 'calibration': 6, 'precondition': 7, 'postcondition': 8,
        }

        relations = [str(rel_2_int.get(rel)) for rel in sorted(relations)]
        relations = [rel for rel in sorted(relations)]
        return ', '.join(relations)

    features, iris = [], []

    for feature, iri in feature_2_iri.items():
        features.append(feature)
        iris.append(iri)

    features = [shorten_feature(f) for f in features]
    data = np.array([features, iris]).T
    features_2_iri_df = pd.DataFrame(columns=['Datenstrom', 'IRI'], data=data)
    features_2_iri_df.index.name = 'Index'
    # features_2_iri_df = features_2_iri_df.sort_values(by='IRI', ascending=True)
    print(features_2_iri_df.to_latex(longtable=True, label='tab:streams2iri'))

    rows = []
    for a, (b, c) in zip(responsible_relations, linked_features):
        rows.append([a, b, c])

    df = pd.DataFrame(data=rows, columns=[
        'Relation', 'Feature 1', 'Feature 2'])

    df['F1'] = df.apply(lambda x: x['Feature 1'] if x['Feature 1']
                                                    > x['Feature 2'] else x['Feature 2'], axis=1)
    df['F2'] = df.apply(lambda x: x['Feature 1'] if x['Feature 1']
                                                    < x['Feature 2'] else x['Feature 2'], axis=1)
    df = df.sort_values(by=['F1', 'F2'], ascending=False)
    df = df.drop_duplicates(subset=['F1', 'F2', 'Relation'])
    df = df.groupby(['F1', 'F2'])['Relation'].apply(
        combine_relations).reset_index()
    df = df.drop(df.loc[df['Relation'] == 'component'].index).reset_index()

    df['Datenstrom 1'] = df['F1'].apply(shorten_feature)
    df['Datenstrom 2'] = df['F2'].apply(shorten_feature)
    df = df[['Datenstrom 1', 'Relation', 'Datenstrom 2']]

    print(df.to_string())
    # print(df.to_latex(longtable=True, label='tab:relations', index=False))


def feature_tuples_from_iri_tuples(iri_tuples, iri_2_features: dict):
    feature_tuples = []

    # Create all feature pairs for each iri pair (some features are mapped to the same iri)
    for iri_1, iri_2 in iri_tuples:

        if not iri_1 in iri_2_features.keys() or not iri_2 in iri_2_features.keys():
            continue

        features_iri_1 = iri_2_features.get(iri_1)
        features_iri_2 = iri_2_features.get(iri_2)
        pairs = list(itertools.product(features_iri_1, features_iri_2))
        feature_tuples.extend(pairs)

    feature_tuples = tuple_corrections(feature_tuples)

    return feature_tuples


def a_analysis(a_df: pd.DataFrame):
    print('\nFeatures without links:')
    temp = a_df.loc[(a_df == 0).all(axis=1)]
    print(*temp.index.values, sep='\n')
    print()

    # noinspection PyArgumentList
    temp = a_df.sum(axis=0, skipna=True).sort_values(ascending=False)
    print(temp.to_string())


def prepare_query(a, r1, direct_relation, symmetric_relation, r2=None):
    if not direct_relation:
        assert r2 is not None, 'if not direct, a second relation must be passed'

        return "SELECT ?x WHERE {{ " + a + " " + r1 + " ?y . ?x " + r1 + " ?y . } " + \
               "UNION { ?y " + r2 + " " + a + " .  ?y " + r2 + " ?x .}}"

    if symmetric_relation:
        return "SELECT ?x WHERE {{ " + a + " " + r1 + " ?x . } " + \
               " UNION { ?x " + r1 + " " + a + " . }}"
    else:
        assert r2 is not None, 'if not symmetric, a second relation must be passed'
        return "SELECT ?x WHERE {{ " + a + " " + r1 + " ?x . } " + \
               "UNION { " + a + " " + r2 + " ?x . }" + \
               "}"


def infer_connections(feature_2_iri, iri_2_features, r1, direct_relation, symmetric_relation, r2=None):
    tuples = []

    iris = list(set(feature_2_iri.values()))

    for name, iri in feature_2_iri.items():
        if iri is None:
            continue

        q = prepare_query(iri, r1, direct_relation=direct_relation, symmetric_relation=symmetric_relation, r2=r2)
        try:
            results = list(owl.default_world.sparql(q))
        except ValueError:
            results = []
            print(f'Query error for feature {name} with assigned IRI {iri}')

        relevant_results = [str(res[0]).replace('.', ':') for res in results]
        relevant_results = [iri for iri in relevant_results if iri in iris]
        f = []
        for res_iri in relevant_results:
            f.extend(iri_2_features.get(res_iri))
        tuples.extend([(name, res_name) for res_name in f])

    return tuples


if __name__ == '__main__':
    config = Configuration()
    dataset = Dataset(config)
    dataset.load()

    onto_2_matrix(config, dataset.feature_names, daemon=False)

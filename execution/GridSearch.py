import copy
import itertools
import logging
import os
import sys

# suppress debugging messages of TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from contextlib import contextmanager
from time import perf_counter

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from stgcn.STGCN import STGCN
from execution.Evaluator import Evaluator
from configuration.Configuration import Configuration
from configuration.Enums import GridSearchMode
from configuration.Hyperparameter import Hyperparameter
from stgcn.Dataset import Dataset


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_template(config: Configuration):
    hyper = Hyperparameter()
    hyper.load_from_file(config.grid_search_base_templates_folder + config.grid_search_base_template_file)
    return hyper


def remove_grid_search_settings(grid_config: dict, keys_to_remove: list):
    grid_config_hps_only = copy.deepcopy(grid_config)
    for key in keys_to_remove: grid_config_hps_only.pop(key)
    return grid_config_hps_only


def create_hp_objs_from_combinations(template: Hyperparameter, parameter_names, combinations):
    hp_objects = []

    # Create hyperparameter objects for each combinations based on the template.
    for index, comb in enumerate(combinations):
        hp_obj = copy.deepcopy(template)

        for hyperparameter, value in zip(parameter_names, comb):
            # print('Combination: {}, HP: {}, Value: {}'.format(index, hyperparameter, value))
            setattr(hp_obj, hyperparameter, value)

        hp_objects.append(hp_obj)

    return hp_objects


def create_combinations_standard(template: Hyperparameter, grid_config_hps_only: dict):
    parameter_names = []
    values = []

    # Creation of list of varied parameters and list of lists with the respective values.
    for parameter, single_hp_values in grid_config_hps_only.items():
        parameter_names.append(parameter)
        values.append(single_hp_values)

    # Create all possible combinations using the list created before.
    combinations = list(itertools.product(*values))
    hp_objects = create_hp_objs_from_combinations(template, parameter_names, combinations)

    return hp_objects, parameter_names, combinations


def create_combinations_index_pairs(template: Hyperparameter, grid_config_hps_only: dict):
    parameter_names = []
    value_lists = []

    # Creation of list of varied parameters and list of lists with the respective values.
    for parameter, single_hp_values in grid_config_hps_only.items():
        parameter_names.append(parameter)
        value_lists.append(single_hp_values)

    # Ensue all parameters have the same number of values that should be tested, otherwise we can't create
    # the combinations.
    it = iter(value_lists)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Index pair grid search mode requires all altered hyperparameters '
                         'to have the same number of tested values.')

    # Create 2D array with rows = parameters and cols = values for each
    # --> The column vectors contain the combinations that should be tested
    # dtype=object is necessary in order to also store hp values that are itself lists
    combinations = []
    array = np.array(value_lists, dtype=object)

    for col_index in range(array.shape[1]):
        combinations.append(list(array[:, col_index]))

    hp_objects = create_hp_objs_from_combinations(template, parameter_names, combinations)

    return hp_objects, parameter_names, combinations


def check_hp_validity(config, hp_object: Hyperparameter, index=None):
    try:
        hp_object.check_constrains()
    except Exception as ve:
        print()
        print('############################################################################################')
        print('Error during hp check {}:'.format(index + 1 if index is not None else ''))
        print('############################################################################################')
        print('Error message:', ve)
        print('############################################################################################')
        hp_object.print_hyperparameters()
        print('############################################################################################')
        sys.exit(-1)


def modify_config(config: Configuration):
    print('WARNING: The following aspects to ensure correct execution are NOT checked:')
    print('- That unnecessary hyperparameters are set increasing the number of combinations without an effect')
    print('- That the correct type and format is used for a specific hyperparameter')
    print('- For list-type hyperparameters: That all associated hyperparameters match, '
          'e.g. number of cnn layers and kernel sizes')
    if config.tf_memory_growth_enabled:
        print('- Growing memory allocation is enabled.')
    if config.representation != "RAW":
        print(f'- Selected representation (NOT raw time series data): {config.representation}')
    if config.test_failures_only:
        config.test_failures_only = False
        print(f'- Setting for test_failures_only is ignored.')
    # if config.write_vis_out_to_file:
    #     config.write_vis_out_to_file = False
    #     print(f'- Setting for write_vis_out_to_file is ignored.')
    # if config.create_a_plots:
    #     config.create_a_plots = False
    #     print(f'- Setting for generate_plots is ignored.')
    if config.do_not_save_gs_models:
        print(f'- Option to not save grid search models is enabled. This also includes the plot generation.')

    print()


def grid_search():
    config = Configuration()
    spacer = '#############################################################' * 2
    gs_id = np.random.default_rng().integers(100_000, 999_999)

    if config.tf_memory_growth_enabled:
        import tensorflow as tf
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset = Dataset(config)
    dataset.load(print_info=True)

    grid_config: dict = config.grid_search_config
    metrics: list = grid_config.get('metrics')
    metrics = [metric for metric in metrics if metric in ['Precision', 'Recall', 'F1', 'F2']]

    hyper_objects = []

    modify_config(config)

    if config.grid_search_mode == GridSearchMode.TEST_MULTIPLE_JSONS:

        if not 'json_files' in grid_config:
            raise ValueError('Key "json_files" not found in grid configuration file.')

        json_files = grid_config.get('json_files')
        config_rows = [[file.strip('.json')] for file in json_files]
        config_cols = ['File name']

        for file in json_files:
            hyper = Hyperparameter()
            hyper.load_from_file(config.hyper_file_folder + file)
            hyper_objects.append(hyper)

    elif config.grid_search_mode in [GridSearchMode.STANDARD_GRID_SEARCH, GridSearchMode.INDEX_PAIR_GRID_SEARCH]:
        template: Hyperparameter = load_template(config)
        grid_config_hps_only = remove_grid_search_settings(grid_config, ["notes", "metrics", "sorted_by"])

        for hp in grid_config_hps_only.keys():

            # Ensure the key in hyperparameter defined in the json file exists as a hyperparameter.
            if hasattr(template, hp):
                hp_value = getattr(template, hp)
                if hp_value is not None:
                    raise ValueError(f'Value of hyperparameter {hp} is not None in the template file.\n'
                                     'Aborting grid search to ensure correct configurations.')
            else:
                raise ValueError(f'Hyperparameter {hp} from the grid search configuration was not '
                                 'found in the template file, i.e. it is not a valid hyperparameter name')

        if config.grid_search_mode == GridSearchMode.STANDARD_GRID_SEARCH:
            hyper_objects, config_cols, config_rows = create_combinations_standard(template, grid_config_hps_only)
        elif config.grid_search_mode == GridSearchMode.INDEX_PAIR_GRID_SEARCH:
            hyper_objects, config_cols, config_rows = create_combinations_index_pairs(template, grid_config_hps_only)
        else:
            raise ValueError('Grid search mode not implemented:', config.grid_search_mode)
    else:
        raise ValueError('Grid search mode not implemented:', config.grid_search_mode)

    # Check if all hyperparameter combinations are valid.
    for index, hyper in enumerate(hyper_objects):
        check_hp_validity(config, hyper, index)

    nbr_configurations = len(hyper_objects)
    results_list = []
    result_dfs = np.empty((nbr_configurations,), dtype=object)

    print(f'\nTesting {nbr_configurations} combinations via grid search ({config.grid_search_mode}) '
          f'on the {config.part_tested} dataset. \n')

    # nbr_tested = int(len(hyper_objects) * 0.20)
    # indices_test = np.random.default_rng().choice(np.arange(0, len(hyper_objects)), nbr_tested, replace=False)
    # print(json.dumps({'indices_tested': [int(i) for i in indices_test]}))
    #
    # print('-------------------------------------------------------------------------------------')
    # print(f'WARNING: Only {nbr_tested} out of all {len(hyper_objects)} combinations are tested!')
    # print('-------------------------------------------------------------------------------------')

    start = perf_counter()

    for index, (hyper, config_row) in enumerate(zip(hyper_objects, config_rows)):

        # if index not in indices_test:
        #     print(f'\nSkipping combination {index}/{nbr_configurations - 1} ...\n')
        #     continue

        try:
            print(spacer)
            print(f'Currently testing combination {index}/{nbr_configurations - 1} ...')
            print(spacer, '\n')

            stgcn = STGCN(config, dataset, training=True, hyper=hyper)
            stgcn.trainer.set_file_names('gs_{}_model_{}'.format(gs_id, index))
            stgcn.print_model_info()
            selected_model_name = stgcn.trainer.train_model(skip_model_saving=config.do_not_save_gs_models)

            print('Evaluating ...\n')
            with suppress_stdout():
                stgcn.switch_to_testing()
                evaluator: Evaluator = stgcn.test_model(print_results=False, selected_model_name=selected_model_name)

            if config.output_intermediate_results:
                # evaluator.store_prediction_matrix(config.models_folder + selected_model_name)
                evaluator.print_results(0)
                hyper.print_hyperparameters()
                print()

            results_row = list(config_row) + list(evaluator.results.loc['weighted avg', metrics])
            results_list.append(results_row)
            result_dfs[index] = evaluator.get_results()
        except Exception as e:

            # Catch every exception (resulting from an error in during training) except keyboard interrupts, such
            # that the process can still be stopped by the user.
            if isinstance(e, KeyboardInterrupt):
                sys.exit()

            print(spacer)
            print(f'ERROR DURING TESTING OF THIS COMBINATION:')
            print(e)
            print()
            hyper.print_hyperparameters()
            print(spacer)
            print()

    end = perf_counter()

    cols = config_cols + metrics
    results = pd.DataFrame(columns=cols, index=np.arange(0, len(results_list)), data=results_list)
    results.index.name = 'Comb'

    # Higher = better is assumed.
    sorting_cols = list(np.array((metrics))[grid_config.get('sorted_by')])
    results = results.sort_values(by=sorting_cols, ascending=False)

    print(spacer)
    print('Gird Search Results')
    print(spacer, '\n')

    print(f'\n\nBest combinations tested (also stored in gs_{gs_id}_results.xlsx):\n')
    print(results.head(200).round(5).to_string())
    if len(results) > 0:
        best_comb_index = results.index[0]
        print('\n\nFull result output for the best combination:\n')
        print(result_dfs[best_comb_index].round(5).to_string())
        print('\n\nHyperparameter configuration for the best combination:')
        hyper_objects[best_comb_index].print_hyperparameters()
    print('\n\nExecution time: {}'.format(end - start))

    results.round(5).to_excel(f"../logs/gs_{gs_id}_results.xlsx")


if __name__ == '__main__':
    grid_search()

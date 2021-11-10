import copy
import logging
import os
import sys

# suppress debugging messages of TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
import pandas as pd
from time import perf_counter

from stgcn.STGCN import STGCN
from configuration.Configuration import Configuration
from execution.Evaluator import Evaluator
from stgcn.Dataset import Dataset
from configuration.Hyperparameter import Hyperparameter
from execution import GridSearch


def test_config_values():
    STGCN.reset_state()
    config = Configuration()
    GridSearch.modify_config(config)
    spacer = '**************************************************************' * 2
    cvs_id = np.random.default_rng().integers(100_000, 999_999)

    #
    # Configuration
    #
    metrics = ['F1', 'F2', 'Recall', 'Precision']
    config_variable_tested = []
    config_variable_values = [

    ]

    generate_plots = True

    #
    #
    #

    results_list = []
    nbr_configurations = len(config_variable_values)
    result_dfs = np.empty((nbr_configurations,), dtype=object)

    start = perf_counter()
    for index, value in enumerate(config_variable_values):
        STGCN.reset_state()

        subset_config = copy.deepcopy(config)
        subset_config.create_a_plots = generate_plots
        subset_config.a_plot_display_labels = generate_plots

        print(spacer)
        if isinstance(config_variable_tested, str):
            subset_config.__setattr__(config_variable_tested, value)
            desc = str(value)
            print(f'Currently testing: {desc} ...')
        else:
            for var_index, var_name in enumerate(config_variable_tested):
                # noinspection PyTypeChecker
                subset_config.__setattr__(var_name, value[var_index])
            # noinspection PyTypeChecker
            desc = "_".join([str(v) for v in value])
            desc = desc.replace('/', '_').replace('\\', '_').replace('.', '')
            print(f'Currently testing {index}/{nbr_configurations - 1}: {desc}')
        print(spacer, '\n')

        dataset = Dataset(subset_config)
        dataset.load(print_info=False)

        hyper = Hyperparameter()
        hyper.load_from_file(subset_config.hyper_file)

        stgcn = STGCN(subset_config, dataset, training=True, hyper=hyper)
        stgcn.print_model_info()

        stgcn.trainer.set_file_names('cvs_{}_model_{}'.format(cvs_id, index))
        selected_model_name = stgcn.train_model()
        stgcn.switch_to_testing()
        evaluator: Evaluator = stgcn.test_model(print_results=False, selected_model_name=selected_model_name)

        if config.output_intermediate_results:
            evaluator.print_results(0)

        # Store the results of the current run
        r = evaluator.get_results()
        result_dfs[index] = r
        results_row = list(r.loc['weighted avg', metrics])
        results_list.append(results_row)

    end = perf_counter()

    cols = metrics
    results = pd.DataFrame(columns=cols, data=results_list)
    results.index.name = 'Comb'

    if isinstance(config_variable_tested, str):
        results['Description'] = config_variable_values
    else:
        # noinspection PyTypeChecker
        results['Description'] = [', '.join([f'{n} = {v}' for n, v in zip(config_variable_tested, values)])
                                  for values in config_variable_values]

    # higher = better is assumed
    results = results.sort_values(by=metrics, ascending=False)

    print(spacer)
    print('Results')
    print(spacer, '\n')

    print(f'\n\nBest combinations tested (also stored in gs_{cvs_id}_results.xlsx):\n')
    print(results.head(200).round(5).to_string())
    if len(results) > 0:
        best_comb_index = results.index[0]
        print('\n\nFull result output for the best combination:\n')
        print(result_dfs[best_comb_index].round(5).to_string())
    print('\n\nExecution time: {}'.format(end - start))
    print(spacer, '\n')

    results.round(5).to_excel(f"../logs/cvs_{cvs_id}_results.xlsx")


if __name__ == '__main__':
    test_config_values()

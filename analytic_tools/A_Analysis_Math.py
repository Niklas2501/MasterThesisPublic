import numpy as np
import pandas as pd

from configuration.Enums import A_Variant


def cos_sim(vec_a, vec_b):
    cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a, ord=2) * np.linalg.norm(vec_b, ord=2))
    return cos_sim


def mean_node_wise_cos_sim(a_1, a_2):
    cos_sims_nodes = []

    for k in range(a_1.shape[1]):
        vec_1_k = a_1[:, k]
        vec_2_k = a_2[:, k]

        cos_sims_nodes.append(cos_sim(vec_1_k, vec_2_k))

    return np.mean(cos_sims_nodes)


def mae(a_1, a_2):
    return np.mean(np.abs(a_1 - a_2))


def rmse(a_1, a_2):
    return np.mean(np.power(a_1 - a_2, 2))


def rse(a_1, a_2):
    return np.sqrt(np.sum(np.power(a_1 - a_2, 2)))


def sum_abs_errors(a_1, a_2):
    return np.sum(np.abs(a_1 - a_2))


def get_rounded_value(a_1, a_2, metric: str):
    value = globals()[metric](a_1, a_2)
    return np.round(value, 4)


def fro_norm(a_1, a_2):
    return np.linalg.norm(x=(a_1 - a_2), ord='fro')


def fro_norm_a_wise(a_1, a_2):
    return np.abs(np.linalg.norm(x=(a_1), ord='fro') - np.linalg.norm(x=(a_2), ord='fro'))


def mathematical_comparison(a_1, a_2, desc_1, desc_2):
    mae = get_rounded_value(a_1, a_2, 'mae')
    rmse = get_rounded_value(a_1, a_2, 'rmse')
    rse = get_rounded_value(a_1, a_2, 'rse')
    sum_abs_errors = get_rounded_value(a_1, a_2, 'sum_abs_errors')
    fro_norm = get_rounded_value(a_1, a_2, 'fro_norm')
    fro_norm_a_wise = get_rounded_value(a_1, a_2, 'fro_norm_a_wise')
    mnwcs = get_rounded_value(a_1, a_2, 'mean_node_wise_cos_sim')

    print('Mathematical comparison:')
    print(desc_1, desc_2)
    print()
    print('MAE (element wise metric)', mae)
    print('RMSE (element wise metric)', rmse)
    print('Sum of abs. errors', sum_abs_errors)
    print('Root of sum of squared errors', rse)
    print('Frobenius norm on diff', fro_norm)
    print('Frobenius norm on A\'s (beta)', fro_norm_a_wise)
    print('Mean node wise cos sim', mnwcs)


def get_sim_matrix(a_outs, a_outs_info, include_class_wise, metric):
    rows = []

    for row_i in range(len(a_outs)):
        model, av, desc, head = a_outs_info[row_i]

        # Skip class wise gat outputs if parameter to include those is not set
        if av == A_Variant.A_PRE and desc not in ['AVG', 'None'] and not include_class_wise:
            continue
        else:
            rows.append(row_i)

    sim_matrix = np.zeros(shape=(len(rows), len(rows)))

    for sim_matrix_i, row_i in enumerate(rows):
        for sim_matrix_j, row_j in enumerate(rows):
            # if row_j > row_i:
            #    break

            sim_ij = get_rounded_value(a_outs[row_i], a_outs[row_j], metric)
            sim_matrix[sim_matrix_i, sim_matrix_j] = sim_ij
    rows = [a_outs_info[row] for row in rows]
    rows = [f'{m}_{av}' for m, av, _, _ in rows]
    sim_matrix = pd.DataFrame(data=sim_matrix, index=rows, columns=rows)
    return sim_matrix

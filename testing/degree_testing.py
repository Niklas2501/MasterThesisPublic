import numpy as np


def diff_preprocess_numpy(A):
    degree_size = np.size(A, 0)

    out_degree = np.zeros((degree_size, degree_size))
    in_degree = np.zeros((degree_size, degree_size))

    out_degree_sum = np.array(A.sum(axis=0).flat)
    in_degree_sum = np.array(A.sum(axis=1).flat)

    for j in range(0, degree_size):
        in_degree[j, j] = in_degree_sum[j]

    for i in range(0, degree_size):
        out_degree[i, i] = out_degree_sum[i]

    t_f = np.matmulg(np.linal.matrix_power(out_degree, -1), A)
    t_b = np.matmul(np.linalg.matrix_power(in_degree, -1), A.T)

    return [t_f, t_b]


if __name__ == '__main__':

    A = np.matrix('1, 0.5, 1; 1, 0, 0,; 0, 0, 1')

    print(A)
    A_T = A.T
    # print(A)

    degree_size = np.size(A, 0)

    out_degree = np.zeros((degree_size, degree_size))
    in_degree = np.zeros((degree_size, degree_size))

    out_degree_sum = np.array(A.sum(axis=0).flat)
    in_degree_sum = np.array(A.sum(axis=1).flat)

    T_out_degree_sum = np.array(A_T.sum(axis=0).flat)
    T_in_degree_sum = np.array(A_T.sum(axis=1).flat)

    for j in range(0, degree_size):
        in_degree[j, j] = in_degree_sum[j]

    for i in range(0, degree_size):
        out_degree[i, i] = out_degree_sum[i]

    degrees = np.diag(np.array(A.sum(axis=1)).flatten())

    print('D_out')
    print(out_degree)

    print('D_in')
    print(in_degree)

    print('Spektral D')
    print(degrees)

    print('Forward by 016')
    print(np.matmul(np.linalg.matrix_power(out_degree, -1), A))

    print('Forward normed by 003')
    print((A.T / out_degree_sum).T)

    print('Backward by 016')
    print(np.matmul(np.linalg.matrix_power(in_degree, -1), A.T))

    print('Row normed by 003')
    print((A / in_degree_sum).T)

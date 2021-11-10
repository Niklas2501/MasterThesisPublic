from pathlib import Path

import pandas as pd


def main():
    cols = ['F1', 'F2', 'Recall', 'Precision']
    test = ['main', 'snn', 'ablation_study']
    mode = test[2]
    sort_by = 'F1'  # cols[0]

    print_latex = True
    diff_to_first = True

    if mode == 'main':
        file_path = Path('../logs_archive/_results/final_test_results.xlsx', )
        group_by = ["Comb", "File name"]
    elif mode == 'snn':
        file_path = Path('../logs_archive/_results/snn_test_results.xlsx', )
        group_by = ["Comb"]
    elif mode == 'ablation_study':
        file_path = Path('../logs_archive/_results_ablation_study/ablation_study_results.xlsx')
        group_by = ["Comb", "File name"]
    else:
        raise ValueError()

    df = pd.read_excel(file_path, engine='openpyxl')

    for col in cols:
        df[col] = pd.to_numeric(df[col])

    if mode == 'ablation_study':
        # Add full model from main test
        df.loc[-1] = ['-1', -1, 'Baseline', 0.9258, 0.9330, 0.9387, 0.9245]

    df = df.groupby(by=group_by).mean()
    df = df.drop(['Seed', 'Notes'], axis=1, errors='ignore')
    df = df.round(4)
    df = df.sort_values(by=sort_by, ascending=sort_by == 'Comb')

    print('Sorted by:', sort_by)
    print(df.to_string(), '\n')

    if print_latex:
        df = df.sort_values(by=['Comb'], ascending=sort_by == 'Comb')
        print(df.to_latex(), '\n')

    if diff_to_first:
        df = df[['F1', 'F2']]
        df = df.sort_values(by=sort_by, ascending=sort_by == 'Comb')
        best_comb_index = df.index[0]

        for col in ['F1', 'F2']:
            x = df.loc[best_comb_index, col]
            df['Delta ' + col] = df[col] - x

        df = df[['F1', 'Delta F1', 'F2', 'Delta F2']]

        print('Sorted by:', sort_by)
        print(df.to_string(), '\n')

        if print_latex:
            print(df.to_latex(), '\n')

    if sort_by != 'F1':
        print('WARNING: Not sorted by F1')


if __name__ == '__main__':
    main()
